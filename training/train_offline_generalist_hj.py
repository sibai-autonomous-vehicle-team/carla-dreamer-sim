"""Pooled nine-condition offline HJ-SAC, using a frozen generalist RGB backbone.

Twin action critics train from logged actions and successor images. By default,
their continuation policy is fixed full braking. --backup-policy actor enables
the legacy learned-actor HJ-SAC experiment explicitly.
Frozen CNN features are cached once; no online transitions or imitation labels
are used. The output loads directly into OnlineHJ and its paired CARLA benchmark.
Offline SAC can overestimate unseen actions; held-out CARLA evaluation is required.
"""
import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import random
import time

import torch
from torch import nn
from torch.nn import functional as F

from .train_online_hj import GeneralistPID, OnlineHJ, append_json, hj_backup, parse_args as online_args
from .visual_algorithms import soft_update
from .visual_models import PREPROCESS


CONDITIONS = tuple(f'{t}_{w}' for t in ('day','sunset','night') for w in ('sunny','cloudy','rainy'))


def transition_rows(actions, distances, costs, *, metadata=None, margin_cap=9., failure_margin=1.):
    """Return image rows, commands, aligned margins, and collision flags.

    Original images/states precede actions, distances/costs follow them. Without
    terminal images, only interior transitions 1..T-2 are usable. Braking metadata
    supplies initial gap and terminal image, retaining transitions 0..T-1.
    """
    length = len(actions)
    if distances.shape != (length,) or costs.shape != (length,):
        raise ValueError('Action/gap/cost lengths disagree')
    if actions.ndim != 2 or actions.shape[1] != 2:
        raise ValueError('Expected logged longitudinal and steering requests')
    if not all(torch.isfinite(x).all() for x in (actions,distances,costs)):
        raise ValueError('Nonfinite recorded transition')
    if (distances<0).any() or not ((costs==0)|(costs==1)).all():
        raise ValueError('Invalid unsigned gaps or collision flags')
    failed = costs.bool().clone()
    start, stop = 1, length-1
    if metadata is not None:
        frames, timestamps = metadata['frames'],metadata['timestamps']
        if len(frames)!=length+1 or not ((frames[1:]-frames[:-1])==1).all():
            raise ValueError('Nonconsecutive braking frame IDs')
        if not math.isclose(metadata['dt'],.1) or not torch.allclose(
                timestamps[1:]-timestamps[:-1],torch.full_like(timestamps[1:],.1),atol=1e-4,rtol=0):
            raise ValueError('Unexpected braking timestep')
        start, stop = 0, length
        # Events are transition outcomes, not state-time labels.
        for frame, _ in metadata['collision_events']:
            failed |= (frames[:-1]<frame)&(frames[1:]>=frame)
        gaps = torch.cat((metadata['initial_gap'].reshape(1),distances))
        h, nh = gaps[:-1],gaps[1:]
        departures = metadata.get('lane_departure_frames',[])
        if departures:
            ends = (frames[1:]>=min(departures)).nonzero().flatten()
            if len(ends): stop = min(stop,int(ends[0])+1)
    else:
        h = torch.cat((distances[:1],distances[:-1]))
        nh = distances
    hits = failed.nonzero().flatten()
    if len(hits): stop = min(stop,int(hits[0])+1)
    rows = torch.arange(start,max(start,stop))
    h = h.clamp(max=margin_cap)
    nh = nh.clamp(max=margin_cap)
    nh = torch.where(failed,torch.full_like(nh,-failure_margin),nh)
    return rows,actions[rows,0:1].clamp(-3,3),h[rows],nh[rows],failed[rows]


def build_condition(condition, cli, encoder, device):
    """Cache independent episode histories and retain the CNN's episode split."""
    cache = (cli.cache_root if cli.cache_root is not None else cli.outdir/'cache')/f'{condition}.pt'
    if cli.cache_root is not None and not cache.is_file():
        raise FileNotFoundError(f'Missing requested cache: {cache}')
    split_path = cli.data_root/condition/'split.json'
    split = json.loads(split_path.read_text())
    selected = {i:part for part in ('train','val') for i in split[f'{part}_ids']}
    if set(split['train_ids'])&set(split['val_ids']) or set(selected)&set(split['test_ids']):
        raise ValueError('Overlapping train/validation/test episodes')
    fingerprint = hashlib.sha256(split_path.read_bytes()+cli.cnn_sha.encode()).hexdigest()
    if cache.exists():
        data = torch.load(cache,map_location='cpu',weights_only=True)
        if data['fingerprint'] != fingerprint: raise ValueError('Cache configuration mismatch')
        return data
    features, pending, pending_count = [],[],0
    refs, commands, margins, next_margins, failures, partitions = [],[],[],[],[],[]
    frame_offset = 0
    counts = Counter()

    def flush():
        nonlocal pending_count
        if not pending: return
        images = torch.cat(pending)
        with torch.inference_mode():
            for frames in images.split(cli.encode_batch):
                # FP32 matches inference; cached FP32 avoids a representation shift.
                features.append(encoder.network(encoder.preprocess(frames.to(device))).cpu())
        pending.clear()
        pending_count = 0

    for source, root, offset, expected in [('original',cli.original_root,0,800),('braking',cli.braking_root,800,500)]:
        folder = root/condition
        columns = {k:torch.load(folder/f'{k}.pth',map_location='cpu',weights_only=True)
                   for k in ('actions','distances','costs','seq_lengths')}
        if len(columns['seq_lengths']) != expected: raise ValueError(f'Unexpected episode count: {folder}')
        for episode, length in enumerate(columns['seq_lengths'].tolist()):
            merged = offset+episode
            if merged not in selected: continue
            path = folder/'obses'/f'episode_{episode:03d}.pth'
            images = torch.load(path,map_location='cpu',weights_only=True,mmap=True)
            if images.dtype!=torch.uint8 or images.shape!=(length,224,224,3):
                raise ValueError(f'Invalid episode images: {path}')
            metadata = None
            if source == 'braking':
                metadata = torch.load(folder/'metadata'/f'episode_{episode:03d}.pth',weights_only=True)
                images = torch.cat((images,metadata['terminal_image'][None]))
            rows, action, h, nh, failed = transition_rows(columns['actions'][episode],
                columns['distances'][episode],columns['costs'][episode],metadata=metadata)
            if not len(rows): continue
            current = (rows[:,None]+torch.arange(-3,1)[None]).clamp_min(0)+frame_offset
            successor = ((rows+1)[:,None]+torch.arange(-3,1)[None]).clamp_min(0)+frame_offset
            refs.append(torch.stack((current,successor),1))
            commands.append(action); margins.append(h); next_margins.append(nh); failures.append(failed)
            partitions.append(torch.full((len(rows),),selected[merged]=='val',dtype=torch.bool))
            pending.append(images); pending_count += len(images)
            frame_offset += len(images)
            counts[f'{source}_{selected[merged]}_episodes'] += 1
            counts[f'{source}_{selected[merged]}_transitions'] += len(rows)
            if pending_count >= cli.encode_batch: flush()
            if (episode+1)%100 == 0:
                print(json.dumps(dict(phase='encoding',condition=condition,source=source,episode=episode+1,
                                      encoded_frames=frame_offset)),flush=True)
    flush()
    data = dict(features=torch.cat(features),refs=torch.cat(refs),action=torch.cat(commands),
                margin=torch.cat(margins),next_margin=torch.cat(next_margins),failed=torch.cat(failures),
                validation=torch.cat(partitions),fingerprint=fingerprint,coverage=dict(counts))
    if data['features'].shape!=(frame_offset,512): raise ValueError('Feature-cache indexing mismatch')
    temporary = cache.with_suffix('.tmp')
    torch.save(data,temporary); temporary.replace(cache)
    print(json.dumps(dict(phase='cached',condition=condition,coverage=dict(counts))),flush=True)
    return data


class FeatureReplay:
    def __init__(self, data):
        self.data = data
        self.indices = {part:[torch.where(d['validation']==validation)[0] for d in data]
                        for part,validation in [('train',False),('val',True)]}
        if any(not len(ids) for values in self.indices.values() for ids in values):
            raise ValueError('Every condition needs nonempty train and validation transitions')

    def sample(self, count, part, device, generator=None):
        # Equal expected condition exposure; no large condition dominates replay.
        conditions = torch.randint(len(self.data),(count,),generator=generator)
        batches = []
        for condition,d in enumerate(self.data):
            size = int((conditions==condition).sum())
            if not size: continue
            ids = self.indices[part][condition]
            rows = ids[torch.randint(len(ids),(size,),generator=generator)]
            refs = d['refs'][rows]
            current = d['features'][refs[:,0]].flatten(1)
            successor = d['features'][refs[:,1]].flatten(1)
            batches.append((current,d['action'][rows],successor,d['margin'][rows],
                            d['next_margin'][rows],d['failed'][rows]))
        return tuple(torch.cat(parts).to(device) for parts in zip(*batches))


def feature_target(learner, batch):
    current, action, successor, h, nh, failed = batch
    gamma = learner.schedule(learner.args.gamma_start,learner.args.gamma_final)
    low,high = learner.safety_bounds
    if not ((h>=low)&(h<=high)&(nh>=low)&(nh<=high)).all():
        raise ValueError('Safety labels outside the declared collision-value bounds')
    with torch.no_grad():
        if getattr(learner.args,'offline_backup_policy','actor') == 'full-brake':
            next_action=torch.full_like(action,-3.)
        else:
            next_action,_ = learner.actor.sample_features(successor)
        continuation = torch.minimum(*(q.value(successor,next_action) for q in learner.targets))
        return hj_backup(h,nh,failed,continuation,gamma).clamp(*learner.safety_bounds)


def update_features(learner, batch):
    features,action,*_ = batch
    gamma = learner.schedule(learner.args.gamma_start,learner.args.gamma_final)
    target = feature_target(learner,batch)
    values = [q.value(features,action) for q in learner.critics]
    loss = sum(F.mse_loss(q,target) for q in values)/2
    if not torch.isfinite(loss): raise FloatingPointError('Nonfinite offline critic loss')
    learner.critic_optimizer.zero_grad(set_to_none=True)
    loss.backward()
    nn.utils.clip_grad_norm_(learner.critics.parameters(),10.)
    learner.critic_optimizer.step()
    fixed_brake=getattr(learner.args,'offline_backup_policy','actor') == 'full-brake'
    if fixed_brake:
        proposed=torch.full_like(action,-3.)
        actor_loss=torch.zeros((),device=action.device)
        alpha=0.
    else:
        parameters = [p for p in learner.critics.parameters() if p.requires_grad]
        for p in parameters: p.requires_grad_(False)
        try:
            proposed,log_prob = learner.actor.sample_features(features)
            q = torch.minimum(*(c.value(features,proposed) for c in learner.critics))
            alpha = learner.schedule(learner.args.alpha,learner.args.alpha_final)
            actor_loss = (alpha*log_prob-q).mean()
            if not torch.isfinite(actor_loss): raise FloatingPointError('Nonfinite offline actor loss')
            learner.actor_optimizer.zero_grad(set_to_none=True)
            actor_loss.backward()
            nn.utils.clip_grad_norm_(learner.actor.parameters(),10.)
            learner.actor_optimizer.step()
        finally:
            for p in parameters: p.requires_grad_(True)
    for target_q,q in zip(learner.targets,learner.critics):
        for name in ('state_head','action_head','head'):
            soft_update(getattr(target_q,name),getattr(q,name),learner.args.tau)
    learner.updates += 1
    return dict(update=learner.updates,gamma=gamma,alpha=alpha,actor_updated=not fixed_brake,
                critic_mse=float(loss.detach()),actor_loss=float(actor_loss.detach()),
                q_mean=float(torch.minimum(*values).detach().mean()),
                mean_action=float(proposed.detach().mean()),braking_fraction=float((proposed.detach()<0).float().mean()))


def offline_anneal_updates(total_updates, requested=None):
    """Leave a fixed-objective phase instead of stopping at the annealing boundary."""
    if total_updates < 1:
        raise ValueError('updates must be positive')
    count = min(5000,total_updates//4) if requested is None else requested
    if not 0 <= count < total_updates:
        raise ValueError('anneal-updates must be nonnegative and smaller than updates')
    return count


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--outdir',type=Path,required=True)
    p.add_argument('--cnn',type=Path,default=Path('experiments/generalist_1300_2026-11-24/training/best.pth'))
    p.add_argument('--data-root',type=Path,default=Path('experiments/specialists_braking_1300_2026-11-23/data'))
    p.add_argument('--original-root',type=Path,default=Path('dataset/combined_800'))
    p.add_argument('--braking-root',type=Path,default=Path('dataset/braking_lane_tracked_500_per_condition'))
    p.add_argument('--updates',type=int,default=20000)
    p.add_argument('--backup-policy',choices=('full-brake','actor'),default='full-brake')
    p.add_argument('--batch-size',type=int,default=256)
    p.add_argument('--encode-batch',type=int,default=256)
    p.add_argument('--cache-root',type=Path,help='Reuse caches built for the same dataset splits and CNN')
    p.add_argument('--anneal-updates',type=int,help='Default: min(5000, updates/4); 0 trains at final settings from the start')
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--require-cuda',action='store_true')
    cli = p.parse_args()
    if min(cli.updates,cli.batch_size,cli.encode_batch)<1: p.error('Counts must be positive')
    try:
        anneal_updates = offline_anneal_updates(cli.updates,cli.anneal_updates)
    except ValueError as error:
        p.error(str(error))
    if cli.require_cuda and not torch.cuda.is_available(): raise RuntimeError('CUDA unavailable')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    random.seed(cli.seed); torch.manual_seed(cli.seed); torch.set_num_threads(4)
    cli.outdir.mkdir(parents=True,exist_ok=False)
    (cli.outdir/'cache').mkdir()
    nominal = GeneralistPID(cli.cnn,device)
    cli.cnn_sha = nominal.sha256
    # Reuse the trained generalist representation, frozen identically in all networks.
    encoder = nominal.encoder
    encoder.network.fc = nn.Identity()
    args = online_args(argv=['--outdir',str(cli.outdir),'--pid-checkpoint',str(cli.cnn),
                            '--freeze-encoder','--no-pretrained'])
    args.batch_size, args.seed = cli.batch_size, cli.seed
    args.anneal_updates = anneal_updates
    args.offline_trainer_revision = 3
    args.offline_backup_policy = cli.backup_policy
    args.offline_conditions = list(CONDITIONS)
    args.offline_data_root = str(cli.data_root.resolve())
    learner = OnlineHJ(args,device)
    if cli.backup_policy == 'full-brake':learner.actor.requires_grad_(False)
    weights = encoder.network.state_dict()
    for model in [learner.actor,*learner.critics,*learner.targets]:
        model.network.load_state_dict(weights)
        model.eval()
    manifest = dict(conditions=list(CONDITIONS),updates=cli.updates,batch_size=cli.batch_size,
        trainer_revision=3,backup_policy=cli.backup_policy,actor_trained=cli.backup_policy=='actor',
        anneal_updates=anneal_updates,final_objective_updates=cli.updates-anneal_updates,
        cache_root=str(cli.cache_root.resolve()) if cli.cache_root is not None else None,
        seed=cli.seed,cnn_sha256=cli.cnn_sha,preprocessing=PREPROCESS,
        representation='frozen generalist gap CNN; FP32 cached frame features; four-frame history',
        objective=('collision-only discounted HJ value under full-brake continuation; twin-min critics; no learned backup actor'
                   if cli.backup_policy=='full-brake' else
                   'collision-only discounted HJ-SAC; twin-min critics; no BC or online experience'),
        sampling='uniform conditions, then uniform training transitions',
        split='existing generalist episode IDs; train updates only; validation diagnostics only; test untouched',
        original_alignment='RGB[t], action[t], gap[t-1] -> RGB[t+1], gap[t]; t=1..T-2',
        braking_alignment='all transitions, with initial gap and terminal image from metadata; exact event frames',
        caveats=['Offline actor actions may be out of distribution; Bellman residual is not safety accuracy.',
                 'Logged steering differs by collector; learned control is longitudinal, evaluation uses fixed route steering.',
                 'Episode splits do not establish route-disjoint or condition-disjoint generalization.'])
    (cli.outdir/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    start = time.monotonic()
    try:
        data = [build_condition(c,cli,encoder,device) for c in CONDITIONS]
        replay = FeatureReplay(data)
        manifest['coverage'] = {c:d['coverage'] for c,d in zip(CONDITIONS,data)}
        (cli.outdir/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
        # A local CPU generator leaves both CPU and CUDA training streams intact.
        validation = replay.sample(1024,'val',device,
                                  generator=torch.Generator().manual_seed(cli.seed+1))
        for step in range(1,cli.updates+1):
            metrics = update_features(learner,replay.sample(cli.batch_size,'train',device))
            if step%100 == 0 or step == 1:
                metrics['seconds'] = time.monotonic()-start
                append_json(cli.outdir/'updates.jsonl',metrics)
                print(json.dumps(metrics),flush=True)
            if step%1000 == 0 or step == cli.updates:
                with torch.no_grad(),torch.random.fork_rng():
                    torch.manual_seed(cli.seed+2)
                    target = feature_target(learner,validation)
                    residual = sum(F.mse_loss(q.value(validation[0],validation[1]),target)
                                   for q in learner.critics)/2
                append_json(cli.outdir/'validation.jsonl',dict(update=step,bellman_mse=float(residual)))
                learner.save(cli.outdir/'checkpoint_latest.pt',0)
        learner.save(cli.outdir/'checkpoint_final.pt',0)
        (cli.outdir/'.complete').touch()
    except BaseException:
        learner.save(cli.outdir/'checkpoint_interrupted.pt',0)
        raise


if __name__ == '__main__':
    main()
