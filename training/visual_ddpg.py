"""End-to-end RGB HJ-DDPG using the gap estimator's ResNet-18 backbone."""
import json
import math
import os
import time
from copy import deepcopy
from dataclasses import asdict, replace
from pathlib import Path

import torch
from torch.nn import functional as F

from .config import load_data, setup_device
from .train_ddpg import ACTION_CONVENTION, OBJECTIVE, MILESTONES, describe, hj_target, save_checkpoint
from .visual_dataset import ImageReplay
from .visual_models import ARCHITECTURE, PREPROCESS, ResNetActor, ResNetCritic
from .visual_algorithms import LearnerOptions, ControllerBC, HJTD3BC, HJSAC


@torch.no_grad()
def soft_update_visual(target, online, tau):
    # ResNet BatchNorm running statistics and integer counters are state too.
    source_state = online.state_dict()
    for name,destination in target.state_dict().items():
        source = source_state[name]
        if destination.is_floating_point():
            destination.lerp_(source,tau)
        else:
            destination.copy_(source)


class VisualHJDDPG:
    def __init__(self,cfg,device,pretrained=True):
        self.actor = ResNetActor(cfg.action_low,cfg.action_high,pretrained,freeze_batch_norm=True,history_frames=cfg.visual_history_frames).to(device)
        self.critic = ResNetCritic(cfg.hidden_sizes,pretrained,freeze_batch_norm=True,history_frames=cfg.visual_history_frames).to(device)
        self.target_actor = deepcopy(self.actor).eval().requires_grad_(False)
        self.target_critic = deepcopy(self.critic).eval().requires_grad_(False)
        self.actor_optimizer = torch.optim.AdamW(self.actor.parameters(),lr=cfg.actor_lr)
        self.critic_optimizer = torch.optim.AdamW(self.critic.parameters(),lr=cfg.critic_lr)
        self.gamma,self.tau = cfg.gamma,cfg.tau
        self.actor_updated = True

    def update(self,batch):
        image,action,successor,h,nh,failed = batch
        self.actor.train(); self.critic.train()
        with torch.no_grad():
            nq = self.target_critic(successor,self.target_actor(successor))
            target = hj_target(h,nh,failed,nq,self.gamma)
        self.critic_optimizer.zero_grad(set_to_none=True)
        loss = F.mse_loss(self.critic(image,action),target)
        loss.backward()
        self.critic_optimizer.step()
        # Freeze critic weights AND its BatchNorm buffers during actor ascent.
        self.critic.requires_grad_(False).eval()
        self.actor_optimizer.zero_grad(set_to_none=True)
        actor_loss = -self.critic(image,self.actor(image)).mean()
        actor_loss.backward()
        self.actor_optimizer.step()
        self.critic.requires_grad_(True).train()
        soft_update_visual(self.target_actor,self.actor,self.tau)
        soft_update_visual(self.target_critic,self.critic,self.tau)
        return loss.detach(),actor_loss.detach()


@torch.inference_mode()
def diagnostics(trainer,replay,indices,batch_size,device):
    trainer.actor.eval()
    if trainer.critic is not None: trainer.critic.eval()
    if getattr(trainer,"critic2",None) is not None: trainer.critic2.eval()
    outputs,qa,qu = [],[],[]
    steering_span = None
    for ids in indices.split(batch_size):
        image,u,*_ = replay.batch(ids)
        image,u = image.to(device),u.to(device)
        action = trainer.actor(image)
        outputs.append(action.cpu())
        if trainer.critic is not None:
            aq,lq = trainer.critic(image,action),trainer.critic(image,u)
            if getattr(trainer,'critic2',None) is not None:
                aq = torch.minimum(aq,trainer.critic2(image,action))
                lq = torch.minimum(lq,trainer.critic2(image,u))
            qa.append(aq.cpu()); qu.append(lq.cpu())
            if steering_span is None:
                values=[]
                for steering in (-1.,0.,1.):
                    command=u.clone();command[:,1]=steering
                    value=trainer.critic(image,command)
                    if getattr(trainer,'critic2',None) is not None:
                        value=torch.minimum(value,trainer.critic2(image,command))
                    values.append(value)
                grid=torch.stack(values,1)
                steering_span=float((grid.max(1).values-grid.min(1).values).median())
    actions = torch.cat(outputs)
    result = dict(longitudinal=describe(actions[:,0]),steering=describe(actions[:,1]),
                braking_fraction=float((actions[:,0]<0).float().mean()),
                abs_steer_ge_095_fraction=float((actions[:,1].abs()>=.95).float().mean()),
                abs_long_ge_285_fraction=float((actions[:,0].abs()>=2.85).float().mean()),
                imitation_eligible_fraction=float(replay.bc_mask[indices].float().mean()))
    mask = replay.bc_mask[indices]
    if mask.any():
        scale = torch.tensor([3.,1.])
        result['validation_bc_mse'] = float(((actions[mask]-replay.bc_target[indices][mask])/scale).square().mean())
    if qa:
        result.update(q_actor_m=describe(torch.cat(qa)),q_logged_m=describe(torch.cat(qu)),
                      validation_steering_q_span_median_m=steering_span)
        logged_q = torch.cat(qu)
        margin = replay.margin[indices]
        result['current_margin_rmse_m'] = float((logged_q - margin).square().mean().sqrt())
        correlation = torch.corrcoef(torch.stack((logged_q, margin)))[0, 1]
        result['current_margin_correlation'] = float(correlation) if torch.isfinite(correlation) else None
    return result


def train_visual(data_path,cfg,output,selection='combined',resume=None,require_cuda=False,
                 checkpoint_interval=None,image_root=None,pretrained=True,validation_probes=256,
                 learner_options=None,actor_init=None):
    options = (learner_options or LearnerOptions()).validate()
    if not cfg.safety_buffer_m < options.teacher_gap < cfg.max_gap_m:
        raise ValueError("Teacher gap must lie inside the configured bounds")
    if cfg.updates<1 or cfg.batch_size<1 or validation_probes<1 or not 0<=cfg.gamma<=1 or not 0<cfg.tau<=1:
        raise ValueError('Check updates, batch size, validation probes, gamma and tau')
    if checkpoint_interval is not None and checkpoint_interval<1:
        raise ValueError('checkpoint_interval must be positive')
    if selection not in ('combined','new'):
        raise ValueError('Choose combined or new data')
    output = Path(output)
    if output.exists() and resume is None:
        raise FileExistsError(output)
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
    os.environ.setdefault('TORCH_HOME',str(Path(__file__).resolve().parents[1]/'.local'/'torch'))
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(min(4,torch.get_num_threads()))
    device = setup_device(cfg.seed)
    if require_cuda and device.type != 'cuda':
        raise RuntimeError('CUDA requested but unavailable')
    cfg = replace(cfg,dataset_selection=selection)
    data = load_data(data_path)
    train_ids = [i for i in data['splits']['train'] if selection=='combined' or data['sources'][i]=='new']
    replay = ImageReplay(data,train_ids,cfg,image_root,teacher_options=options)
    validation = ImageReplay(data,data['splits']['val'],cfg,image_root,teacher_options=options)
    if options.algorithm in ('bc', 'hj-td3-bc') and (not replay.bc_mask.any() or not validation.bc_mask.any()):
        raise ValueError("Training and validation need eligible imitation labels")
    probes = torch.randperm(len(validation),generator=torch.Generator().manual_seed(42))[:validation_probes]
    if options.algorithm == 'ddpg':
        trainer = VisualHJDDPG(cfg,device,pretrained=pretrained and resume is None)
        state_keys = ('actor','critic','target_actor','target_critic','actor_optimizer','critic_optimizer')
    else:
        learner_class = {'bc': ControllerBC, 'hj-td3-bc': HJTD3BC, 'hj-sac': HJSAC}[options.algorithm]
        trainer = learner_class(cfg,device,pretrained and resume is None,options)
        state_keys = trainer.state_keys
    if actor_init:
        if resume:
            raise ValueError('Choose resume or actor initialization, not both')
        initial = torch.load(actor_init,map_location='cpu',weights_only=True)
        if (initial.get('architecture') != ARCHITECTURE or initial.get('algorithm') != 'bc' or
                (initial['dataset_id'],initial['split_id']) != (data['dataset_id'],data['split_id']) or
                initial.get('preprocessing') != PREPROCESS):
            raise ValueError('Actor initialization requires a BC checkpoint from this dataset/split')
        if initial['settings'].get('visual_history_frames',1) != cfg.visual_history_frames:
            raise ValueError('Actor initialization history length differs; start a fresh temporal actor')
        for key in ('action_low','action_high'):
            if tuple(initial['settings'][key]) != tuple(asdict(cfg)[key]):
                raise ValueError('Actor initialization action bounds differ')
        trainer.actor.load_state_dict(initial['actor'])
        if hasattr(trainer,'target_actor'): trainer.target_actor.load_state_dict(initial['actor'])
    sampler = torch.Generator().manual_seed(cfg.seed)
    history,start_step,elapsed_before = [],0,0.
    initialization = "ImageNet DEFAULT" if pretrained else "random"
    if resume:
        saved = torch.load(resume,map_location='cpu',weights_only=True)
        if (saved.get('architecture') != ARCHITECTURE or saved.get('format_version') != 4 or
                saved.get('objective') != ('controller_imitation' if options.algorithm == 'bc' else OBJECTIVE) or saved.get('action_convention') != ACTION_CONVENTION):
            raise ValueError('Resume requires a format-4 checkpoint; old runs need a fresh experiment')
        if saved['settings'].get('visual_history_frames',1) != cfg.visual_history_frames:
            raise ValueError('Resume history length differs')
        saved_options = asdict(LearnerOptions(**saved.get('learner_options', {})))
        if saved_options != asdict(options):
            raise ValueError('Resume learner options differ')
        if (saved['dataset_id'],saved['split_id'],saved['dataset_selection']) != (data['dataset_id'],data['split_id'],selection):
            raise ValueError('Dataset or split differs from checkpoint')
        if saved['preprocessing'] != PREPROCESS:
            raise ValueError('Image preprocessing differs')
        for key,value in saved['settings'].items():
            if key != 'updates' and asdict(cfg)[key] != value:
                raise ValueError(f'Resume settings differ: {key}')
        if cfg.updates <= saved['step']:
            raise ValueError('--updates must exceed the resumed step')
        if output.exists() and any(int(p.stem.split('_')[-1])>saved['step'] for p in output.glob('checkpoint_*.pt')):
            raise FileExistsError('Later checkpoints already exist; use a new run directory')
        for key in state_keys:
            getattr(trainer,key).load_state_dict(saved[key])
        sampler.set_state(saved['sampler_state'])
        torch.set_rng_state(saved['torch_rng_state'])
        if device.type == 'cuda' and saved.get('cuda_rng_state') is not None:
            torch.cuda.set_rng_state(saved['cuda_rng_state'],device)
        history,start_step,elapsed_before = saved['history'],saved['step'],saved['elapsed_seconds']
        initialization = saved['initialization']
        trainer.step = saved['step']
    output.mkdir(parents=True,exist_ok=resume is not None)
    critic_losses,actor_losses = [],[]
    latest_actor_loss = 0.
    actor_initialization = str(actor_init) if actor_init else None
    if resume:
        critic_losses = [torch.tensor(v) for v in saved.get('pending_critic_losses',[])]
        actor_losses = [torch.tensor(v) for v in saved.get('pending_actor_losses',[])]
        latest_actor_loss = saved.get('latest_actor_loss',0.)
        actor_initialization = saved.get('actor_initialization')
    started = time.perf_counter()
    warmup_history = saved.get('critic_warmup_history', []) if resume else []
    image_paths = sorted({str(p) for p in replay.paths+validation.paths})
    def save(step):
        elapsed = elapsed_before+time.perf_counter()-started
        payload = dict(format_version=4,architecture=ARCHITECTURE,
                       objective="controller_imitation" if options.algorithm == "bc" else OBJECTIVE,
                       algorithm=options.algorithm,learner_options=asdict(options),normalization="frozen_running_stats",
                       actor_initialization=actor_initialization,
                       pending_critic_losses=[float(v) for v in critic_losses],
                       pending_actor_losses=[float(v) for v in actor_losses],latest_actor_loss=latest_actor_loss,
                       action_convention=ACTION_CONVENTION,preprocessing=PREPROCESS,
                       settings=asdict(cfg),step=step,dataset_id=data['dataset_id'],split_id=data['split_id'],
                       dataset_selection=selection,image_root=str(Path(image_root).resolve()) if image_root else None,
                       image_paths=image_paths,initialization=initialization,
                       sampler_state=sampler.get_state(),torch_rng_state=torch.get_rng_state(),
                       cuda_rng_state=torch.cuda.get_rng_state(device) if device.type=='cuda' else None,
                       history=history,elapsed_seconds=elapsed)
        payload['critic_warmup_history'] = warmup_history
        for key in state_keys:
            payload[key] = getattr(trainer,key).state_dict()
        save_checkpoint(output/f'checkpoint_{step:05d}.pt',payload)
        metrics = {k:v for k,v in payload.items() if k in ('architecture','objective','algorithm','learner_options','settings','step','dataset_id','split_id','history','elapsed_seconds')}
        metrics.update(device=str(device),device_name=torch.cuda.get_device_name(device) if device.type=='cuda' else 'CPU',
                       critic_warmup_history=warmup_history,
                       train_episodes=len(replay.paths),train_transitions=len(replay),validation_transitions=len(validation),
                       imitation_train_rows=int(replay.bc_mask.sum()),imitation_val_rows=int(validation.bc_mask.sum()),
                       validation_probe_count=len(probes),sampling='uniform eligible image transitions with replacement',
                       preprocessing=PREPROCESS,checkpoint_note='ResNet checkpoints include optimizer and target networks and are hundreds of MB')
        temporary = output/'metrics.json.tmp'
        temporary.write_text(json.dumps(metrics,indent=2,allow_nan=False)+'\n')
        temporary.replace(output/'metrics.json')
    if not resume:
        for warmup_step in range(1, options.critic_margin_warmup + 1):
            indices = torch.randint(len(replay), (cfg.batch_size,), generator=sampler)
            batch = tuple(x.to(device) for x in replay.batch(indices))
            loss = trainer.warmup_critic(batch)
            if not torch.isfinite(loss):
                raise FloatingPointError('Nonfinite critic warmup loss')
            if warmup_step % 100 == 0 or warmup_step == options.critic_margin_warmup:
                record = diagnostics(trainer, validation, probes, cfg.batch_size, device)
                record.update(warmup_step=warmup_step, training_batch_margin_mse=float(loss))
                warmup_history.append(record)
                print(f"Critic warmup {warmup_step}: validation margin RMSE "
                      f"{record['current_margin_rmse_m']:.3f} m, correlation "
                      f"{record['current_margin_correlation']}", flush=True)
        if options.critic_margin_warmup:
            trainer.target_critic.load_state_dict(trainer.critic.state_dict())
            trainer.target_critic2.load_state_dict(trainer.critic2.state_dict())
            trainer.critic_optimizer.state.clear()
        save(0)
    print(f'RGB replay: {len(replay):,} train / {len(validation):,} validation transitions; batch {cfg.batch_size}',flush=True)
    for step in range(start_step+1,cfg.updates+1):
        indices = torch.randint(len(replay),(cfg.batch_size,),generator=sampler)
        batch = tuple(x.to(device) for x in replay.batch(indices))
        if options.algorithm in ('ddpg', 'hj-sac'):
            critic_loss,actor_loss = trainer.update(batch)
        else:
            critic_loss,actor_loss = trainer.update(batch,replay.bc_target[indices].to(device),replay.bc_mask[indices].to(device))
        critic_losses.append(critic_loss.cpu())
        if trainer.actor_updated: actor_losses.append(actor_loss.cpu())
        if step==1 or step%50==0 or step==cfg.updates:
            critic_loss = float(torch.stack(critic_losses).mean()); critic_losses.clear()
            if actor_losses:
                latest_actor_loss = float(torch.stack(actor_losses).mean()); actor_losses.clear()
            actor_loss = latest_actor_loss
            if not all(math.isfinite(v) for v in (critic_loss,actor_loss)):
                raise FloatingPointError('Nonfinite visual training loss')
            record = diagnostics(trainer,validation,probes,cfg.batch_size,device)
            record.update(step=step,critic_mse=critic_loss,actor_loss=actor_loss)
            record.update(getattr(trainer,'last_metrics',{}))
            json.dumps(record,allow_nan=False)
            history.append(record)
            if options.algorithm == 'bc':
                print(f'{step:5d} imitation loss {actor_loss:.5f} validation imitation MSE {record.get("validation_bc_mse",0.):.5f}',flush=True)
            else:
                print(f'{step:5d} critic MSE {critic_loss:.5f} actor loss {actor_loss:.5f} validation braking {record["braking_fraction"]:.1%}',flush=True)
        if step in MILESTONES or step==cfg.updates or (checkpoint_interval and step%checkpoint_interval==0):
            save(step)
    return output/f'checkpoint_{cfg.updates:05d}.pt'


@torch.inference_mode()
def evaluate_visual(args):
    from .config import Settings
    from .evaluate_ddpg import action_stats,distribution
    from .visual_models import RGBSafetyFilter
    data = load_data(args.data)
    saved = torch.load(args.checkpoint,map_location='cpu',weights_only=True)
    if (saved['dataset_id'],saved['split_id']) != (data['dataset_id'],data['split_id']):
        raise ValueError('Dataset or split differs from checkpoint')
    device = setup_device(42)
    torch.set_num_threads(4)
    policy = RGBSafetyFilter(saved,device)
    cfg = Settings(**saved['settings'])
    root = args.image_root or saved.get('image_root')
    options = LearnerOptions(**saved.get('learner_options',{}))
    replay = ImageReplay(data,data['splits'][args.split],cfg,root,teacher_options=options)
    actions,recorded,qa,qu = [],[],[],[]
    for ids in torch.arange(len(replay)).split(args.batch_size):
        image,u,*_ = replay.batch(ids)
        image,u = image.to(device),u.to(device)
        action = policy.actor(image)
        actions.append(action.cpu());recorded.append(u.cpu())
        if policy.has_safety_critic:
            qa.append(policy.conservative_q(image,action).cpu());qu.append(policy.conservative_q(image,u).cpu())
    actions,recorded = torch.cat(actions),torch.cat(recorded)
    if not all(t.isfinite().all() for t in [actions,recorded,*qa,*qu]):
        raise FloatingPointError('Nonfinite visual evaluation')
    result = dict(architecture=ARCHITECTURE,dataset_id=data['dataset_id'],split_id=data['split_id'],
                  split=args.split,step=saved['step'],actor=action_stats(actions),recorded=action_stats(recorded),
                  algorithm=saved.get('algorithm','ddpg'),
                  imitation_eligible_rows=int(replay.bc_mask.sum()),
                  interpretation='Offline RGB policy diagnostics on recorded frames; not a closed-loop safety certificate')
    if replay.bc_mask.any():
        result['imitation_normalized_mse'] = float(((actions[replay.bc_mask]-replay.bc_target[replay.bc_mask])/torch.tensor([3.,1.])).square().mean())
    if qa:
        result.update(q_actor_m=distribution(torch.cat(qa)),q_recorded_m=distribution(torch.cat(qu)))
    args.output.mkdir(parents=True,exist_ok=True)
    (args.output/'evaluation.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(args.output,result['actor'],flush=True)
