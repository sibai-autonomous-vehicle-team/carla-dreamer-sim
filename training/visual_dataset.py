"""Uniform offline transition replay backed by memory-mapped RGB episode files."""
import json
import math
from collections import OrderedDict
from pathlib import Path

import torch

from .dataset import episode_slices, first_failure, safety_margin, transition_window


class ImageReplay:
    def __init__(self, data, ids, cfg, image_root=None, cache_episodes=16, teacher_options=None):
        from .visual_algorithms import LearnerOptions
        options = (teacher_options or LearnerOptions()).validate()
        if cfg.gap_offset > 0 or cfg.dt <= 0:
            raise ValueError('Invalid temporal alignment')
        self.history_frames = cfg.visual_history_frames
        if self.history_frames < 1:
            raise ValueError("visual_history_frames must be positive")
        self.frames = OrderedDict()
        self.cache_episodes = cache_episodes
        if cache_episodes < 1:
            raise ValueError('cache_episodes must be positive')
        self.paths, self.lengths, refs = [], [], []
        actions, margins, next_margins, failed = [], [], [], []
        bc_targets,bc_masks = [],[]
        slices = episode_slices(data)
        root = Path(image_root).resolve() if image_root else None
        manifest_path = root/'merge_manifest.json' if root else None
        if root and not manifest_path.exists():
            manifest_path = root.parent/'merge_manifest.json'
        manifest = json.loads(manifest_path.read_text()) if root and manifest_path.exists() else None
        offsets, counts, offset = {}, {}, 0
        if manifest:
            for source in manifest['sources']:
                offsets[source['folder']] = offset
                counts[source['folder']] = source['count_per_condition']
                offset += source['count_per_condition']
        checked = {}
        for episode in ids:
            sl = slices[episode]
            length = int(data['lengths'][episode])
            gap = data['distances'][sl]
            hit = first_failure(safety_margin(gap,cfg.safety_buffer_m,cfg.max_gap_m))
            start,stop = transition_window(length,hit,cfg)
            if start >= stop:
                continue
            condition,source = data['conditions'][episode],data['sources'][episode]
            local_id = int(data['episode_ids'][episode])
            original = Path(data['source_paths'][source])
            image_id = local_id
            if root:
                folder = root if root.name == condition else root/condition
                if manifest:
                    name = original.name
                    if name not in offsets or not 0 <= local_id < counts[name]:
                        raise ValueError('Cannot match cache source to combined-image manifest')
                    image_id += offsets[name]
                elif len(set(data['sources'])) > 1:
                    raise ValueError('A merged image root requires merge_manifest.json for source indexing')
            else:
                folder = original if original.name == condition else original/condition
            # Verify image-folder metadata against the frozen cache; no silent
            # source swaps, new episode ordering, or recomputed train/test splits.
            if folder not in checked:
                checked[folder] = {k:torch.load(folder/f'{k}.pth',map_location='cpu',weights_only=True)
                                   for k in ('states','actions','distances','costs','seq_lengths')}
            metadata = checked[folder]
            if int(metadata['seq_lengths'][image_id]) != length or any(
                    not torch.equal(metadata[k][image_id],data[k][sl])
                    for k in ('states','actions','distances','costs')):
                raise ValueError(f'Image-folder metadata does not match cache episode: {folder}/{image_id}')
            path = folder/'obses'/f'episode_{image_id:03d}.pth'
            if not path.is_file():
                raise FileNotFoundError(path)
            file_index = len(self.paths)
            self.paths.append(path)
            self.lengths.append(length)
            rows = torch.arange(start,stop)
            refs.append(torch.stack((torch.full_like(rows,file_index),rows),1))
            command = data['actions'][sl][rows+cfg.action_offset].clone()
            command[:,0].clamp_(-cfg.longitudinal_saturation,cfg.longitudinal_saturation)
            h = safety_margin(gap[rows+cfg.gap_offset],cfg.safety_buffer_m,cfg.max_gap_m)
            nh = safety_margin(gap[rows+cfg.gap_offset+1],cfg.safety_buffer_m,cfg.max_gap_m)
            # Labels use only decision-time poses and current/past gaps. They
            # are generated controller targets, not executed expert outcomes.
            current_gap_rows = rows+cfg.gap_offset
            g = gap[current_gap_rows]
            past_g = gap[(current_gap_rows-1).clamp_min(0)]
            longitudinal = options.teacher_kp*(g-options.teacher_gap)+options.teacher_kd*(g-past_g)/cfg.dt
            pose = data['states'][sl][rows]
            displacement = pose[:,6:8]-pose[:,:2]
            heading = torch.atan2(displacement[:,1],displacement[:,0])
            error = (heading-torch.deg2rad(pose[:,4])+math.pi)%(2*math.pi)-math.pi
            teacher = torch.stack((longitudinal.clamp(-3,3),(-options.teacher_steer_gain*error).clamp(-1,1)),1)
            costs = data['costs'][sl]
            eligible = (h>0)&(nh>0)&(costs[current_gap_rows]==0)&(costs[current_gap_rows+1]==0)
            if options.bc_target == 'recorded':
                # When explicitly cloning logged actions, exclude commands far
                # from the deterministic teacher (unknown exploration flags).
                difference = ((command-teacher)/torch.tensor([3.,1.])).abs().amax(1)
                eligible &= difference <= options.recorded_tolerance
                bc_targets.append(command.clone())
            else:
                bc_targets.append(teacher)
            bc_masks.append(eligible)
            actions.append(command); margins.append(h); next_margins.append(nh); failed.append(nh<=0)
        if not refs:
            raise ValueError('No eligible image transitions in this split')
        self.bc_target,self.bc_mask = torch.cat(bc_targets),torch.cat(bc_masks)
        self.refs = torch.cat(refs)
        self.action,self.margin,self.next_margin,self.failed = [torch.cat(x) for x in (actions,margins,next_margins,failed)]

    def __len__(self):
        return len(self.refs)

    def episode(self,index):
        if index not in self.frames:
            image = torch.load(self.paths[index],mmap=True,map_location='cpu',weights_only=True)
            if image.dtype != torch.uint8 or image.shape != (self.lengths[index],224,224,3):
                raise ValueError(f'Expected uint8 RGB [T,224,224,3]: {self.paths[index]}')
            self.frames[index] = image
            if len(self.frames) > self.cache_episodes:
                self.frames.popitem(last=False)
        self.frames.move_to_end(index)
        return self.frames[index]

    def batch(self,indices):
        indices = torch.as_tensor(indices,dtype=torch.long)
        images,successors = [],[]
        for episode,row in self.refs[indices].tolist():
            frames = self.episode(episode)
            images.append(self.observation(frames,row))
            successors.append(self.observation(frames,row+1))
        return (torch.stack(images),self.action[indices],torch.stack(successors),
                self.margin[indices],self.next_margin[indices],self.failed[indices])

    def observation(self,frames,row):
        if self.history_frames == 1:
            return frames[row]
        # Left-pad with the episode's first image; never cross an episode boundary.
        indices = torch.arange(row-self.history_frames+1,row+1).clamp_min(0)
        return frames[indices]
