"""Quick CARLA state-input control for the RGB SAGE diagnostic.

Run from the repository root:
  .venv/bin/python -m training.test_sage_state --start-server

Uses the existing simulator, replay, reward, gate and optimizer updates. Only
observations/networks change: true gap, ego-minus-leader speed, and ego speed
replace RGB history. No camera, pretrained downloads, or CUDA required.
Default settings match sage_diagnostic.sbatch except for CPU and output/port.
Learning here supports a visual-learning bottleneck; failure in a short run is
inconclusive. This is privileged simulator input, not a deployable controller.
"""
from pathlib import Path
import json
import sys
from datetime import datetime
from unittest.mock import patch

import torch
from torch import nn

from . import train_online_hj as hj
from . import online_sage as sage


class StateActor(nn.Module):
    forward = hj.Actor.forward
    sample_features = hj.Actor.sample_features

    def __init__(self, *contract):
        super().__init__()
        self.head = nn.Sequential(nn.Linear(3, 64), nn.ReLU(),
                                  nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, 2))
        nn.init.uniform_(self.head[-1].weight, -.003, .003)
        nn.init.zeros_(self.head[-1].bias)

    def features(self, images, speed):
        return images[:, -1]


class StateCritic(nn.Module):
    value = hj.Critic.value
    forward = hj.Critic.forward

    def __init__(self, *contract, value_bounds=None):
        super().__init__()
        self.value_bounds = value_bounds
        self.state_head = nn.Sequential(nn.Linear(3, 64), nn.LayerNorm(64), nn.ReLU())
        self.action_head = nn.Sequential(nn.Linear(1, 32), nn.ReLU())
        self.head = nn.Sequential(nn.Linear(96, 64), nn.ReLU(), nn.Linear(64, 1))

    def features(self, images, speed):
        return images[:, -1]


class StateHistory:
    def __init__(self, length, episode, dt):
        if length != 1:
            raise ValueError('State diagnostic requires --history-frames 1')
        self.episode, self.dt, self.last_frame = episode, dt, None

    def capture(self, env):
        from .closed_loop import read_snapshot
        frame, _, _, gap = read_snapshot(env, self.dt)
        snapshot = env.world.get_snapshot()
        if snapshot.frame != frame or (self.last_frame is not None and frame != self.last_frame + 1):
            raise RuntimeError('State observation skipped a simulator frame')
        ego = snapshot.find(env.ego.id).get_velocity()
        leader = snapshot.find(env.nonego.id).get_velocity()
        ego_speed = (ego.x**2 + ego.y**2 + ego.z**2)**.5
        leader_speed = (leader.x**2 + leader.y**2 + leader.z**2)**.5
        state = torch.tensor([gap/10., (ego_speed-leader_speed)/10., ego_speed/30.], dtype=torch.float32)
        self.last_frame = frame
        return hj.Observation((state,), (frame,), ego_speed, self.episode)


class StateSAGE(sage.OnlineSAGE):
    architecture = 'diagnostic_sage_true_state_v1'

    def __init__(self, args, device):
        with patch.multiple(hj, Actor=StateActor, Critic=StateCritic), \
             patch.multiple(sage, Actor=StateActor, Critic=StateCritic):
            super().__init__(args, device)


def main():
    if '--outdir' not in sys.argv and not any(a.startswith('--outdir=') for a in sys.argv):
        sys.argv.extend(['--outdir', str(Path('training/_local') /
                         datetime.now().strftime('sage_state_%Y%m%d_%H%M%S_%f'))])
    args = hj.parse_args(dict(performance_policy='sac', include_speed=True,
        history_frames=1, no_pretrained=True, cpu=True, steps=10000,
        random_steps=2000, learning_starts=2000, gate_start_updates=0,
        exploration_episode_fraction=0., exploration_hold_steps=[1, 1],
        alpha=0., alpha_final=0., gamma_start=.99, gamma_final=.99,
        port=2890), description=__doc__)
    if args.performance_policy != 'sac' or args.history_frames != 1 or args.windowed:
        raise ValueError('Use SAC, --history-frames 1, and no --windowed for this state diagnostic')
    from .braking_scenario import BrakingScenario

    class StateScenario(BrakingScenario):
        def __init__(self, client, config, *rest, **kwargs):
            config.camera = False
            super().__init__(client, config, *rest, **kwargs)

    print(f'STATE diagnostic -> {args.outdir}\nTrue gap / closing speed / ego speed; no RGB.', flush=True)
    # Scoped substitutions reuse the production training/update implementations.
    with patch.multiple(hj, Actor=StateActor, Critic=StateCritic, FrameHistory=StateHistory), \
         patch.multiple(sage, Actor=StateActor, Critic=StateCritic, OnlineSAGE=StateSAGE), \
         patch('training.braking_scenario.BrakingScenario', StateScenario):
        try:
            hj.main(args)
        finally:
            manifest_path = args.outdir / 'manifest.json'
            if manifest_path.exists():
                manifest = json.loads(manifest_path.read_text())
                manifest.update(actor_inputs='true gap / 10, ego-minus-leader speed / 10, ego speed / 30',
                    critic_inputs='true state + normalized longitudinal request',
                    sensor_contract='snapshot state, consecutive frames; RGB camera disabled',
                    critic_fusion='state Linear64/LayerNorm/ReLU + action Linear32/ReLU',
                    preprocessing='privileged state; no RGB',
                    diagnostic='State-input control; compare with a fresh matched RGB diagnostic run')
                manifest_path.write_text(json.dumps(manifest, indent=2)+'\n')


if __name__ == '__main__':
    main()
