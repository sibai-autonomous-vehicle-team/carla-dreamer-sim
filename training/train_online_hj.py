"""Online visual HJ actor-critic on a dedicated, frame-synchronized CARLA server.

No imitation loss or offline dataset is used. The learned action is longitudinal
request [-3, 3]; route steering is supplied by the same fixed tracker every time.
Default inputs are four RGB captures. --include-speed adds measured ego speed.
Contextual CNN-PIDs propose nominal actions after replay exploration. After
critic warmup the HJ gate switches unsafe PID proposals to full braking. Fast-closing resets
maintain hazardous-state coverage; recurring random episodes are opt-in.
The default collision objective has no lower/upper gap failure boundary.
Excessive separation and lane departure truncate rollouts without a failure label.
Alternative following objectives are available explicitly.
--performance-policy safety executes only the safety actor after random warmup,
without a PID, goal actor, or switching gate.
Automatic held-out evaluation runs before training, every 5000 steps between
episodes, and at completion. Results go to evaluation.jsonl; only final or
interrupted weights are saved. --eval-interval 0 disables evaluation.
Continue with --resume CHECKPOINT --outdir NEW_DIR --steps TOTAL_STEPS.
Training settings, weights, optimizers and counters are restored; replay is
rebuilt with the restored policy. Simulator episodes and RNG restart.
The six default starts are a screening suite, not a safety certificate. Each PID
gets PID-only, deterministic safety, filtered PID, and one-action recovery probes.
Probes gather real frame history under full braking before testing a PID
action; pre-probe collisions are reported separately from value-prediction tests.
Evaluation uses no replay/updates/exploration or real-time pacing. Increase
--eval-cases for stronger evidence at the cost of additional simulator time.
Open OUTDIR/evaluation_dashboard.html in a browser for refreshing plots, or
OUTDIR/evaluation_plot.png in an image viewer. Both update after each evaluation.

Example (start a dedicated CARLA server separately, with rendering enabled):
  python -m training.train_online_hj --outdir training/_local/online_hj_cloudy \
      --condition day_cloudy --port 2800 --require-cuda
Add --start-server to launch the bundled simulator. Checkpoints have a separate
format; they are NOT compatible with the existing offline RGBSafetyFilter.
Use --pid-checkpoint PATH for a generalist CNN-PID instead of specialist profiles;
--target-gap and --pid-kp/--pid-ki/--pid-kd configure its frozen PID.
Add --windowed for a combined chase/model-camera preview; --real-time limits
decision speed when compute is faster than simulation time. The server itself
renders off-screen; the preview is a separate Python window.
Add --action-diagnostics-interval 10 for a live same-observation action/Q scan,
also written to action_diagnostics.jsonl and W&B when enabled. The overlay
identifies its capture step; diagnostics do not select controls or consume RNG.
"""
from collections import deque, Counter
from copy import deepcopy
from dataclasses import dataclass
import argparse
import json
import hashlib
import math
from pathlib import Path
import random
import queue
import sys
import time
from types import SimpleNamespace

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .visual_algorithms import soft_update
from .visual_models import RGBEncoder, PREPROCESS, temporal_features


@dataclass(frozen=True)
class Observation:
    # Individual immutable captures are shared between overlapping histories.
    frames: tuple
    frame_ids: tuple
    speed: float
    episode: int


class RandomExplorer:
    """Temporally coherent random controls, not independent 10 Hz jitter."""
    def __init__(self, hold_steps, rng=None):
        self.low,self.high = hold_steps
        if not 1 <= self.low <= self.high: raise ValueError('Exploration hold bounds must be positive and ordered')
        self.rng = rng if rng is not None else random
        self.remaining,self.command = 0,0.

    def action(self):
        if self.remaining == 0:
            self.command = self.rng.uniform(-3.,3.)
            self.remaining = self.rng.randint(self.low,self.high)
        self.remaining -= 1
        return self.command


class ContextPID:
    """Frozen specialist CNN plus its PID; only synchronized RGB supplies gap."""
    def __init__(self, folder, device):
        self.folder = Path(folder)
        self.profile = json.loads((self.folder/'pid.json').read_text())
        path = self.folder/'cnn.pth'
        self.sha256 = hashlib.sha256(path.read_bytes()).hexdigest()
        if self.profile.get('cnn_sha256') != self.sha256:
            raise ValueError(f'PID CNN checksum mismatch: {path}')
        self.load_encoder(path, device)

    def load_encoder(self, path, device):
        self.checkpoint = Path(path)
        self.device = device
        self.encoder = RGBEncoder(pretrained=False,freeze_batch_norm=True)
        self.encoder.network.fc = nn.Linear(512,1)
        self.encoder.network.load_state_dict(torch.load(path,map_location='cpu',weights_only=True)['model'])
        self.encoder.to(device).eval().requires_grad_(False)
        self.reset()

    def reset(self):
        from run_gap_pid import GapPID
        p=self.profile
        self.pid=GapPID(p['target_gap'],p['kp'],p['ki'],p['kd'])

    @torch.inference_mode()
    def propose(self, observation, dt):
        frame=observation.frames[-1][None].to(self.device)
        gap=float(self.encoder.network(self.encoder.preprocess(frame))[0,0])
        if not math.isfinite(gap): raise FloatingPointError('Nonfinite nominal CNN gap')
        return self.pid.update(gap,dt),gap


class GeneralistPID(ContextPID):
    """Frozen pooled gap CNN with one PID, independent of weather profiles."""
    def __init__(self, checkpoint, device, *, target_gap=6., kp=.5, ki=0., kd=.5):
        values = dict(target_gap=target_gap, kp=kp, ki=ki, kd=kd)
        if any(not math.isfinite(v) or v < 0 for v in values.values()) or target_gap <= 0:
            raise ValueError('PID target must be positive and gains finite and nonnegative')
        path = Path(checkpoint)
        self.folder = path.parent
        self.sha256 = hashlib.sha256(path.read_bytes()).hexdigest()
        self.profile = dict(values, cnn_sha256=self.sha256)
        self.load_encoder(path, device)


class SafetyPolicy:
    """Adapter for direct execution and deterministic evaluation of one safety actor."""
    is_safety = True

    def __init__(self, learner, target_gap):
        self.learner = learner
        self.profile = dict(target_gap=target_gap)

    def reset(self): pass

    def propose(self, observation, dt, deterministic=False):
        return self.learner.action(observation, deterministic=deterministic), None


def behavior_action(learner, observation, nominal, explorer, step, args, exploration_episode):
    """Select random warmup, direct safety execution, or a filtered proposer."""
    result=dict(nominal_action=nominal,nominal_q=None,intervened=False,gate_active=False)
    if step < args.random_steps:
        result.update(action=explorer.action(),source='warmup_random')
    elif getattr(args,'performance_policy','pid') == 'safety':
        result.update(action=nominal,source='safety_actor')
    elif exploration_episode:
        result.update(action=explorer.action(),source='coverage_random')
    elif learner.updates < args.gate_start_updates:
        result.update(action=nominal,source='pid_critic_warmup')
    else:
        q=learner.score(observation,nominal)
        intervene=q < args.switching_threshold
        pid_mode=getattr(args,'performance_policy','pid') == 'pid'
        fallback=-3. if pid_mode else None
        result.update(action=(fallback if pid_mode else learner.action(observation)) if intervene else nominal,
                      source=('full_brake_fallback' if pid_mode else 'safety_actor') if intervene else 'contextual_pid',
                      nominal_q=q,intervened=intervene,gate_active=True)
    return result


def sample_start(args, rng):
    challenge=rng.random() < args.challenge_fraction
    closing=rng.uniform(*args.closing_speed_range) if challenge else 0.
    leader=rng.uniform(args.speed_range[0],args.speed_range[1]-closing)
    bounds=args.challenge_gap_range if challenge else args.gap_range
    return dict(challenge=challenge,closing_speed=closing,leader_speed=leader,
                ego_speed=leader+closing,gap=rng.uniform(*bounds),gap_bounds=bounds,
                exploration_episode=rng.random() < args.exploration_episode_fraction)


class TrainingViewer:
    """One window: synchronized chase view and the exact newest training RGB."""
    def __init__(self):
        import pygame
        self.pygame = pygame
        pygame.display.init(); pygame.font.init()
        self.window = pygame.display.set_mode((1600,720))
        pygame.display.set_caption('Online HJ training — chase view + model RGB')
        self.font = pygame.font.Font(None,24)
        self.chase_frames = queue.Queue()
        self.chase_frame, self.chase = None, None
        self.action_diagnostics = None

    def attach(self, env):
        # Attached relative to ego: no extra world tick or changed initial state.
        self.chase_frames = queue.Queue()
        self.chase_frame, self.chase = None, None
        self.action_diagnostics = None
        camera = env.world.get_blueprint_library().find('sensor.camera.rgb')
        for key,value in {'image_size_x':'1280','image_size_y':'720','fov':'90','sensor_tick':'0.0'}.items():
            camera.set_attribute(key,value)
        sensor = env.world.spawn_actor(camera,env.carla.Transform(
            env.carla.Location(x=-8.,z=5.),env.carla.Rotation(pitch=-22.)),attach_to=env.ego)
        env.actors.append(sensor)
        captures = self.chase_frames
        def callback(event):
            bgra = np.frombuffer(event.raw_data,dtype=np.uint8).reshape(event.height,event.width,4)
            captures.put((event.frame,bgra[:,:,:3][:,:,::-1].copy()))
        sensor.listen(callback)

    def synchronize(self, frame):
        deadline = time.monotonic()+30
        while True:
            self.pump()
            remaining = deadline-time.monotonic()
            if remaining <= 0: raise RuntimeError(f'Chase-camera timeout for frame {frame}')
            try: captured,image = self.chase_frames.get(timeout=min(.1,remaining))
            except queue.Empty: continue
            if captured > frame: raise RuntimeError('Chase camera is ahead of the training snapshot')
            if captured == frame:
                self.chase_frame,self.chase = captured,image
                return

    def pump(self):
        for event in self.pygame.event.get():
            if event.type == self.pygame.QUIT or (event.type == self.pygame.KEYDOWN and event.key == self.pygame.K_ESCAPE):
                raise KeyboardInterrupt('Training viewer closed; saving interrupted checkpoint')

    def show(self, observation, *, step, action, steering, gap, q, updates, phase, metrics,
             actor_active=False, epsilon=1., controller_label=None):
        self.pump()
        if self.chase_frame != observation.frame_ids[-1]:
            raise RuntimeError('Preview cameras do not match the training decision frame')
        pygame = self.pygame
        self.window.fill((15,19,26))
        self.window.blit(pygame.surfarray.make_surface(self.chase.swapaxes(0,1)),(0,0))
        # Native size, raw RGB, no normalization/resizing/annotation on the image.
        raw = observation.frames[-1].numpy()
        self.window.blit(pygame.surfarray.make_surface(raw.swapaxes(0,1)),(1328,55))
        labels = ['Model input: RGB 224 x 224',f'Capture / chase: {self.chase_frame}',
                  f'History: {len(observation.frames)} frames',
                  f'Episode {observation.episode+1} | step {step}',phase,
                  'SAFETY ACTOR - training' if actor_active else (controller_label or 'RANDOM EXPLORATION'),
                  f'Intervention epsilon: {epsilon:.2f} m',
                  f'Longitudinal: {action:+.3f}',f'Steering: {steering:+.3f}',
                  f'Q(action): {q:+.3f} m',f'Updates: {updates}',
                  f'Debug gap: {gap:.2f} | speed: {observation.speed:.2f}']
        if metrics:
            labels.extend([f"Critic MSE: {metrics['critic_mse']:.5f}",
                           f"Gamma: {metrics['gamma']:.5f} | alpha: {metrics['alpha']:.3f}"])
        labels.extend(['Esc / close: save and stop','Training may run below real time.'])
        self.window.blit(self.font.render(labels[0],True,(240,245,250)),(1300,15))
        for index,label in enumerate(labels[1:]):
            color = (255,75,75) if label.startswith('SAFETY ACTOR') else (225,232,242)
            self.window.blit(self.font.render(label,True,color),(1292,295+index*26))
        diagnostic = getattr(self,'action_diagnostics',None)
        if diagnostic is not None:
            pygame.draw.rect(self.window,(15,19,26),(15,535,1030,175))
            lines = [f"Same-observation diagnostic | step {diagnostic['step']} | frame {diagnostic['frame']}",
                f"Executed: {diagnostic['executed_action']:+.3f} | deterministic: {diagnostic['deterministic_action']:+.3f} | latent std: {diagnostic['latent_std']:.3f}",
                f"Q brake: {diagnostic['brake_q']:+.4f} | coast: {diagnostic['coast_q']:+.4f} | throttle: {diagnostic['throttle_q']:+.4f}",
                f"Q deterministic: {diagnostic['deterministic_q']:+.4f} | executed: {diagnostic['executed_q']:+.4f}",
                f"Best grid action: {diagnostic['best_grid_action']:+.2f} | Q advantage over actor: {diagnostic['actor_q_gap']:.4f}",
                f"Gap: {diagnostic['gap']:.2f} m | ego-minus-leader speed: {diagnostic['closing_speed']:+.2f} m/s"]
            for index,label in enumerate(lines):
                self.window.blit(self.font.render(label,True,(240,245,250)),(25,545+index*26))
        pygame.display.flip()

    def close(self):
        self.pygame.display.quit(); self.pygame.font.quit()


def with_preview(base, viewer):
    """Install the chase camera before the existing first sensor-barrier tick."""
    class PreviewScenario(base):
        def tick(self):
            if not getattr(self,'_training_preview_attached',False):
                viewer.attach(self)
                self._training_preview_attached = True
            frame = super().tick()
            viewer.synchronize(frame)
            return frame
    return PreviewScenario


def route_speed_limit(tracker, location, desired, lateral_accel=2., deceleration=2.):
    """Curvature speed bounds with preview distance for service braking."""
    progress,_ = tracker.project(location)
    limit = desired
    for i in range(max(1,tracker.index),len(tracker.route)-1):
        distance = tracker.lengths[i]-progress
        if distance < -2.: continue
        if distance > max(15.,desired**2/(2*deceleration)+10.): break
        a,b,c = tracker.route[i-1:i+2]
        ab = math.hypot(b[0]-a[0],b[1]-a[1])
        bc = math.hypot(c[0]-b[0],c[1]-b[1])
        ac = math.hypot(c[0]-a[0],c[1]-a[1])
        cross = abs((b[0]-a[0])*(c[1]-a[1])-(b[1]-a[1])*(c[0]-a[0]))
        curvature = 2*cross/max(ab*bc*ac,1e-6)
        if curvature > 1e-6:
            limit = min(limit,math.sqrt(lateral_accel/curvature+2*deceleration*max(0.,distance-5.)))
    return limit


def with_route_aware_leader(base, lateral_accel=2., deceleration=2.):
    class RouteAwareScenario(base):
        def initialize_speed(self, speed):
            from .braking_scenario import InvalidBrakingScenario
            limit = route_speed_limit(self.tracker(self.nonego),self.nonego.get_location(),speed,
                                     lateral_accel,deceleration)
            if speed > limit+.1:
                raise InvalidBrakingScenario(f'Requested start speed {speed:.2f} m/s exceeds curve preview limit {limit:.2f} m/s')
            return super().initialize_speed(speed)

        def leader_control(self):
            tracker = self.tracker(self.nonego)
            steer = tracker.steering()
            elapsed = (self.world.get_snapshot().frame-self.initial_frame)*self.dt
            if self.args.leader_brake_after <= elapsed < self.args.leader_brake_after+self.args.leader_brake_duration:
                return self.carla.VehicleControl(brake=1.,steer=steer)
            target = route_speed_limit(tracker,self.nonego.get_location(),self.args.initial_speed,
                                      lateral_accel,deceleration)
            error = target-self.nonego.get_velocity().length()
            return self.carla.VehicleControl(throttle=min(2/3,max(0.,.35*error)),
                                            brake=min(1.,max(0.,-.35*error)),steer=steer)
    return RouteAwareScenario


def with_closing_starts(base, lateral_accel=2., deceleration=2.):
    """Initialize separate ego/leader speeds, then align the moving gap."""
    class ClosingScenario(base):
        def initialize_speed(self, speed):
            from .braking_scenario import InvalidBrakingScenario
            desired=(self.args.ego_initial_speed,speed)
            if abs(desired[0]-speed)<1e-6: return super().initialize_speed(speed)
            actors=(self.ego,self.nonego)
            for actor,target in zip(actors,desired):
                limit=route_speed_limit(self.tracker(actor),actor.get_location(),target,lateral_accel,deceleration)
                if target>limit+.1: raise InvalidBrakingScenario('Fast-closing start exceeds route curvature limit')
            commands=list(desired)
            for _ in range(math.ceil(3/self.dt)):
                self.validate_lanes()
                for actor,command in zip(actors,commands):
                    actor.apply_control(self.carla.VehicleControl(steer=self.tracker(actor).steering()))
                    f=actor.get_transform().get_forward_vector()
                    actor.set_target_velocity(self.carla.Vector3D(x=command*f.x,y=command*f.y,z=command*f.z))
                self.world.tick()
                actual=[a.get_velocity().length() for a in actors]
                if all(abs(v-target)<=.05 for v,target in zip(actual,desired)): return
                commands=[max(0.,min(target+3.,command+.5*(target-v)))
                          for command,v,target in zip(commands,actual,desired)]
            raise InvalidBrakingScenario('Separate ego/leader speeds did not settle')

        def set_moving_gap(self, bounds):
            closing=self.args.ego_initial_speed-self.args.initial_speed
            if abs(closing)<1e-6: return super().set_moving_gap(bounds)
            from .braking_scenario import InvalidBrakingScenario
            for _ in range(5):
                if bounds[0]<=self.get_distance()<=bounds[1]:
                    if self.events: raise InvalidBrakingScenario('Contact during moving-start setup')
                    self.validate_lanes(); return
                self.place_ego(sum(bounds)/2+closing*self.dt)
                for actor,speed in ((self.ego,self.args.ego_initial_speed),(self.nonego,self.args.initial_speed)):
                    f=actor.get_transform().get_forward_vector()
                    actor.set_target_velocity(self.carla.Vector3D(x=speed*f.x,y=speed*f.y,z=speed*f.z))
                    actor.apply_control(self.carla.VehicleControl(steer=self.tracker(actor).steering()))
                self.tick()
            raise InvalidBrakingScenario('Could not establish close moving-start gap')

        def validate_handoff_speed(self):
            super().validate_handoff_speed()
            from .braking_scenario import InvalidBrakingScenario
            for actor,target in ((self.ego,self.args.ego_initial_speed),(self.nonego,self.args.initial_speed)):
                if abs(actor.get_velocity().length()-target)>.2:
                    raise InvalidBrakingScenario('Handoff speed differs from requested closing scenario')
    return ClosingScenario


def planned_lane_status(env, actor):
    tracker = env.tracker(actor)
    transform = actor.get_transform()
    progress,offset = tracker.project(transform.location)
    _,yaw = tracker.point(progress)
    heading = math.radians(transform.rotation.yaw-yaw)
    half_width = abs(math.cos(heading))*actor.bounding_box.extent.y + abs(math.sin(heading))*actor.bounding_box.extent.x
    waypoint = env.map.get_waypoint(transform.location,lane_type=env.carla.LaneType.Driving)
    allowance = waypoint.lane_width/2-half_width+.15
    return dict(offset_m=offset,allowance_m=allowance,inside=offset <= allowance)


class FrameHistory:
    def __init__(self, length, episode, dt=.1):
        if length < 1 or not math.isfinite(dt) or dt <= 0:
            raise ValueError('Positive history length and finite timestep required')
        self.length, self.episode, self.dt = length, episode, dt
        self.frames, self.ids = deque(maxlen=length), deque(maxlen=length)
        self.timestamp = None

    def capture(self, env):
        snapshot = env.world.get_snapshot()
        frame = snapshot.frame
        if not math.isclose(snapshot.timestamp.delta_seconds, self.dt, abs_tol=1e-4):
            raise RuntimeError('Simulator timestep differs from observation contract')
        if self.ids and (frame != self.ids[-1] + 1 or not math.isclose(
                snapshot.timestamp.elapsed_seconds-self.timestamp, self.dt, abs_tol=1e-4)):
            raise RuntimeError('Nonconsecutive action-time snapshots')
        image = env.get_rgb_observation(frame)
        if image.dtype != torch.uint8 or tuple(image.shape) != (224, 224, 3):
            raise ValueError('Expected synchronized uint8 RGB [224,224,3]')
        if not image.any():
            raise RuntimeError('Camera returned a black placeholder')
        # Freeze callback storage once, not once per overlapping history.
        image = image.cpu().clone()
        if not self.ids:
            self.frames.extend([image]*self.length)
            self.ids.extend([frame]*self.length)
        else:
            self.frames.append(image); self.ids.append(frame)
        self.timestamp = snapshot.timestamp.elapsed_seconds
        actor = snapshot.find(env.ego.id)
        if actor is None:
            raise RuntimeError('Ego missing from camera-time snapshot')
        velocity = actor.get_velocity()
        speed = math.sqrt(velocity.x**2+velocity.y**2+velocity.z**2)
        if env.world.get_snapshot().frame != frame:
            raise RuntimeError('Another client advanced the simulator during observation capture')
        return Observation(tuple(self.frames), tuple(self.ids), speed, self.episode)


@dataclass
class Transition:
    observation: Observation
    action: float
    successor: Observation
    margin: float
    next_margin: float
    failed: bool
    reward: float = 0.
    terminated: bool = False


def tensors(observations, device):
    images = torch.stack([torch.stack(o.frames) for o in observations]).to(device)
    speed = torch.tensor([[o.speed/30.] for o in observations], device=device)
    return images, speed


class Replay:
    def __init__(self, capacity):
        if capacity < 1: raise ValueError('Replay capacity must be positive')
        self.rows = deque(maxlen=capacity)

    def __len__(self): return len(self.rows)

    def extend(self, rows):
        for row in rows:
            if (row.observation.episode != row.successor.episode or
                    row.successor.frame_ids[-1] != row.observation.frame_ids[-1] + 1):
                raise ValueError('Replay transition crosses a reset or skips a frame')
            if row.observation.frame_ids[1:] != row.successor.frame_ids[:-1]:
                raise ValueError('Successor history does not shift by exactly one capture')
            self.rows.append(row)

    def sample(self, count, device):
        rows = random.choices(self.rows, k=count)
        image, speed = tensors([r.observation for r in rows], device)
        successor, next_speed = tensors([r.successor for r in rows], device)
        scalar = lambda field: torch.tensor([getattr(r, field) for r in rows], device=device)
        return (image, speed, scalar('action')[:, None], successor, next_speed, scalar('margin'),
                scalar('next_margin'), scalar('failed').bool(), scalar('reward'), scalar('terminated').bool())


class VisualNetwork(RGBEncoder):
    def __init__(self, history, include_speed, pretrained, freeze_encoder):
        super().__init__(pretrained=pretrained, freeze_batch_norm=True)
        self.history_frames, self.include_speed = history, include_speed
        self.network.fc = nn.Identity()
        if freeze_encoder: self.network.requires_grad_(False)
        self.width = 512*history + int(include_speed)

    def features(self, images, speed):
        features = temporal_features(self, images)
        return torch.cat((features, speed), -1) if self.include_speed else features


class Actor(VisualNetwork):
    def __init__(self, *args):
        super().__init__(*args)
        self.head = nn.Sequential(nn.Linear(self.width, 256), nn.ReLU(), nn.Linear(256, 2))
        nn.init.uniform_(self.head[-1].weight, -.003, .003)
        nn.init.zeros_(self.head[-1].bias)

    def forward(self, images, speed, deterministic=False):
        return self.sample_features(self.features(images, speed), deterministic)

    def sample_features(self, features, deterministic=False):
        """Sample a physical request; log density uses normalized [-1,1] actions.

        The constant scale of three is applied after the tanh correction, as in
        the normalized-action SAC convention. Cached features require a frozen
        visual encoder.
        """
        mean, log_std = self.head(features).chunk(2, -1)
        distribution = torch.distributions.Normal(mean, log_std.clamp(-5, 2).exp())
        raw = mean if deterministic else distribution.rsample()
        log_jacobian = 2*(math.log(2)-raw-F.softplus(-2*raw))
        log_prob = (distribution.log_prob(raw)-log_jacobian).sum(-1)
        return 3*raw.tanh(), log_prob


class Critic(VisualNetwork):
    def __init__(self, *args, value_bounds=None):
        super().__init__(*args)
        # Compress and normalize vision before fusion. A single action scalar
        # beside 2048 unnormalized visual features is easily ignored by the head.
        self.state_head = nn.Sequential(nn.Linear(self.width,128),nn.LayerNorm(128),nn.ReLU())
        self.action_head = nn.Sequential(nn.Linear(1,32),nn.ReLU())
        self.value_bounds = value_bounds
        self.head = nn.Sequential(nn.Linear(160, 256), nn.ReLU(), nn.Linear(256, 128),
                                  nn.ReLU(), nn.Linear(128, 1))

    def value(self, features, action):
        raw = self.head(torch.cat((self.state_head(features),self.action_head(action/3)), -1)).squeeze(-1)
        if self.value_bounds is None: return raw
        low,high=self.value_bounds
        return low+(high-low)*torch.sigmoid(raw)

    def forward(self, images, speed, action):
        return self.value(self.features(images,speed),action)


def hj_backup(margin, next_margin, failed, continuation, gamma):
    """SAGE's discounted safety backup, distinct from SAC's reward backup.

    Entropy regularizes the safety actor only; adding an entropy bonus here
    would change the safety-value interpretation used by the switching gate.
    """
    # A reset due to time limit is NOT a failure; it retains a bootstrap.
    continuation = torch.where(failed, next_margin, continuation)
    return (1-gamma)*margin + gamma*torch.minimum(margin, continuation)


class OnlineHJ:
    architecture = 'online_hj_rgb_longitudinal_v2'
    model_names = ('actor', 'critics', 'targets')
    snapshot_names = model_names + ('actor_optimizer', 'critic_optimizer')
    def __init__(self, args, device):
        contract = (args.history_frames, args.include_speed,
                    not args.no_pretrained and not getattr(args,'resume',None), args.freeze_encoder)
        self.actor = Actor(*contract).to(device)
        self.safety_bounds = ((-args.event_failure_margin,args.max_gap)
                              if getattr(args,'safety_objective','following') == 'collision' else None)
        self.critics = nn.ModuleList([Critic(*contract,value_bounds=self.safety_bounds) for _ in range(2)]).to(device)
        self.targets = deepcopy(self.critics).eval().requires_grad_(False)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=args.actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critics.parameters(), lr=args.critic_lr)
        self.device, self.args, self.updates = device, args, 0

    def schedule(self, start, end):
        if self.args.anneal_updates == 0:
            return end
        return start + (end-start)*min(1., self.updates/self.args.anneal_updates)

    @torch.no_grad()
    def action(self, observation, deterministic=False):
        self.actor.eval()
        images, speed = tensors([observation], self.device)
        return float(self.actor(images, speed, deterministic)[0][0, 0])

    @torch.no_grad()
    def score(self, observation, action):
        """Frozen twin-min Q for a longitudinal request with the fixed route tracker."""
        if not math.isfinite(action) or not -3 <= action <= 3:
            raise ValueError('Longitudinal request must be finite and in [-3,3]')
        self.critics.eval()
        images, speed = tensors([observation], self.device)
        command = torch.tensor([[action]], device=self.device, dtype=torch.float32)
        return float(torch.minimum(*(q(images,speed,command) for q in self.critics))[0])

    @torch.no_grad()
    def action_diagnostics(self, observation, executed_action):
        """Compare actions at one observation, without sampling or advancing CARLA."""
        modes=[(m,m.training) for model in (self.actor,self.critics) for m in model.modules()]
        try:
            self.actor.eval(); self.critics.eval()
            images,speed=tensors([observation],self.device)
            mean,log_std=self.actor.head(self.actor.features(images,speed)).chunk(2,-1)
            deterministic=3*mean.tanh()
            grid=torch.linspace(-3.,3.,21,device=self.device).reshape(-1,1)
            actions=torch.cat((grid,deterministic,grid.new_tensor([[executed_action]])))
            values=[]
            for critic in self.critics:
                features=critic.features(images,speed).expand(len(actions),-1)
                values.append(critic.value(features,actions))
            q=torch.minimum(*values)
            best=int(q[:21].argmax())
            return dict(executed_action=float(executed_action),
                deterministic_action=float(deterministic[0,0]),
                latent_std=float(log_std.clamp(-5,2).exp()[0,0]),
                brake_q=float(q[0]),coast_q=float(q[10]),throttle_q=float(q[20]),
                deterministic_q=float(q[21]),executed_q=float(q[22]),
                best_grid_action=float(grid[best,0]),
                actor_q_gap=max(0.,float(q[best]-q[21])))
        finally:
            for module,mode in modes: module.training=mode

    def filter_action(self, observation, nominal_action):
        """PID gates use full braking; learned goal policies retain their safety actor."""
        score = self.score(observation, nominal_action)
        intervene = score < getattr(self.args,'switching_threshold',1.)
        if intervene:
            action=(-3. if getattr(self.args,'performance_policy','pid') == 'pid'
                    else self.action(observation,True))
        else:
            action=nominal_action
        return dict(action=action,
                    nominal_q=score, intervened=intervene)

    @classmethod
    def from_checkpoint(cls, path, device='cpu'):
        """Load a frozen evaluator; this does not resume simulator/replay training."""
        payload = torch.load(path, map_location=device, weights_only=True)
        if payload.get('format_version') != 1 or payload.get('architecture') != cls.architecture:
            raise ValueError('Expected an online HJ longitudinal checkpoint')
        config = dict(payload['config'])
        config['no_pretrained'] = True  # Saved weights supply all initialization.
        learner = cls(SimpleNamespace(**config), device)
        for name in cls.model_names:
            model = getattr(learner,name)
            model.load_state_dict(payload[name]); model.eval().requires_grad_(False)
            if any(not torch.isfinite(p).all() for p in model.parameters()):
                raise ValueError('Nonfinite checkpoint weights')
        learner.updates = payload['updates']
        return learner

    def update(self, batch):
        image, speed, action, successor, next_speed, margin, next_margin, failed = batch[:8]
        gamma = self.schedule(self.args.gamma_start, self.args.gamma_final)
        alpha = self.schedule(self.args.alpha, self.args.alpha_final)
        self.actor.train(); self.critics.train()
        with torch.no_grad():
            next_action, _ = self.actor(successor, next_speed)
            continuation = torch.minimum(*(q(successor, next_speed, next_action) for q in self.targets))
            target = hj_backup(margin, next_margin, failed, continuation, gamma)
            if self.safety_bounds is not None:
                low,high=self.safety_bounds
                if not ((margin>=low)&(margin<=high)&(next_margin>=low)&(next_margin<=high)).all():
                    raise ValueError('Safety labels outside the declared collision-value bounds')
                target=target.clamp(low,high)
        values = [q(image, speed, action) for q in self.critics]
        critic_loss = sum(F.mse_loss(q, target) for q in values)/2
        if not torch.isfinite(critic_loss): raise FloatingPointError('Nonfinite HJ critic loss')
        self.critic_optimizer.zero_grad(set_to_none=True)
        critic_loss.backward()
        nn.utils.clip_grad_norm_(self.critics.parameters(), 10.)
        self.critic_optimizer.step()
        # Preserve explicitly frozen encoder parameters when toggling critic grads.
        parameters = [p for p in self.critics.parameters() if p.requires_grad]
        for p in parameters: p.requires_grad_(False)
        self.critics.eval()
        try:
            proposed, log_prob = self.actor(image, speed)
            q = torch.minimum(*(c(image, speed, proposed) for c in self.critics))
            action_gradient=torch.autograd.grad(q.sum(),proposed,retain_graph=True)[0].detach().abs().mean()
            actor_loss = (alpha*log_prob-q).mean()
            if not torch.isfinite(actor_loss): raise FloatingPointError('Nonfinite HJ actor loss')
            self.actor_optimizer.zero_grad(set_to_none=True)
            actor_loss.backward()
            nn.utils.clip_grad_norm_(self.actor.parameters(), 10.)
            self.actor_optimizer.step()
        finally:
            for p in parameters: p.requires_grad_(True)
        soft_update(self.targets, self.critics, self.args.tau)
        self.updates += 1
        return dict(updates=self.updates, gamma=gamma, alpha=alpha,
                    critic_mse=float(critic_loss.detach()), actor_loss=float(actor_loss.detach()),
                    target_mean=float(target.mean()), q_mean=float(values[0].detach().mean()),
                    q_std=float(values[0].detach().std(unbiased=False)),
                    disagreement=float((values[0]-values[1]).detach().abs().mean()),
                    entropy=float(-log_prob.detach().mean()),
                    safety_action_gradient=float(action_gradient),
                    q_min=float(torch.minimum(*values).detach().min()),
                    q_max=float(torch.maximum(*values).detach().max()))

    def restore_training(self, path):
        """Restore trainable networks and optimizers; replay and simulator restart."""
        payload = torch.load(path, map_location=self.device, weights_only=True)
        if payload.get('format_version') != 1 or payload.get('architecture') != self.architecture:
            raise ValueError('Checkpoint architecture does not match this trainer')
        for name in self.snapshot_names:
            getattr(self,name).load_state_dict(payload[name])
        for name in self.model_names:
            if any(not p.isfinite().all() for p in getattr(self,name).parameters()):
                raise ValueError('Nonfinite checkpoint weights')
        self.updates = int(payload['updates'])
        return int(payload['steps'])

    def save(self, path, steps):
        payload = dict(format_version=1, architecture=self.architecture,
                       objective={'collision':'discounted_collision_first_hit',
                                  'collision-following':'discounted_collision_and_maximum_separation_first_hit_with_lane_failure',
                                  'following':'discounted_two_sided_first_hit_with_collision_and_lane_failure'}[
                                      getattr(self.args,'safety_objective','following')],
                       config={k:str(v) if isinstance(v, Path) else v for k,v in vars(self.args).items()},
                       preprocessing=PREPROCESS, steps=steps, updates=self.updates,
                       note='Training continuation supported; replay and simulator/RNG state are not saved. Fresh replay is collected before updates resume.')
        for name in self.snapshot_names:
            payload[name] = getattr(self, name).state_dict()
        temporary = path.with_suffix('.tmp')
        torch.save(payload, temporary); temporary.replace(path)


def margin(gap, args):
    if getattr(args,'safety_objective','following') == 'collision-following':
        return min(gap,args.max_gap-gap)
    if getattr(args,'safety_objective','following') == 'collision':
        # The unsigned archive gap cannot encode penetration. Contact callbacks
        # provide the negative successor label in finalize_episode instead.
        # Cap positive margins for numerical scale; this is NOT a far-gap failure.
        return min(gap,args.max_gap)
    return min(gap-args.min_gap, args.max_gap-gap)


def gap_failure(gap, args):
    """Collision mode relies on contact events, never on an unsigned gap cutoff."""
    if getattr(args,'safety_objective','following') == 'collision-following':
        return gap >= args.max_gap
    return getattr(args,'safety_objective','following') == 'following' and margin(gap,args) <= 0


def gap_termination(gap, args):
    """Episode stop bounds can differ from the unchanged safety constraints."""
    bounds=getattr(args,'termination_gap_range',None)
    if bounds is None: return gap_failure(gap,args)
    return gap <= bounds[0] or gap >= bounds[1]


def first_hit_rows(rows):
    """Preserve the first-hit training objective when collection continues."""
    index=next((i for i,row in enumerate(rows) if row.failed),None)
    return rows if index is None else rows[:index+1]


def finalize_episode(rows, collision_events, departures, failure_margin, *, include_lane_failure=True):
    """Resolve delayed event callbacks before admitting an episode to replay."""
    events = [frame for frame, _ in collision_events]
    if include_lane_failure: events += list(departures)
    if rows:
        events = [f for f in events if rows[0].observation.frame_ids[-1] <= f <= rows[-1].successor.frame_ids[-1]]
    if events and rows:
        first = min(events)
        index = next(i for i, r in enumerate(rows) if r.successor.frame_ids[-1] >= first)
        rows = rows[:index+1]
        rows[-1].failed = True
        rows[-1].next_margin = min(rows[-1].next_margin, -failure_margin)
    return rows


def parse_args(defaults=None, description=None, argv=None):
    p = argparse.ArgumentParser(description=description or __doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--outdir', type=Path, required=True)
    p.add_argument('--resume', type=Path,
                   help='Continue checkpoint weights/optimizers/counters in a NEW outdir; rebuild replay, --steps is the total environment-step target')
    p.add_argument('--performance-policy', choices=('pid','sac','safety'), default='pid',
                   help='safety: execute the safety actor directly after random warmup; no PID or goal actor')
    p.add_argument('--target-gap', type=float, default=6.)
    p.add_argument('--performance-gamma', type=float, default=.99)
    p.add_argument('--performance-tau', type=float, default=.005)
    p.add_argument('--performance-alpha', type=float, default=.2)
    p.add_argument('--collision-penalty', type=float, default=10.)
    p.add_argument('--lane-penalty', type=float, default=5.)
    p.add_argument('--condition', default='day_cloudy', choices=[f'{t}_{w}' for t in ('day','sunset','night') for w in ('sunny','cloudy','rainy')])
    p.add_argument('--steps', type=int, default=50000)
    p.add_argument('--episode-steps', type=int, default=100)
    p.add_argument('--history-frames', type=int, default=4)
    p.add_argument('--include-speed', action='store_true', help='Extra measured ego-speed input; changes the comparison information budget')
    p.add_argument('--freeze-encoder', action='store_true')
    p.add_argument('--no-pretrained', action='store_true')
    p.add_argument('--require-cuda', action='store_true')
    p.add_argument('--cpu', action='store_true')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--capacity', type=int, default=10000, help='Transitions; shared uint8 captures use approximately 1.5 GB at capacity, plus episode boundaries')
    p.add_argument('--batch-size', type=int, default=8)
    p.add_argument('--random-steps', type=int, default=1000)
    p.add_argument('--pid-root', type=Path, default=Path('specialists'), help='Contextual PID bank containing CONDITION/cnn.pth and pid.json')
    p.add_argument('--pid-checkpoint', type=Path, help='Generalist gap CNN checkpoint; replaces the contextual PID bank, using target-gap and pid gains')
    p.add_argument('--pid-kp', type=float, default=.5)
    p.add_argument('--pid-ki', type=float, default=0.)
    p.add_argument('--pid-kd', type=float, default=.5)
    p.add_argument('--nominal-conditions', nargs='+', help='PID profiles to sample per episode; default is weather-matched profile')
    p.add_argument('--gate-start-updates', type=int, default=500, help='Use ungated PID until this many critic updates, then enable safety interventions')
    p.add_argument('--exploration-episode-fraction', type=float, default=0.,
                   help='Optional fraction of random episodes that bypass the safety gate after initial warmup (default 0, disabled)')
    p.add_argument('--challenge-fraction', type=float, default=.5, help='Fraction of close, fast-closing resets')
    p.add_argument('--closing-speed-range', type=float, nargs=2, default=[1.,6.], help='Initial ego-minus-leader speed in challenge resets, m/s')
    p.add_argument('--challenge-gap-range', type=float, nargs=2, default=[.5,3.], help='Bumper gap for challenge resets, m')
    p.add_argument('--exploration-hold-steps', type=int, nargs=2, default=[5,15],
                   help='Hold each random command for this sampled number of ticks (default .5–1.5 s at 10 Hz)')
    p.add_argument('--learning-starts', type=int, default=1000)
    p.add_argument('--actor-lr', type=float, default=1e-4)
    p.add_argument('--critic-lr', type=float, default=1e-4)
    p.add_argument('--tau', type=float, default=.05)
    p.add_argument('--gamma-start', type=float, default=.85)
    p.add_argument('--gamma-final', type=float, default=.9999)
    p.add_argument('--alpha', type=float, default=.05)
    p.add_argument('--alpha-final', type=float, default=0.)
    p.add_argument('--anneal-updates', type=int, default=20000)
    p.add_argument('--safety-objective', choices=('collision','collision-following','following'), default='collision',
                   help='collision: contact failures only (default); collision-following: contact or max-gap failure; following: min/max-gap failure')
    p.add_argument('--min-gap', type=float, default=1., help='Lower failure boundary for following objective only')
    p.add_argument('--max-gap', type=float, default=9.,
                   help='Upper separation failure boundary (default 9 m); in pure collision mode, positive margin cap only')
    p.add_argument('--termination-gap-range',type=float,nargs=2,metavar=('MIN','MAX'),
                   help='Independent episode stop gaps in metres; safety margins/failure labels still use min-gap/max-gap. Default: stop at safety boundaries')
    p.add_argument('--max-separation', type=float, default=50.,
                   help='End an uninformative distant rollout as a truncation, retaining a bootstrap, not a failure')
    p.add_argument('--event-failure-margin', type=float, default=1.)
    p.add_argument('--switching-threshold', type=float, default=1.,
                   help='Provisional frozen-filter intervention epsilon in metres (default 1); separate from failure bounds, requires held-out validation')
    p.add_argument('--gap-range', type=float, nargs=2, default=[2., 8.])
    p.add_argument('--speed-range', type=float, nargs=2, default=[10., 20.])
    p.add_argument('--brake-delay-range', type=float, nargs=2, default=[1., 4.])
    p.add_argument('--brake-duration', type=float, default=2.)
    p.add_argument('--leader-lateral-accel', type=float, default=2., help='Leader curve speed bound in m/s²')
    p.add_argument('--leader-deceleration', type=float, default=2., help='Leader curve preview service-deceleration bound in m/s²')
    p.add_argument('--collision-grace', type=float, default=.02)
    p.add_argument('--dt', type=float, default=.1)
    p.add_argument('--port', type=int, default=2800)
    p.add_argument('--map', default='Town03')
    p.add_argument('--road-environment',choices=('town','straight'),default='town',
                   help='straight: generate a minimal junction-free OpenDRIVE road instead of a town')
    p.add_argument('--straight-road-length',type=float,default=1000.)
    p.add_argument('--straight-vehicle',default='vehicle.audi.tt',help='Straight-road vehicle; Audi TT has integrated headlights and tail lights')
    p.add_argument('--start-server', action='store_true')
    p.add_argument('--windowed', action='store_true',
                   help='Show a window with synchronized third-person view, native 224x224 model RGB and training status')
    p.add_argument('--real-time', action='store_true',
                   help='Pace decision iterations to at most 10 Hz at default dt; training may be slower')
    p.add_argument('--server', type=Path, default=Path('.local/carla-0.9.15/CarlaUE4.sh'))
    p.add_argument('--eval-interval', '--checkpoint-interval', dest='eval_interval', type=int, default=5000,
                   help='Evaluate between episodes every N training steps; 0 disables. Legacy checkpoint-interval is an alias; no periodic weights are saved')
    p.add_argument('--eval-cases', type=int, default=6, help='Held-out starts per nominal PID, half ordinary and half fast-closing')
    p.add_argument('--eval-seed', type=int, default=1000003)
    p.add_argument('--log-interval', type=int, default=100)
    p.add_argument('--action-diagnostics-interval', type=int, default=0,
                   help='Compare safety actor and action Q scan every N steps; show in window, log JSONL/W&B; 0 disables')
    p.add_argument('--wandb',action='store_true',help='Log metrics and evaluation plots to Weights & Biases')
    p.add_argument('--wandb-project',default='carla-sage')
    p.add_argument('--wandb-entity')
    p.add_argument('--wandb-name')
    p.add_argument('--wandb-group')
    p.add_argument('--wandb-mode',choices=('online','offline'),default='online')
    if defaults: p.set_defaults(**defaults)
    # Inherit training settings, but let this invocation choose runtime settings.
    bootstrap = argparse.ArgumentParser(add_help=False)
    bootstrap.add_argument('--resume',type=Path)
    resume_path = bootstrap.parse_known_args(argv)[0].resume
    runtime = {'outdir','resume','steps','server','port','start_server','windowed','real_time',
               'cpu','require_cuda','eval_interval','eval_cases','eval_seed','log_interval','action_diagnostics_interval','termination_gap_range',
               'wandb','wandb_project','wandb_entity','wandb_name','wandb_group','wandb_mode'}
    saved = None
    if resume_path:
        try:
            saved = torch.load(resume_path,map_location='cpu',weights_only=True)
        except (OSError,RuntimeError,ValueError) as error:
            p.error(f'Cannot read resume checkpoint: {error}')
        if saved.get('format_version') != 1 or saved.get('architecture') not in (
                OnlineHJ.architecture,'online_sage_rgb_longitudinal_v2'):
            p.error('Resume requires an online HJ/SAGE v2 checkpoint')
        known = {action.dest for action in p._actions}
        p.set_defaults(**{k:v for k,v in saved['config'].items() if k in known-runtime})
    args = p.parse_args(argv)
    if saved is not None:
        for key,value in saved['config'].items():
            current = getattr(args,key,None)
            if key not in runtime and key in known and str(current) != str(value):
                p.error(f'Resume must preserve {key}={value!r}; got {current!r}')
        if args.steps <= saved['steps']:
            p.error(f'--steps must exceed saved step {saved["steps"]}; it is the total target')
    if not 0 <= args.performance_gamma < 1 or not 0 < args.performance_tau <= 1:
        p.error('Require performance-gamma in [0,1), performance-tau in (0,1]')
    for key in ('target_gap','performance_alpha','collision_penalty','lane_penalty'):
        if not math.isfinite(getattr(args,key)) or getattr(args,key)<0: p.error(f'{key} must be finite and nonnegative')
    if args.target_gap <= 0: p.error('target-gap must be positive')
    integers = ('steps','episode_steps','history_frames','capacity','batch_size','anneal_updates','eval_cases','log_interval','learning_starts')
    if any(getattr(args,k)<1 for k in integers) or min(args.random_steps, args.collision_grace) < 0:
        p.error('Counts must be positive; random steps and collision grace must be nonnegative')
    if args.eval_interval < 0: p.error('eval-interval must be nonnegative')
    if not math.isfinite(args.straight_road_length) or args.straight_road_length<400:
        p.error('straight-road-length must be at least 400 m')
    if args.action_diagnostics_interval < 0: p.error('action-diagnostics-interval must be nonnegative')
    if args.eval_cases < 2: p.error('eval-cases must be at least two to cover ordinary and closing starts')
    if not 0 <= args.eval_seed < 2**32: p.error('eval-seed must be in [0, 2**32)')
    if args.capacity < max(args.batch_size, args.history_frames+1, args.learning_starts):
        p.error('Replay capacity must cover learning-starts, batch size and a history')
    if not 1 <= args.exploration_hold_steps[0] <= args.exploration_hold_steps[1]:
        p.error('Exploration hold bounds must be positive and ordered')
    if not 0 <= args.gamma_start <= args.gamma_final < 1 or not 0 < args.tau <= 1:
        p.error('Require 0 <= gamma-start <= gamma-final < 1 and 0 < tau <= 1')
    for k in ('actor_lr','critic_lr','dt','event_failure_margin','brake_duration','max_separation','max_gap','leader_lateral_accel','leader_deceleration'):
        if not math.isfinite(getattr(args,k)) or getattr(args,k)<=0: p.error(f'{k} must be finite and positive')
    for k in ('alpha','alpha_final','collision_grace','min_gap','max_gap'):
        if not math.isfinite(getattr(args,k)) or getattr(args,k)<0: p.error(f'{k} must be finite and nonnegative')
    if not math.isfinite(args.switching_threshold) or args.switching_threshold < 0:
        p.error('switching-threshold must be finite and nonnegative')
    for k in ('gap_range','speed_range','brake_delay_range'):
        lo,hi=getattr(args,k)
        if not all(map(math.isfinite,(lo,hi))) or not 0 < lo <= hi: p.error(f'{k} must have positive ordered bounds')
    if args.safety_objective == 'following' and not args.min_gap < args.gap_range[0] <= args.gap_range[1] < args.max_gap:
        p.error('Starting gaps must be strictly inside the failure bounds')
    if args.safety_objective == 'collision-following' and args.gap_range[1] >= args.max_gap:
        p.error('Starting gaps must be below max-gap')
    if args.max_separation <= args.gap_range[1]: p.error('max-separation must exceed the starting gap range')
    if args.termination_gap_range is not None:
        lo,hi=args.termination_gap_range
        if not all(map(math.isfinite,(lo,hi))) or not 0<=lo<hi:
            p.error('termination-gap-range requires finite 0 <= MIN < MAX')
        if not lo<args.gap_range[0]<=args.gap_range[1]<hi:
            p.error('Starting gaps must be inside termination-gap-range')
        if args.challenge_fraction and not lo<args.challenge_gap_range[0]<=args.challenge_gap_range[1]<hi:
            p.error('Challenge gaps must be inside termination-gap-range')
        if args.max_separation<=hi:
            p.error('max-separation must exceed termination MAX')
    if args.cpu and args.require_cuda: p.error('--cpu conflicts with --require-cuda')
    if args.gate_start_updates<0: p.error('gate-start-updates must be nonnegative')
    for key in ('challenge_fraction','exploration_episode_fraction'):
        if not math.isfinite(getattr(args,key)) or not 0<=getattr(args,key)<=1: p.error(f'{key} must be in [0,1]')
    if not all(map(math.isfinite,args.closing_speed_range)) or not 0<=args.closing_speed_range[0]<=args.closing_speed_range[1]<=args.speed_range[1]-args.speed_range[0]:
        p.error('Closing speeds must be nonnegative, ordered and fit inside speed-range')
    if not all(map(math.isfinite,args.challenge_gap_range)) or not 0<args.challenge_gap_range[0]<=args.challenge_gap_range[1]<args.max_separation:
        p.error('Challenge gaps must be positive, ordered and below max-separation')
    if args.safety_objective!='collision' and args.challenge_fraction>0:
        low=args.min_gap if args.safety_objective=='following' else 0.
        if not low<args.challenge_gap_range[0]<=args.challenge_gap_range[1]<args.max_gap:
            p.error('Challenge gaps must start inside the selected following constraints')
    for key in ('pid_kp','pid_ki','pid_kd'):
        if not math.isfinite(getattr(args,key)) or getattr(args,key)<0:
            p.error(f'{key} must be finite and nonnegative')
    if args.pid_checkpoint:
        if args.performance_policy != 'pid': p.error('--pid-checkpoint requires --performance-policy pid')
        if args.nominal_conditions: p.error('--pid-checkpoint replaces --nominal-conditions')
        if not args.pid_checkpoint.is_file(): p.error(f'Missing generalist checkpoint: {args.pid_checkpoint}')
    args.nominal_conditions=args.nominal_conditions or [args.condition]
    if len(set(args.nominal_conditions))!=len(args.nominal_conditions): p.error('Duplicate nominal PID profiles')
    for name in args.nominal_conditions if args.performance_policy == 'pid' and not args.pid_checkpoint else []:
        if name not in [f'{t}_{w}' for t in ('day','sunset','night') for w in ('sunny','cloudy','rainy')]:
            p.error(f'Unknown nominal PID profile: {name}')
        if not all((args.pid_root/name/f).is_file() for f in ('pid.json','cnn.pth')):
            p.error(f'Missing contextual PID files in {args.pid_root/name}')
    return args


def append_json(path, record):
    with path.open('a') as stream: stream.write(json.dumps(record, allow_nan=False)+'\n')


def main(args=None, *, manifest_overrides=None):
    args = parse_args() if args is None else args
    if args.require_cuda and not torch.cuda.is_available(): raise RuntimeError('CUDA requested but unavailable')
    device = torch.device('cuda' if torch.cuda.is_available() and not args.cpu else 'cpu')
    torch.set_num_threads(4)
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    args.outdir.mkdir(parents=True, exist_ok=False)
    # Save the input/target contract before connecting to the simulator.
    manifest = {k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()}
    manifest.update(actor_inputs='ordered RGB history' + (' + ego speed / 30' if args.include_speed else ''),
                    critic_inputs='actor inputs + normalized longitudinal request',
                    steering='fixed route tracker, outside learned actor',
                    margin=('min(gap,max_gap-gap); no lower-gap stopping cutoff; contact/lane event supplies negative successor margin'
                            if args.safety_objective == 'collision-following' else
                            'min(gap,max_gap) (positive cap only); contact supplies negative successor margin; lane departures truncate without failure'
                            if args.safety_objective == 'collision' else
                            'min(gap-min_gap, max_gap-gap); collision/lane event forces negative successor margin'),
                    time_limit='truncation with bootstrap, never a failure label',
                    distant_rollout='gap >= max_separation is a truncation with bootstrap, not a failure',
                    intervention='Q(PID) < switching_threshold selects full braking (-3, brake=1) after gate_start_updates; explicit exploration episodes bypass the gate',
                    sensor_contract='camera capture ID equals snapshot ID; consecutive captures exactly dt apart',
                    collision_contract='per-step grace and final callback drain before replay admission; event sensors have no strict delivery watermark',
                    replay='new experience only; completed episodes admitted after event drain; one update per subsequent simulator step',
                    interpretation='approximate discounted HJ safety under the sampled leader behavior, not adversarial-disturbance certification',
                    comparison='freeze models for held-out evaluation; match steering, observations, scenarios, and control limits with PID baselines')
    manifest.update(leader='route tracker + curvature-aware speed tracking, with scheduled full-brake pulse',
                    exploration='uniform longitudinal requests held for sampled exploration_hold_steps; reset each episode; no controller imitation',
                    lane_validation='both vehicle footprints checked against planned route every decision; invalid leader rollouts excluded from replay')
    if getattr(args,'road_environment','town')=='straight':
        from .straight_weather import make_weather,CAMERA_ATTRIBUTES
        manifest['controlled_scene']=dict(vehicle=args.straight_vehicle,
            surface='7.2 m asphalt, dashed center line, grass verges; visual meshes do not collide',
            weather=str(make_weather(args.condition)),
            camera_attributes=CAMERA_ATTRIBUTES,
            lights='position + low beam at sunset/night and in rain; brake lamps follow brake commands',
            particles='spectator follows ego for offscreen precipitation rendering')
    manifest['episode_gap_termination']=dict(bounds=getattr(args,'termination_gap_range',None),
        default='safety boundaries when no override is supplied',
        replay='only transitions through the first safety failure are admitted; later rollout remains diagnostic')
    if args.performance_policy == 'sac':
        from .online_sage import OnlineSAGE, GoalPolicy
        learner = OnlineSAGE(args,device)
        nominal_policies = {'goal':GoalPolicy(learner,args.target_gap)}
        manifest.update(performance='SAC following actor and twin critics; all executed transitions retained',
            reward='1 - abs(gap-target_gap)/target_gap - .2*abs(ego_speed-leader_speed)/5 - .05*((action-previous_action)/3)^2; contact/lane terminal penalties',
            intervention='Q(goal action) < switching_threshold selects safety actor after initial random transitions',
            paper_differences='CARLA following task; ResNet18 with four frames, fixed route steering, collision margin; replay/batch/lr chosen for visual memory budget')
    elif args.performance_policy == 'safety':
        learner = OnlineHJ(args, device)
        nominal_policies = {'safety':SafetyPolicy(learner,args.target_gap)}
        manifest.update(performance='single HJ safety actor; no separate goal actor or PID',
            intervention='disabled; random warmup followed by direct stochastic safety actor execution')
    else:
        learner = OnlineHJ(args, device)
        if getattr(args,'pid_checkpoint',None):
            nominal_policies={'generalist':GeneralistPID(args.pid_checkpoint,device,
                target_gap=args.target_gap,kp=args.pid_kp,ki=args.pid_ki,kd=args.pid_kd)}
        else:
            nominal_policies={name:ContextPID(args.pid_root/name,device) for name in args.nominal_conditions}
        manifest['nominal_policies']={name:dict(path=str(policy.folder.resolve()),cnn_sha256=policy.sha256,
            pid={k:policy.profile[k] for k in ('target_gap','kp','ki','kd')}) for name,policy in nominal_policies.items()}
        if getattr(args,'pid_checkpoint',None):
            manifest['nominal_policies']['generalist']['checkpoint']=str(args.pid_checkpoint.resolve())
    manifest['gate_warmup']=('initial random transitions, then learned goal proposals gated by HJ safety value; no safety guarantee during training'
                            if args.performance_policy == 'sac' else
                            'initial random transitions, then direct safety actor execution; no gate'
                            if args.performance_policy == 'safety' else
                            'ungated contextual PID after initial random steps, until enough critic updates; no safety guarantee during training')
    manifest.update(architecture=learner.architecture if hasattr(learner,'architecture') else 'test',
        critic_fusion='vision -> Linear128/LayerNorm/ReLU; normalized action -> Linear32/ReLU; concatenated value head',
        safety_value_bounds=learner.safety_bounds if hasattr(learner,'safety_bounds') else None,
        evaluation_validation='full-braking setup validation with up to 30 deterministic attempts per case; suite fixed before policy evaluation')
    restored_steps = learner.restore_training(args.resume) if getattr(args,'resume',None) else 0
    if getattr(args,'resume',None):
        manifest['continuation'] = dict(checkpoint=str(args.resume),steps=restored_steps,
            updates=learner.updates,replay='fresh; collect learning_starts rows with restored policy before updating',
            simulator='new episodes',rng='reseeded; not an exact trajectory continuation')
        print(f'Restored step {restored_steps}, update {learner.updates}; rebuilding replay before further updates.',flush=True)
        random.seed(args.seed+restored_steps); np.random.seed((args.seed+restored_steps) % 2**32)
        torch.manual_seed(args.seed+restored_steps)
    manifest.update(manifest_overrides or {})
    (args.outdir/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    replay = Replay(args.capacity)
    import carla
    from .braking_scenario import BrakingScenario, InvalidBrakingScenario
    from .collect_braking import start_server
    from .closed_loop import read_snapshot
    from collect_parallel import stop
    process, env, world, original_settings, viewer = None, None, None, None, None
    from .wandb_logging import WandbLogger
    tracker=None
    steps, episode = restored_steps, restored_steps
    try:
        tracker=WandbLogger(args,manifest)
        if args.windowed: viewer = TrainingViewer()
        scenario_class = with_route_aware_leader(BrakingScenario,args.leader_lateral_accel,args.leader_deceleration)
        scenario_class = with_closing_starts(scenario_class,args.leader_lateral_accel,args.leader_deceleration)
        if getattr(args,'road_environment','town')=='straight':
            from .straight_weather import with_straight_weather
            scenario_class=with_straight_weather(scenario_class)
        if viewer is not None: scenario_class = with_preview(scenario_class,viewer)
        if args.start_server: process = start_server(args)
        client = carla.Client('127.0.0.1', args.port); client.set_timeout(120.)
        if getattr(args,'road_environment','town')=='straight':
            from .straight_road import create_world
            world=create_world(client,args.straight_road_length,args.outdir/'road.xodr')
        else:
            world = client.load_world(args.map)
        original_settings = world.get_settings()
        from .evaluate_online_hj import evaluate
        next_eval = ((steps//args.eval_interval+1)*args.eval_interval) if args.eval_interval else 0
        last_eval = None
        if args.eval_interval:
            evaluate(learner, nominal_policies, client, world, scenario_class, args, steps, viewer,tracker=tracker)
            last_eval = steps
        while steps < args.steps:
            # Different route seed on every episode; retain the complete rollout plan.
            for attempt in range(30):
                from .straight_road import scenarios_for_args
                scenario = scenarios_for_args(world,1,args.seed+episode*1009+attempt,args.episode_steps,args)[0]
                start=sample_start(args,random)
                speed,gap=start['leader_speed'],start['gap']
                # Leave setup space to settle differential wheel speeds before
                # teleporting the ego into its intended close observation state.
                scenario['leader']=list(scenario['full_route'][20])
                scenario['leader'][2]+=.5
                scenario['route']=scenario['full_route'][20:]
                delay = random.uniform(*args.brake_delay_range)
                blueprint=args.straight_vehicle if getattr(args,'road_environment','town')=='straight' else 'vehicle.audi.a2'
                rollout = SimpleNamespace(condition=args.condition, blueprint=blueprint,
                    initial_gap=gap+start['closing_speed']*1.5, initial_speed=speed,
                    ego_initial_speed=start['ego_speed'],leader_brake_test=True, leader_brake_after=delay,
                    leader_brake_duration=args.brake_duration, stationary_leader_test=False,
                    graphics='vulkan', camera=True, real_time=False,
                    handoff_gap_bounds=[max(.05,gap-.15),gap+.15], handoff_speed_bounds=args.speed_range)
                try:
                    env = scenario_class(client, rollout, args.dt, scenario, reload_world=False)
                    break
                except InvalidBrakingScenario as error:
                    append_json(args.outdir/'rejected_starts.jsonl', dict(episode=episode, attempt=attempt, error=str(error)))
            else: raise RuntimeError('Thirty invalid starting scenarios; inspect rejected_starts.jsonl')
            nominal_name=random.choice(list(nominal_policies))
            nominal=nominal_policies[nominal_name]; nominal.reset()
            append_json(args.outdir/'plan.jsonl', dict(episode=episode,scenario=scenario,**start,
                        brake_after=delay,nominal_condition=nominal_name))
            history = FrameHistory(args.history_frames, episode, args.dt)
            observation = history.capture(env)
            _,_,_,handoff_gap=read_snapshot(env,args.dt)
            leader_velocity=env.world.get_snapshot().find(env.nonego.id).get_velocity()
            handoff_leader_speed=math.sqrt(leader_velocity.x**2+leader_velocity.y**2+leader_velocity.z**2)
            append_json(args.outdir/'measured_starts.jsonl',dict(episode=episode,frame=observation.frame_ids[-1],
                        gap=handoff_gap,ego_speed=observation.speed,leader_speed=handoff_leader_speed,
                        speed_difference=observation.speed-handoff_leader_speed,challenge=start['challenge']))
            rows, metrics, ending = [], None, 'time_limit'
            invalid_leader_frame = None
            explorer = RandomExplorer(args.exploration_hold_steps)
            action_counts=Counter()
            previous_action = 0.
            for _ in range(min(args.episode_steps, args.steps-steps)):
                iteration_started = time.monotonic()
                frame, _, _, gap = read_snapshot(env, args.dt)
                if frame != observation.frame_ids[-1]: raise RuntimeError('World advanced before action selection')
                leader_velocity=env.world.get_snapshot().find(env.nonego.id).get_velocity()
                leader_speed=math.sqrt(leader_velocity.x**2+leader_velocity.y**2+leader_velocity.z**2)
                if args.performance_policy == 'safety' and steps < args.random_steps:
                    nominal_action,predicted_gap=0.,None
                else:
                    nominal_action,predicted_gap=nominal.propose(observation,args.dt)
                decision=behavior_action(learner,observation,nominal_action,explorer,steps,args,start['exploration_episode'])
                if args.performance_policy == 'sac' and decision['source'] in ('contextual_pid','pid_critic_warmup'):
                    decision['source'] = 'goal_actor'
                action=decision['action']; action_counts[decision['source']]+=1
                diagnostic_interval=getattr(args,'action_diagnostics_interval',0)
                if diagnostic_interval and steps % diagnostic_interval == 0:
                    diagnostic=dict(step=steps,frame=frame,gap=gap,
                        closing_speed=observation.speed-leader_speed,
                        **learner.action_diagnostics(observation,action))
                    append_json(args.outdir/'action_diagnostics.jsonl',diagnostic)
                    tracker.log({f'action_diagnostics/{key}':value for key,value in diagnostic.items()
                                 if key not in ('step','frame')},steps)
                    if viewer is not None: viewer.action_diagnostics=diagnostic
                steering = -env.tracker(env.ego).steering()
                if viewer is not None:
                    viewer.show(observation,step=steps,action=action,steering=steering,gap=gap,
                                q=learner.score(observation,action),updates=learner.updates,
                                phase=(f'Replay warmup: {len(replay)} / {args.learning_starts}'
                                       if len(replay) < args.learning_starts else 'Online actor-critic updates'),metrics=metrics,
                                actor_active=decision['intervened'] and decision['source']=='safety_actor',epsilon=args.switching_threshold,
                                controller_label=decision['source'].replace('_',' ').upper())
                if env.world.get_snapshot().frame != frame: raise RuntimeError('World advanced during policy inference')
                env.step((action, steering))
                successor = history.capture(env)
                next_frame, _, _, next_gap = read_snapshot(env, args.dt)
                if next_frame != successor.frame_ids[-1]: raise RuntimeError('Successor gap and camera frame disagree')
                time.sleep(args.collision_grace)
                ego_lane = planned_lane_status(env,env.ego)
                leader_lane = planned_lane_status(env,env.nonego)
                if not ego_lane['inside'] and next_frame not in env.departures:
                    env.departures.append(next_frame)
                if not leader_lane['inside'] or any(other != env.ego.id for _,other in env.leader_events):
                    invalid_leader_frame = next_frame
                failed = gap_failure(next_gap,args)
                rows.append(Transition(observation,action,successor,margin(gap,args),margin(next_gap,args),failed))
                if args.performance_policy == 'sac':
                    from .online_sage import following_reward
                    velocity=env.world.get_snapshot().find(env.nonego.id).get_velocity()
                    next_leader_speed=math.sqrt(velocity.x**2+velocity.y**2+velocity.z**2)
                    rows[-1].reward=following_reward(next_gap,successor.speed-next_leader_speed,action,previous_action,args.target_gap)
                previous_action = action
                steps += 1
                transition_record=dict(step=steps,episode=episode,frame=frame,next_frame=next_frame,
                    history_frames=observation.frame_ids,successor_history_frames=successor.frame_ids,
                    action=action,steering=steering,gap=gap,next_gap=next_gap,failed_gap=failed,
                    ego_lane=ego_lane,leader_lane=leader_lane,behavior=decision,
                    nominal_condition=nominal_name,predicted_gap=predicted_gap,
                    ego_speed=observation.speed,leader_speed=leader_speed,closing_rate=(gap-next_gap)/args.dt,
                    challenge_start=start['challenge'])
                append_json(args.outdir/'transitions.jsonl',transition_record)
                tracker.transition(transition_record)
                if len(replay) >= args.learning_starts:
                    metrics = learner.update(replay.sample(args.batch_size,device))
                    if steps % args.log_interval == 0:
                        record=dict(step=steps,replay_rows=len(replay),**metrics)
                        append_json(args.outdir/'updates.jsonl',record); print(json.dumps(record),flush=True)
                if steps % args.log_interval == 0:
                    progress = dict(step=steps,replay_rows=len(replay),updates=learner.updates,
                                    phase='replay_warmup' if len(replay) < args.learning_starts else 'learning',
                                    safety_objective=args.safety_objective,
                                    failure_rows_in_replay=sum(r.failed for r in replay.rows))
                    append_json(args.outdir/'progress.jsonl',progress)
                    tracker.progress(progress,metrics)
                    if len(replay) < args.learning_starts: print(json.dumps(progress),flush=True)
                observation = successor
                if args.real_time: time.sleep(max(0.,args.dt-(time.monotonic()-iteration_started)))
                if invalid_leader_frame is not None:
                    ending = 'invalid_leader_rollout'
                    break
                if gap_termination(next_gap,args) or env.events or env.departures:
                    ending = ('collision' if env.events else
                              'gap_failure' if gap_termination(next_gap,args) and failed else
                              'gap_truncation' if gap_termination(next_gap,args) else
                              'lane_truncation' if args.safety_objective == 'collision' else 'lane_failure')
                    break
                if next_gap >= args.max_separation:
                    ending = 'separation_truncation'
                    break
            events = env.measurements()
            # Either contact sensor can identify ego/leader contact. Preserve
            # failures where the ego hits the leader and pushes it off its lane.
            events['collision_events'].extend((frame,env.nonego.id) for frame,other in env.leader_events if other == env.ego.id)
            external_contacts = [frame for frame,other in env.leader_events if other != env.ego.id
                                 and rows[0].observation.frame_ids[-1] <= frame <= rows[-1].successor.frame_ids[-1]]
            if external_contacts:
                invalid_leader_frame = min(external_contacts + ([invalid_leader_frame] if invalid_leader_frame is not None else []))
            contact_before_departure = invalid_leader_frame is not None and any(
                rows[0].observation.frame_ids[-1] <= frame <= invalid_leader_frame for frame,_ in events['collision_events'])
            invalid_leader = invalid_leader_frame is not None and not contact_before_departure
            rows = finalize_episode(rows,events['collision_events'],events['lane_departure_frames'],args.event_failure_margin,
                                    include_lane_failure=args.safety_objective != 'collision')
            if args.performance_policy == 'sac':
                from .online_sage import finalize_performance
                rows = finalize_performance(rows,events['collision_events'],events['lane_departure_frames'],args)
            if invalid_leader:
                append_json(args.outdir/'invalid_rollouts.jsonl',dict(episode=episode,frame=invalid_leader_frame,
                            reason='leader left planned lane or collided with external object',discarded_rows=len(rows)))
            else:
                replay.extend(first_hit_rows(rows))
            record=dict(episode=episode,step=steps,rows=len(rows),ending=ending,
                        action_sources=dict(action_counts),challenge_start=start['challenge'],
                        exploration_episode=start['exploration_episode'],nominal_condition=nominal_name,
                        admitted_to_replay=not invalid_leader,
                        failed=any(r.failed for r in rows),
                        safety_replay_rows=len(first_hit_rows(rows)) if not invalid_leader else 0,
                        minimum_safety_margin=min(r.next_margin for r in rows) if rows else None,
                        last_admitted_frame=first_hit_rows(rows)[-1].successor.frame_ids[-1] if rows and not invalid_leader else None,
                        collision_events=events['collision_events'],lane_departures=events['lane_departure_frames'])
            if args.performance_policy == 'sac': record['performance_return']=sum(r.reward for r in rows)
            if record['failed'] and ending == 'time_limit': record['ending'] = 'delayed_event_failure'
            if contact_before_departure: record['ending'] = 'collision_or_lane_failure'
            if args.safety_objective == 'collision' and record['failed']: record['ending'] = 'collision'
            append_json(args.outdir/'episodes.jsonl',record); print(json.dumps(record),flush=True)
            tracker.episode(record)
            env.close(); env=None
            episode += 1
            if args.eval_interval and steps >= next_eval:
                evaluate(learner, nominal_policies, client, world, scenario_class, args, steps, viewer,tracker=tracker)
                last_eval = steps
                next_eval = (steps//args.eval_interval+1)*args.eval_interval
        if args.eval_interval and last_eval != steps:
            evaluate(learner, nominal_policies, client, world, scenario_class, args, steps, viewer,tracker=tracker)
        learner.save(args.outdir/'checkpoint_final.pt',steps)
    except BaseException:
        learner.save(args.outdir/'checkpoint_interrupted.pt',steps)
        raise
    finally:
        exit_code=0 if sys.exc_info()[0] is None else 1
        try:
            if env is not None: env.close()
            if viewer is not None: viewer.close()
            if world is not None and original_settings is not None: world.apply_settings(original_settings)
            if process is not None: stop(process)
        finally:
            if tracker is not None: tracker.finish(exit_code=exit_code)


if __name__ == '__main__': main()
