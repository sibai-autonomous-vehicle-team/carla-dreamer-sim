"""Nine specialist proposals and critic bounds for the visible braking runner."""
import json
from pathlib import Path

import torch
from torch import nn
from torch.func import functional_call, stack_module_state, vmap
from torchvision.models import resnet18

from run_gap_pid import GapPID
from .conformal_critics import CONDITIONS, ConformalCritics


class ConformalDisplay:
    def __init__(self, root, calibration, device):
        self.device = torch.device(device)
        self.predictor = ConformalCritics(root, calibration, device=device)
        models, self.profiles = [], []
        for condition in CONDITIONS:
            folder = Path('specialists') / condition
            profile = json.loads((folder / 'pid.json').read_text())
            model = resnet18(weights=None)
            model.fc = nn.Linear(model.fc.in_features, 1)
            model.load_state_dict(torch.load(folder / 'cnn.pth', map_location='cpu', weights_only=True)['model'])
            models.append(model.eval().requires_grad_(False))
            self.profiles.append(profile)
        params, buffers = stack_module_state(models)
        self.params = {k: v.to(self.device) for k, v in params.items()}
        self.buffers = {k: v.to(self.device) for k, v in buffers.items()}
        self.model = models[0].to('meta')
        self.mean = torch.tensor([.485, .456, .406], device=self.device).view(1, 3, 1, 1)
        self.std = torch.tensor([.229, .224, .225], device=self.device).view(1, 3, 1, 1)
        self.reset()

    def reset(self):
        self.pids = [GapPID(p['target_gap'], p['kp'], p['ki'], p['kd']) for p in self.profiles]

    @torch.inference_mode()
    def decide(self, history, steer, dt):
        frame = history[-1].to(self.device).permute(2, 0, 1).float().unsqueeze(0) / 255
        frame = (frame - self.mean) / self.std
        def estimate(params, buffers):
            return functional_call(self.model, (params, buffers), (frame,)).reshape(())
        with torch.backends.cudnn.flags(enabled=False):
            gaps = vmap(estimate)(self.params, self.buffers).cpu().tolist()
        actions = [(pid.update(gap, dt), steer) for pid, gap in zip(self.pids, gaps)]
        raw, lower = self.predictor.predict(history, actions)
        return actions, gaps, raw.cpu().tolist(), lower.cpu().tolist()


def show_bounds(display, image, raw, lower, trajectory):
    pygame = display.pygame
    for event in pygame.event.get():
        if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
            raise KeyboardInterrupt('Evaluation window closed')
    display.window.blit(pygame.surfarray.make_surface(image.swapaxes(0, 1)), (0, 0))
    font = pygame.font.Font(None, 22)
    panel = pygame.Surface((1248, 124), pygame.SRCALPHA)
    panel.fill((12, 18, 26, 225))
    display.window.blit(panel, (16, 12))
    display.window.blit(font.render(f'Trajectory {trajectory} | day-sunny controller | Q / lower bound (m)', True, (255,255,255)), (26, 18))
    for i, condition in enumerate(CONDITIONS):
        row, col = divmod(i, 3)
        label = f'{condition:14s}  {raw[i]:+6.2f} / {lower[i]:+6.2f}'
        display.window.blit(font.render(label, True, (235, 242, 250)), (26 + col * 410, 45 + row * 27))
    pygame.display.flip()
