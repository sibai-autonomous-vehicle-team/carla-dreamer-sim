"""Pinned DINOv2 Small CLS encoder for ordered 224x224 RGB observations.

Run `python -m training.dino_models` once on a login node to cache the official
source and weights before submitting a GPU job. No CARLA or GPU is required.
"""
from copy import deepcopy

import torch
from torch import nn

from .visual_models import PREPROCESS


DINO_REPOSITORY='facebookresearch/dinov2:7764ea0f912e53c92e82eb78a2a1631e92725fc8'
DINO_MODEL='dinov2_vits14'


def load_backbone(pretrained=True):
    # PyTorch's native attention fallback is used; no xformers installation needed.
    return torch.hub.load(DINO_REPOSITORY,DINO_MODEL,pretrained=pretrained,
                          trust_repo=True,skip_validation=True)


class DinoVisual(nn.Module):
    def __init__(self,prototype,history,include_speed,unfreeze_blocks):
        super().__init__()
        self.network=deepcopy(prototype)
        if getattr(self.network,'chunked_blocks',False):
            raise ValueError('Expected the unchunked official DINOv2 hub backbone')
        if not 0<=unfreeze_blocks<=len(self.network.blocks):
            raise ValueError('Invalid number of DINO transformer blocks')
        self.network.requires_grad_(False)
        if unfreeze_blocks:
            for block in self.network.blocks[-unfreeze_blocks:]:block.requires_grad_(True)
            self.network.norm.requires_grad_(True)
        self.history_frames,self.include_speed=history,include_speed
        self.width=history*self.network.embed_dim+int(include_speed)
        self.register_buffer('image_mean',torch.tensor(PREPROCESS['mean']).view(1,3,1,1))
        self.register_buffer('image_std',torch.tensor(PREPROCESS['std']).view(1,3,1,1))
        self.network.eval()

    def train(self,mode=True):
        super().train(mode)
        # Evaluation mode suppresses any stochastic encoder layers without
        # disabling gradients through the selected blocks or final LayerNorm.
        self.network.eval()
        return self

    def features(self,images,speed):
        if images.dtype!=torch.uint8 or images.ndim!=5 or tuple(images.shape[1:])!=(self.history_frames,224,224,3):
            raise ValueError('DINO expects ordered uint8 RGB [B,history,224,224,3]')
        batch=images.shape[0]
        x=images.reshape(-1,224,224,3).permute(0,3,1,2).float()/255.
        x=(x-self.image_mean)/self.image_std
        cls=self.network.forward_features(x)['x_norm_clstoken']
        features=cls.reshape(batch,-1)
        return torch.cat((features,speed),-1) if self.include_speed else features


if __name__=='__main__':
    torch.set_num_threads(4)
    model=load_backbone()
    print(f'Cached {DINO_MODEL}: {sum(p.numel() for p in model.parameters()):,} parameters; {DINO_REPOSITORY}')
