"""Online visual HJ safety actor-critic with pretrained DINOv2 Small CLS features.

Four ordered camera frames, existing synchronized CARLA training/evaluation,
and the unchanged HJ backup. Last two transformer blocks + final norm adapt by
default. The remaining backbone stays frozen; final features are not cached.
"""
import math
from unittest.mock import patch

import torch
from torch import nn

from . import train_online_hj as hj
from .dino_models import DinoVisual,load_backbone,DINO_MODEL,DINO_REPOSITORY


def contract(args):
    return dict(kind='dinov2_cls_rgb',model=DINO_MODEL,repository=DINO_REPOSITORY,
        shape=[224,224,3],dtype='uint8',channel_order='RGB',
        mean=hj.PREPROCESS['mean'],std=hj.PREPROCESS['std'],
        embedding_per_frame=384,history=args.history_frames,
        fusion='ordered concatenation of per-frame normalized CLS tokens',
        unfreeze_blocks=args.dino_unfreeze_blocks,
        final_norm_trainable=bool(args.dino_unfreeze_blocks),encoder_lr=args.encoder_lr,
        encoder_mode='eval, with gradients enabled for selected parameters',
        feature_cache=False)


def model_classes(prototype,unfreeze_blocks):
    class DinoActor(DinoVisual):
        forward=hj.Actor.forward
        sample_features=hj.Actor.sample_features

        def __init__(self,history,include_speed,*unused):
            super().__init__(prototype,history,include_speed,unfreeze_blocks)
            self.head=nn.Sequential(nn.Linear(self.width,256),nn.ReLU(),nn.Linear(256,2))
            nn.init.uniform_(self.head[-1].weight,-.003,.003)
            nn.init.zeros_(self.head[-1].bias)

    class DinoCritic(DinoVisual):
        forward=hj.Critic.forward
        value=hj.Critic.value

        def __init__(self,history,include_speed,*unused,value_bounds=None):
            super().__init__(prototype,history,include_speed,unfreeze_blocks)
            self.value_bounds=value_bounds
            self.state_head=nn.Sequential(nn.Linear(self.width,128),nn.LayerNorm(128),nn.ReLU())
            self.action_head=nn.Sequential(nn.Linear(1,32),nn.ReLU())
            self.head=nn.Sequential(nn.Linear(160,256),nn.ReLU(),nn.Linear(256,128),nn.ReLU(),nn.Linear(128,1))
    return DinoActor,DinoCritic


def optimizer(models,head_lr,encoder_lr):
    if isinstance(models,DinoVisual):models=[models]
    encoders=[];heads=[]
    for model in models:
        for name,param in model.named_parameters():
            if param.requires_grad:
                (encoders if name.startswith('network.') else heads).append(param)
    groups=[dict(params=heads,lr=head_lr,name='heads')]
    if encoders:groups.append(dict(params=encoders,lr=encoder_lr,name='encoder'))
    return torch.optim.Adam(groups)


class DinoHJ(hj.OnlineHJ):
    architecture='online_hj_dinov2_small_cls_v1'

    def __init__(self,args,device):
        prototype=load_backbone(pretrained=not args.no_pretrained and not getattr(args,'resume',None))
        actor,critic=model_classes(prototype,args.dino_unfreeze_blocks)
        with patch.multiple(hj,Actor=actor,Critic=critic):
            super().__init__(args,device)
        self.actor_optimizer=optimizer(self.actor,args.actor_lr,args.encoder_lr)
        self.critic_optimizer=optimizer(self.critics,args.critic_lr,args.encoder_lr)

    def save(self,path,steps):
        with patch.object(hj,'PREPROCESS',contract(self.args)):
            super().save(path,steps)


def extra_arguments(parser):
    parser.add_argument('--dino-unfreeze-blocks',type=int,default=2,
        help='Number of final DINO transformer blocks to train (0 freezes encoder); final norm also trains when nonzero')
    parser.add_argument('--encoder-lr',type=float,default=1e-5)


def parse_args(argv=None):
    with patch.object(hj,'OnlineHJ',DinoHJ):
        args=hj.parse_args(dict(performance_policy='safety',history_frames=4,
            road_environment='straight',condition='day_sunny',steps=100000,
            safety_objective='following',min_gap=1.,max_gap=9.,gap_range=[2.,8.],
            challenge_gap_range=[1.5,3.],termination_gap_range=[.2,11.],
            gamma_start=.99,gamma_final=.99,alpha=.05,alpha_final=.05,
            exploration_episode_fraction=0.,port=2980),description=__doc__,argv=argv,
            extra_arguments=extra_arguments)
    if args.performance_policy!='safety':raise ValueError('DINO trainer requires --performance-policy safety')
    if args.freeze_encoder:raise ValueError('Use --dino-unfreeze-blocks 0 to freeze DINO')
    if args.no_pretrained and not args.resume:raise ValueError('DINO training requires pretrained initialization')
    if not 0<=args.dino_unfreeze_blocks<=12:raise ValueError('DINO unfreeze blocks must be between 0 and 12')
    if not math.isfinite(args.encoder_lr) or args.encoder_lr<=0:raise ValueError('encoder-lr must be positive and finite')
    return args


def main():
    args=parse_args()
    with patch.object(hj,'OnlineHJ',DinoHJ):
        hj.main(args,manifest_overrides=dict(
            actor_inputs='ordered per-frame DINOv2 Small CLS embeddings'+(' + ego speed / 30' if args.include_speed else ''),
            critic_inputs='same visual embeddings + normalized longitudinal request',
            preprocessing=contract(args),encoder=contract(args),
            critic_fusion='DINO CLS concat -> Linear128/LayerNorm/ReLU + action Linear32/ReLU',
            comparison='new visual backbone and fixed gamma/entropy; does not isolate encoder effects from schedule effects'))


if __name__=='__main__':main()
