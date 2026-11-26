"""Train SAGE-style SAC following and HJ safety policies in synchronized CARLA.

python -m training.train_online_sage --outdir training/_local/sage_cloudy \
    --condition day_cloudy --start-server --require-cuda --steps 250000

This adapts SAGE Algorithm 2 to longitudinal following, not a reproduction of its
racing benchmark. Four ResNet RGB frames + ego speed replace the paper's VAE.
Both policies train from all executed transitions. After 2000 initial random
steps, the goal actor proposes and the HJ gate selects the safety actor if Q < 1.
No recurring random episodes, PID imitation, or PID files are required.
Task reward and target gap are in the run manifest. Safety remains collision-only.
Replay size 10000, batch size 8 and LR 1e-4 retain the visual trainer's memory
budget; paper settings are replay 250000, batch 256 and LR .003. Entropy .2,
performance gamma .99, safety gamma .85 to .9999, and target taus .005/.05.
Final/interrupted checkpoints contain both pairs. Training resume is not supported.
"""
from .train_online_hj import main as train, parse_args


def main():
    args=parse_args(dict(performance_policy='sac',include_speed=True,random_steps=2000,
        learning_starts=2000,gate_start_updates=0,exploration_episode_fraction=0.,
        exploration_hold_steps=[1,1],alpha=.2,alpha_final=.2,steps=250000),description=__doc__)
    if args.performance_policy != 'sac': raise ValueError('Use train_online_hj for the PID proposer')
    train(args)


if __name__ == '__main__': main()
