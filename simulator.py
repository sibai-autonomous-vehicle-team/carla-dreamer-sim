from generate_dataset import collect_carla_dataset
from carla_dset import load_carla_slice_train_val

def main():
    # collect_carla_dataset(
    #     outdir="./dataset/",
    #     n_traj=10,
    #     seed=42
    # )
    load_carla_slice_train_val(
        data_path="./dataset/",
        n_rollout=10,
        normalize_action=True,
        split_ratio=0.8,
        num_hist=5,
        num_pred=5,
        frameskip=1
    )

if __name__ == "__main__":
    main()
