# CARLA dataset collection on Linux

Verified on November 7, 2026 with Ubuntu 24.04.5, an RTX 4070 (12 GB), NVIDIA
driver 595.99.02, Python 3.10.22, and CARLA server/client 0.9.15. This uses the
precompiled Linux release directly; Docker and a CUDA PyTorch installation are
not required. Run all commands from the repository root unless noted otherwise.

## Current machine: start collecting

The simulator is installed at `.local/carla-0.9.15` and Python at `.venv`.
If the server is already running on port 2000, skip the first command.

Terminal A (leave running; Ctrl+C stops the server):

```bash
VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json \
  .local/carla-0.9.15/CarlaUE4.sh \
  -RenderOffScreen -quality-level=Epic -nosound -carla-rpc-port=2000
```

Use **Epic** quality on this machine. Low quality repeatedly crashed in Unreal
Engine's Vulkan graphics pipeline creation when loading Town03. Selecting the
NVIDIA driver alone did not fix that crash; Epic quality did.
Offscreen rendering still produces camera images. Do not enable CARLA's
`no_rendering_mode` for this dataset.

Terminal B: check connectivity, then collect a small dataset into a new directory:

```bash
.venv/bin/python -B -c 'import carla; c=carla.Client("127.0.0.1", 2000); c.set_timeout(30); print("client:", c.get_client_version(), "server:", c.get_server_version()); print(c.get_world().get_map().name)'
.venv/bin/python -B -u generate_dataset.py --headless --n-traj 3 --outdir ./dataset/smoke
```

The collector loads Town03 itself, so the initial server map can be different.
The first collection startup also initializes the map renderer and can take
longer than subsequent episode resets. `--headless` disables the Flask
observation monitor, independently of the simulator's offscreen rendering.

For a larger run, choose a fresh directory:

```bash
.venv/bin/python -B -u generate_dataset.py --headless --n-traj 2000 --outdir ./dataset/run01
```

This starts a new run at episode zero; it does not resume an existing dataset.
Reusing a directory overwrites files and can leave old image files behind.
The `--seed` option seeds NumPy only; Python's `random` is not currently seeded.
Use `-B` to avoid regenerating the Python bytecode files tracked in this repo.

## Time of day and weather

Choose lighting and weather independently with `--time-of-day day|sunset|night`
and `--weather sunny|cloudy|rainy`. Defaults are `day` and `sunny`. Each run uses
fixed conditions, applied after the map loads and preserved across episode resets.
Use a fresh output directory for each combination; directory names are not added
automatically, and the saved tensor structure is unchanged.

```bash
.venv/bin/python -B -u generate_dataset.py --headless --time-of-day day --weather sunny --outdir ./dataset/day_sunny
.venv/bin/python -B -u generate_dataset.py --headless --time-of-day night --weather sunny --outdir ./dataset/night_sunny
.venv/bin/python -B -u generate_dataset.py --headless --time-of-day day --weather rainy --outdir ./dataset/day_rainy
.venv/bin/python -B -u generate_dataset.py --headless --time-of-day night --weather rainy --outdir ./dataset/night_rainy
```

Day, sunset, and night use sun altitudes of 45, 5, and -30 degrees, respectively.
Weather uses CARLA's ClearNoon, CloudyNoon, and MidRainyNoon presets with the sun
overridden independently. Rainy also sets road wetness to 80. These are visual
conditions; they do not change tire friction or vehicle physics ([CARLA API](https://carla.readthedocs.io/en/0.9.15/python_api/#carla.WeatherParameters)).
Night and rain images have not yet been visually validated with these flags.
Run these collections sequentially on a single server: starting another collector
reloads its map and interferes with an active run. Editing the script does not
change an already-running collection.

## Additional full-range controller

The original `compute_continuous_action()` is unchanged. Following collection
defaults to `--controller original`; select `--controller full-range` to use only
the new `FullRangeFollowingController` in `collection_controllers.py`.
Its feedback uses center-to-center following distance (target 8 m) and relative
speed, permitting braking when close or closing quickly. Every five recorded
steps, it selects uniform full-range exploration with probability 0.35; otherwise
it applies feedback with small held Gaussian noise. All commands are clipped to
acceleration `[-3, 3]` and steering `[-1, 1]`. These are exploration commands,
not measured physical acceleration. Coverage of particular state/action pairs
must still be assessed from the collected data.

To collect an additional **500 trajectories per configuration**, using only this
controller, across all nine conditions:

```bash
./collect_full_range.sh ./dataset/full_range_run01
```

This starts its own Epic servers, runs one configuration at a time by default,
and collects 4,500 new trajectories. The output root must be new or empty. It uses
fresh ports from 2400 through 2480, saves each worker under `batch_1/` through
`batch_9/`, and preserves the existing following dataset format. Existing datasets
are not merged or modified. An optional second argument overrides the count:

```bash
./collect_full_range.sh ./dataset/full_range_smoke 3
```

For the 300-per-configuration run:

```bash
./collect_full_range.sh ./dataset/full_range_300_retry 300
```

Single-server mode became the default after a November 12 map-loading crash in
Unreal's Vulkan texture eviction code while two Epic servers were running. The
stack is consistent with GPU memory pressure but does not prove an out-of-memory
error. To explicitly opt into two concurrent workers, prefix the command with
`WORKERS=2`; that mode uses five batches and has less GPU memory headroom.

For a single manually started server:

```bash
.venv/bin/python -B -u generate_dataset.py --headless --controller full-range \
  --n-traj 500 --outdir ./dataset/new_day_sunny
```

The parallel launcher also forwards `--controller full-range` to its following
workers. Lane-offset collection does not use this controller. Camera warm-up
still uses the existing unsaved zero-control steps; every recorded action in a
full-range run is produced by the new controller.

Verified on November 11, 2026: three live day/sunny and three night/rainy
episodes completed with only the full-range controller. The 842 saved actions
included 143 negative acceleration commands; all acceleration/steering commands
were within `[-3, 3]` / `[-1, 1]`. Numeric values, trajectory lengths/shapes,
and nonblank images passed validation. Feedback and seeded exploration tests
also passed. The original controller method was verified unchanged. Smoke
results are in `.local/full-range-controller-smoke/validation.json`; the full
500-per-configuration run has not been started.

## Collect two datasets in parallel

### Both collectors across all nine conditions

```bash
./collect_both.sh ./dataset/both_collectors_run01
```

This collects **500 trajectories per condition per collector**: 4,500 following
trajectories plus 4,500 lane-offset trajectories (9,000 total). One following
collector and one lane-offset collector run concurrently for a condition; then
the script advances to the next of the nine conditions. Both use Epic servers.
Lane-offset episodes contain 100 frames at 20 simulated Hz; following episodes
keep their existing termination-based lengths.

For a small sweep, override the count with the second argument:

```bash
./collect_both.sh ./dataset/both_collectors_smoke 1
```

Outputs are separated by condition and collector:

```text
both_collectors_run01/
  day_sunny/
    worker_0_day_sunny/     # generate_dataset.py: existing following format
    worker_1_day_sunny/     # collect_lane_offsets.py: episode dictionaries
    worker_0_day_sunny.log
    worker_1_day_sunny.log
    server_2000.log
    server_2010.log
  day_cloudy/
  day_rainy/
  sunset_sunny/
  sunset_cloudy/
  sunset_rainy/
  night_sunny/
  night_cloudy/
  night_rainy/
```

The root must be new or empty. Collection stops on a failed job; it does not
resume partial runs. Servers are started and stopped automatically. Distinct
port pairs are used per condition (2000/2010, 2020/2030, ... 2160/2170), avoiding
TCP teardown delays between batches. Set `BASE_PORT` to select another starting
port. The traffic-manager ports remain RPC port + 6000.

The parallel launcher also accepts `--first-collector following|lane-offsets`,
`--second-collector following|lane-offsets`, and `--lane-frames` (default 100).
Existing commands continue to default to two following collectors.

Following collection retries camera initialization up to ten resets, logging
termination reasons, before failing. Lane-offset collection unregisters each
vehicle from autopilot and advances the synchronous world before actor cleanup,
preventing the destroyed-actor shutdown abort found during its first live test.
Lane-offset collection also retries rejected episodes up to ten attempts by
default (`--max-attempts` when running `collect_lane_offsets.py` directly).
Black/missing camera frames, missing poses, invalid lane labels, and spawn
failures discard the attempted episode and retry with fresh actors after cleanup.
Only successfully collected episodes are saved and counted; completed files are
preserved if attempts are exhausted. Cleanup/server failures and programming
errors still stop the run rather than repeatedly retrying with uncertain state.
This adds retries within a running collection, not resume support after exit.

Verified on November 8, 2026: `collect_both.sh` completed one trajectory from each
collector in all nine configurations (18 total). Each lane-offset episode had
100 contiguous frame IDs, finite labels, no black frames, and zero consecutive
duplicate images. Following episode lengths were 58, 77, 62, 67, 60, 61, 53,
123, and 50 in condition order; saved tensor lengths/shapes, finite values, and
nonblank/changing images passed checks. Day/night lane camera samples were
visually inspected. An earlier isolated lane test also completed three 100-frame
episodes after the autopilot cleanup fix. This is smoke coverage, not a completed
500-trajectory run or a proof of following-camera frame synchronization.
Results are in `.local/both-collectors-smoke-v2/`, including `validation.json`;
the sweep log is `.local/logs/both-collectors-smoke-v2.log`.

To collect all nine day/sunset/night and sunny/cloudy/rainy combinations with
2,000 trajectories each (18,000 total), run:

```bash
./collect_all.sh ./dataset/all_conditions_run01
```

This runs four pairs followed by the ninth dataset alone, using Epic quality.
It starts/stops its own servers. Outputs are under `batch_1/` through `batch_5/`,
each containing the named worker dataset directories and logs. The output root
must be new or empty; omitting it creates a timestamped directory. The script stops on
the first failed batch and does not resume partial runs.

`collect_parallel.py` starts two independent Epic-quality servers, waits for
readiness, runs two collectors concurrently, and stops its servers when finished
or interrupted. It refuses an existing output directory and checks that the
required ports are available. Stop other collectors on those ports first.

```bash
VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json \
  .venv/bin/python -B -u collect_parallel.py --n-traj 2000 \
  --outdir ./dataset/parallel_run01 \
  --first-time-of-day night --first-weather sunny \
  --second-time-of-day day --second-weather rainy
```

`--n-traj` is the number of episodes **per dataset**; its default is 3 for a
small test. Outputs retain the existing format in separate subdirectories:
`worker_0_night_sunny/` and `worker_1_day_rainy/`. Separate server/collector logs
are written in the parent output directory. This does not add resume support.

The default RPC ports are 2000 and 2010, with streaming/secondary ports 2001/2002
and 2011/2012, and traffic-manager ports 8000 and 8010. `--base-port` changes the
first RPC port; the second is always ten higher. The individual collector also
accepts `--port` when connecting to a server you started yourself.

Server startup is staggered to reduce simultaneous loading pressure. Both use
Epic quality, a 20 FPS cap, and a 256 x 256 unused spectator viewport. The dataset
cameras remain 224 x 224. The cap may limit throughput; parallel speedup versus
an uncapped single server has not been benchmarked.

Verified on November 8, 2026: two concurrent collections completed three episodes
each. Night/sunny lengths were 77, 8, and 86; day/rainy lengths were 33, 34, and
48. All saved numeric data were finite, image/tensor lengths matched, and there
were no black placeholder frames. Saved image samples showed distinct night and
rain conditions. A live weather query confirmed sun altitude -30/rain 0 on the
first server and sun altitude 45/rain 60/wetness 80 on the second. Exact frame
synchronization remains unverified, and a few consecutive duplicate images remain.

Observed GPU usage was about 9.5 GB during collection, with startup samples up to
11.2 GB of 12.3 GB. These are sampled measurements, not a measured peak. Two
instances worked in this short test; three are not recommended on this GPU.
Results and logs are in `.local/parallel-epic-test-v3/`; both test servers were
stopped after completion.

## Install on a fresh machine

First check `nvidia-smi`. A working NVIDIA graphics/Vulkan driver is required.
Keep at least 30 GB free for the simulator, its compressed archive, and Python
dependencies, plus additional space for collected images. The verified machine
already had the required system graphics libraries installed.

Download the official [CARLA 0.9.15 Linux release](https://github.com/carla-simulator/carla/releases/tag/0.9.15):

```bash
mkdir -p .local/downloads .local/carla-0.9.15 .local/bin
wget -c -O .local/downloads/CARLA_0.9.15.tar.gz https://downloads.carlasim.com/Linux/CARLA_0.9.15.tar.gz
tar -xzf .local/downloads/CARLA_0.9.15.tar.gz -C .local/carla-0.9.15
```

The archive is about 7.8 GiB and extracts to about 19 GiB. Town03 is included;
the additional maps archive is not needed.

Create an isolated Python 3.10 environment. The system Python 3.12 cannot use
the CARLA 0.9.15 wheels. These commands install a portable uv binary locally:

```bash
wget -O .local/downloads/uv.tar.gz https://github.com/astral-sh/uv/releases/download/0.13.7/uv-x86_64-unknown-linux-gnu.tar.gz
tar -xzf .local/downloads/uv.tar.gz -C .local/bin --strip-components=1
UV_CACHE_DIR="$PWD/.local/uv-cache" UV_PYTHON_INSTALL_DIR="$PWD/.local/python" \
  .local/bin/uv venv --python 3.10 .venv
UV_CACHE_DIR="$PWD/.local/uv-cache" \
  .local/bin/uv pip install --python .venv/bin/python -r requirements-collection.txt
UV_CACHE_DIR="$PWD/.local/uv-cache" \
  .local/bin/uv pip check --python .venv/bin/python
```

The dependency file pins the matching CARLA client, Gym's existing API, NumPy
1.x, and CPU PyTorch. See CARLA's [package installation guide](https://carla.readthedocs.io/en/0.9.15/start_quickstart/)
for upstream installation details. The NVIDIA ICD path in the launch command
is the path verified on this machine; other distributions may use a different path.

## Saved files and verification

The output layout is unchanged:

```text
<outdir>/
  obses/episode_000.pth       # uint8 [T, 224, 224, 3], RGB
  obses/episode_001.pth
  ...
  actions.pth                # list of float32 [T, 2] tensors
  states.pth                 # list of float32 [T, 12] tensors
  costs.pth                  # list of float32 [T] tensors
  distances.pth              # list of float32 [T] tensors
  seq_lengths.pth            # int64 [number_of_episodes]
```

Each observation/state is captured before its corresponding action. Costs and
distances are measured after that action. Full state contains ego pose followed
by leader pose; each pose is `(x, y, z, pitch, yaw, roll)`. Rewards, terminal
flags, and the final next observation are not saved.

The collector copies camera arrays immediately to avoid buffer reuse changing
previously collected images. It takes up to 20 zero-control warm-up steps until
a nonzero camera image arrives before recording an episode. This excludes the
initial zero placeholders; it does not establish exact sensor frame alignment.

Files are checkpointed after each episode, using temporary files and atomic
replacement. The sequence lengths file is written last. Individual files are
protected against partial writes, but the group of files is not a transaction.

The final smoke run is in `.local/verified-dataset`; its log is
`.local/logs/collection-verified.log`. It completed three episodes of lengths
51, 62, and 32. Verification checked all file counts, tensor shapes, finite
numeric values, zero black frames, and changing images. Each episode had one
consecutive duplicate image pair. Train and validation samples also loaded
successfully through `load_carla_slice_train_val` with `frameskip=1` and ten
frames per slice.

Sensor handler code was not modified. Exact camera/state synchronization and
collision event timing still need separate validation before using the data for
training. This smoke test verifies installation and the collection/loading path.

## Troubleshooting

- If the Python client times out, check the server terminal, confirm port 2000,
  and check for a server crash. The traffic manager uses port 8000.
- If Town03 loading crashes with `FVulkanVertexInputStateInfo::Generate`, verify
  that the server was launched with `-quality-level=Epic`, not Low.
- If GPU checks fail only inside a restricted execution sandbox, repeat them
  from a normal terminal before changing the driver. GPU access worked outside
  the sandbox during this setup.
- The Gym maintenance warning and float32 Box precision warning appeared during
  successful runs. Keep the pinned dependencies rather than switching Gym APIs
  or upgrading NumPy to 2.x during setup.
- For detailed server diagnostics, append `-stdout -FullStdOutLogOutput` to the
  launch command and redirect output to a log file. The working server log from
  this setup is `.local/logs/carla-server-epic.log`.
