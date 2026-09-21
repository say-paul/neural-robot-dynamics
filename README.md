# Neural Robot Dynamics (NeRD)

> [!Note]
> **This branch is under development, and currently only contains a subset of examples.**

This branch demonstrates the integration of the NeRD into [Newton](https://github.com/newton-physics/newton) as a backend physics solver. 

[***Neural Robot Dynamics***](https://neural-robot-dynamics.github.io/) <br/>
[Jie Xu](https://people.csail.mit.edu/jiex), [Eric Heiden](https://eric-heiden.com/), [Iretiayo Akinola](https://research.nvidia.com/person/iretiayo-akinola), [Dieter Fox](https://homes.cs.washington.edu/~fox/), [Miles Macklin](https://blog.mmacklin.com/about/), [Yashraj Narang](https://research.nvidia.com/person/yashraj-narang) <br/>
***CoRL 2025***

## Installation
The code has been tested on Ubuntu 20.04 with Python 3.12.11, PyTorch 2.5.1, and CUDA 12.9.

Step 1: clone the repo
```
git clone git@github.com:NVlabs/neural-robot-dynamics.git
git checkout nerd_newton_dev
```

Step 2: Create an Anaconda virtual environment (recommended)
```
conda create -n nerd_newton python=3.12
conda activate nerd_newton
```

Step 3: Install [PyTorch 2.5.1](https://pytorch.org/get-started/previous-versions/#v251) (the tested version).
```
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu124
```

Step 4: Install Newton and Warp. Due to the rapid API evolution of [Newton](https://github.com/newton-physics/newton), the current preview version of code works with a previous version of Newton. Use the command below for Newton and Warp installation.
```
python -m pip install mujoco --pre -f https://py.mujoco.org/
python -m pip install warp-lang --pre -U -f https://pypi.nvidia.com/warp-lang/
python -m pip install git+https://github.com/google-deepmind/mujoco_warp.git@main
python -m pip install git+https://github.com/newton-physics/newton@668dfb
```

Step 5: Install rl-games, as well as other missing packages if encountered during running the scripts, e.g., `tqdm`, `pyglet`.
```
pip install rl-games tqdm pyglet
```

## Examples

We released pretrained NeRD models of Cartpole and Ant (pretrained from Warp sim) to demonstrate the integration of the neural dynamics solver into Newton. 

> [!Note]
> If you noticed the neural models under `pretrained_models` folders are not properly cloned, you may need to use `git lfs` for those files by running 
> ```
> git lfs install
> git lfs pull
> ```

### Passive Motion Example
The script [`examples/example_neural_solver_passive.py`](examples/example_neural_solver_passive.py) rollouts the pretrained NeRD model for passive motions.
```
cd examples
python example_neural_solver_passive.py --env-name Cartpole
python example_neural_solver_passive.py --env-name Ant
```

### RL Policy Example
The script [`examples/example_neural_solver_rl.py`](examples/example_neural_solver_rl.py) rollouts pretrained RL policies in Newton with NeRD solvers.
```
cd examples
python example_neural_solver_rl.py --playback ../pretrained_models/RL_policies/Cartpole/0/nn/CartpolePPO.pth --num-envs 1 --num-games 1 --render
python example_neural_solver_rl.py --playback ../pretrained_models/RL_policies/Ant/run/0/nn/AntPPO.pth --num-envs 1 --num-games 1 --render
```

The same script also allows to train RL policies within Newton using NeRD solvers:
```
python example_neural_solver_rl.py --rl-cfg ./rl_cfg/Cartpole/cartpole.yaml --exp-name Cartpole
python example_neural_solver_rl.py --rl-cfg ./rl_cfg/Ant/ant_run.yaml --exp-name Ant/run
```

### Robot Data And NeRD Training

The script [`examples/example_robot_nerd_train.py`](examples/example_robot_nerd_train.py) collects random ground-truth robot trajectories into HDF5 files for `train`, `validation`, and `test`. It trains a causal Transformer to predict relative state changes from contiguous 10-step `states_embedding + joint_f` windows, reloads its `.pt` checkpoint, and evaluates it on the untouched test trajectories. Saturated actuator commands and transitions that begin or end outside joint limits are marked invalid, and training windows never cross those invalid transitions. The checkpoint records the Transformer solver and history configuration required by the rollout example.

> [!Important]
> Older `transition`-mode HDF5 files and MLP checkpoints are not compatible with
> Transformer training or resume. Regenerate all selected dataset splits before
> starting a Transformer run.

```bash
PYTHONPATH="$PWD" .venv312/bin/python examples/example_robot_nerd_train.py \
  --robot-id so101 --num-envs 1 --horizon 1200 --seed 42 --default-pose \
  --render --render-backend rerun --rerun-view camera \
  --grpc-port 19878 --web-port 19092 --browser-host localhost \
  --diagnostics --action-scale 1.0 --keep-open
```

By default, the generated artifacts are `outputs/so101_train.hdf5`, `outputs/so101_validation.hdf5`, `outputs/so101_test.hdf5`, and `outputs/so101_nerd_model.pt`. To generate only test data for a separate NeRD training/evaluation run:

```bash
PYTHONPATH="$PWD" .venv312/bin/python examples/example_robot_nerd_train.py \
  --generate-only --splits test --horizon 1200 --test-dataset-path data/so101_test.hdf5
```

Use `--train-dataset-path`, `--validation-dataset-path`, and `--test-dataset-path` to place the files where a NeRD training configuration expects them.

To train from existing trajectory-mode HDF5 files without regenerating or
overwriting them, use `--skip-generation`. `--resume-checkpoint` continues a
compatible Transformer for the requested number of additional epochs. A resumed
checkpoint must have the same robot ID, state and actuator dimensions, input
contract, architecture, and 10-step history length.

```bash
PYTHONPATH="$PWD" .venv312/bin/python examples/example_robot_nerd_train.py \
  --skip-generation --splits train validation test --robot-id so101 \
  --train-dataset-path outputs/so101_train.hdf5 \
  --validation-dataset-path outputs/so101_validation.hdf5 \
  --test-dataset-path outputs/so101_test.hdf5 \
  --resume-checkpoint outputs/so101_nerd_model.pt \
  --checkpoint-path outputs/so101_nerd_model.pt \
  --epochs 1000 --batch-size 256 --learning-rate 1e-3
```

To simulate the robot with a trained NeRD model, pass the checkpoint to the
robot rollout example. This switches from Newton ground-truth dynamics to
`NeuralSolver`; `--random-actions` supplies random normalized control commands,
while its absence applies zero commands.

```bash
PYTHONPATH="$PWD" .venv312/bin/python examples/example_robot_rollout.py \
  --robot-id so101 --num-envs 1 --horizon 500 --default-pose \
  --nerd-checkpoint outputs/so101_nerd_model.pt \
  --random-actions --action-scale 0.5
```

### Sim-to-Real Roadmap

1. **Robot contract:** Verify the SO-101 model asset, joint ordering, limits,
  position gains, damping, action scale, and control frequency against the
  physical robot.
2. **Dynamics data:** Generate separate train, validation, and test HDF5 files
  from varied initial poses and safe action ranges. Measure one-step and
  multi-step NeRD rollout error on the untouched test split.
3. **Robust simulation:** Randomize mass, inertia, damping, friction, motor
  strength, control latency, sensor noise, payload, and contact properties at
  every RL episode.
4. **Policy training:** Train a bounded, smooth policy across that randomized
  simulation distribution. Test it against held-out parameter combinations,
  disturbances, and long rollouts.
5. **Safe hardware evaluation:** Deploy the unchanged policy at conservative
  speed and workspace limits with an emergency stop and independent limit
  checks. Record tracking and task metrics.
6. **Adaptation, if needed:** Use recorded real trajectories to calibrate the
  simulator or learn NeRD residual dynamics, then retrain or fine-tune. This
  is sim-to-real adaptation, rather than zero-shot transfer.

Zero-shot sim-to-real means the policy is trained only in simulation and is
deployed unchanged on the physical robot. The test result in simulation is not
evidence of zero-shot transfer; it is established only by a separately held-out
real-robot evaluation.

## Citation

If you find our paper or code useful, please consider citing:
```
@inproceedings{
  xu2025neural,
  title={Neural Robot Dynamics},
  author={Jie Xu and Eric Heiden and Iretiayo Akinola and Dieter Fox and Miles Macklin and Yashraj Narang},
  booktitle={9th Annual Conference on Robot Learning},
  year={2025}
}
```