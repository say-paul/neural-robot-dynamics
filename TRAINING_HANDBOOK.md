# Neural Robot Dynamics Training Handbook

This is the command-line workflow for generating robot trajectory data, training the current Transformer NeRD model, monitoring it with TensorBoard, and resuming from a checkpoint. It does not use `scripts/train_so101_transformer.sh`.

The commands below assume the repository root is the current directory and that the project virtual environment is `.venv312`.

## 1. Prerequisites

```bash
cd /home/core/neural-robot-dynamics
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
PYTHON="$PWD/.venv312/bin/python"
```

Use an environment with PyTorch, CUDA, Warp/Newton, h5py, PyYAML, and TensorBoard installed. Confirm the interpreter and GPU before a long run:

```bash
"$PYTHON" -c "import torch; print(torch.__version__); print('CUDA:', torch.cuda.is_available())"
"$PYTHON" examples/example_robot_nerd_train.py --help
"$PYTHON" -m training.train --help
```

## 2. Recommended Workflow

The current Transformer workflow has four independent stages:

1. Generate `train.hdf5`.
2. Generate `validation.hdf5` and `test.hdf5` with separate seeds.
3. Train with `training.train`.
4. Resume with `--resume` and a larger final `--steps` value.

The generated files are trajectory-mode HDF5 files. They contain states, actions, joint forces, next states, validity masks, and contact features when the selected robot config enables contacts.

### 2.1 Choose paths and configuration

```bash
ROBOT_ID=so101
CONFIG=so101
OUT="$PWD/outputs/so101_transformer_contact_62500_1m"
TB="$PWD/outputs/tensorboard/so101_transformer_contact_62500_1m"
mkdir -p "$OUT" "$TB"
```

`CONFIG=so101` resolves to `configs/training/robots/so101.yaml`. A YAML path can be used instead, for example:

```bash
CONFIG=configs/training/robots/so101.yaml
```

The `so101` profile extends `configs/training/base_transformer.yaml`, enables contact features, uses a 10-step sequence, and uses `states_embedding`, `joint_f`, and `self_contact` as model inputs.

### 2.2 Generate the training split

The current large-data settings use 50,000 training trajectories, horizon 100, 128 parallel environments, action scale `0.01`, and action hold `20`.

```bash
"$PYTHON" examples/example_robot_nerd_train.py \
  --generate-only \
  --robot-id "$ROBOT_ID" \
  --training-config "$CONFIG" \
  --splits train \
  --num-envs 128 \
  --num-trajectories 50000 \
  --horizon 100 \
  --seed 42 \
  --action-scale 0.01 \
  --action-hold-steps 20 \
  --generation-print-interval 10000 \
  --train-dataset-path "$OUT/train.hdf5"
```

### 2.3 Generate the validation split

Use a different seed from training. The validation set is used during training for periodic monitoring and must not be used to update model weights.

```bash
"$PYTHON" examples/example_robot_nerd_train.py \
  --generate-only \
  --robot-id "$ROBOT_ID" \
  --training-config "$CONFIG" \
  --splits validation \
  --num-envs 128 \
  --num-trajectories 6250 \
  --horizon 100 \
  --seed 43 \
  --action-scale 0.01 \
  --action-hold-steps 20 \
  --generation-print-interval 1000 \
  --validation-dataset-path "$OUT/validation.hdf5"
```

### 2.4 Generate the test split

The test set is held out until final evaluation. Do not use it for checkpoint selection or tuning.

```bash
"$PYTHON" examples/example_robot_nerd_train.py \
  --generate-only \
  --robot-id "$ROBOT_ID" \
  --training-config "$CONFIG" \
  --splits test \
  --num-envs 128 \
  --num-trajectories 6250 \
  --horizon 100 \
  --seed 44 \
  --action-scale 0.01 \
  --action-hold-steps 20 \
  --generation-print-interval 1000 \
  --test-dataset-path "$OUT/test.hdf5"
```

`--generate-only` writes the selected split and exits. Existing files are overwritten because generation opens the target HDF5 file in write mode. Use a new output path when you need to preserve an earlier dataset.

### 2.5 Train from the generated files

This is the current config-driven Transformer trainer. `--steps` is the final target optimizer step, not the number of steps added to an existing checkpoint.

```bash
"$PYTHON" -m training.train \
  --config "$CONFIG" \
  --train-dataset "$OUT/train.hdf5" \
  --validation-dataset "$OUT/validation.hdf5" \
  --test-dataset "$OUT/test.hdf5" \
  --checkpoint "$OUT/model.pt" \
  --steps 10000 \
  --batch-size 512 \
  --learning-rate 1e-3 \
  --seed 42 \
  --log-dir "$TB" \
  --validation-interval 2000 \
  --validation-windows 4096 \
  --checkpoint-interval 2000 \
  --print-interval 2000 \
  --rollout-horizon 0
```

For a long run, detach it from the terminal:

```bash
nohup "$PYTHON" -m training.train \
  --config "$CONFIG" \
  --train-dataset "$OUT/train.hdf5" \
  --validation-dataset "$OUT/validation.hdf5" \
  --test-dataset "$OUT/test.hdf5" \
  --checkpoint "$OUT/model.pt" \
  --steps 10000 \
  --batch-size 512 \
  --learning-rate 1e-3 \
  --seed 42 \
  --log-dir "$TB" \
  --validation-interval 2000 \
  --validation-windows 4096 \
  --checkpoint-interval 2000 \
  --print-interval 2000 \
  --rollout-horizon 0 \
  > "$OUT/train.log" 2>&1 < /dev/null &
```

Monitor the text log:

```bash
tail -f "$OUT/train.log"
```

Pressing `Ctrl+C` while running `tail -f` stops only the log viewer, not a `nohup` training process.

### 2.6 Resume for another 50,000 steps

A checkpoint completed at step 10,000 must be resumed with `--steps 60000` to perform 50,000 additional steps:

```bash
RESUME_OUT="$PWD/outputs/so101_transformer_contact_62500_resume_50k"
RESUME_TB="$PWD/outputs/tensorboard/so101_transformer_contact_62500_resume_50k"
mkdir -p "$RESUME_OUT" "$RESUME_TB"

nohup "$PYTHON" -m training.train \
  --config "$CONFIG" \
  --train-dataset "$OUT/train.hdf5" \
  --validation-dataset "$OUT/validation.hdf5" \
  --test-dataset "$OUT/test.hdf5" \
  --checkpoint "$RESUME_OUT/model.pt" \
  --resume "$OUT/model.latest.pt" \
  --steps 60000 \
  --batch-size 512 \
  --learning-rate 1e-3 \
  --seed 42 \
  --log-dir "$RESUME_TB" \
  --validation-interval 2000 \
  --validation-windows 4096 \
  --checkpoint-interval 2000 \
  --print-interval 2000 \
  --rollout-horizon 0 \
  > "$RESUME_OUT/train.log" 2>&1 < /dev/null &
```

This preserves the original run and writes resumed checkpoints and TensorBoard events separately. The new run continues using global optimizer step numbers, so its first logged step is approximately `10001`.

If the old checkpoint completed at a different step, use:

```text
new final step = old completed step + desired additional steps
```

To resume for 50,000 total steps rather than 50,000 additional steps, use `--steps 50000` instead.

### 2.7 Checkpoint files

For `--checkpoint path/model.pt`, the trainer creates:

- `path/model.latest.pt`: rolling checkpoint written at every checkpoint interval.
- `path/model.pt`: final checkpoint written after training and final evaluation.

The rolling file is overwritten at each checkpoint interval. If every historical checkpoint is required, copy it to a versioned filename after each checkpoint or add checkpoint archival logic.

A resume checkpoint must match the dataset dimensions and the relevant training configuration: schema, inputs, sequence length, network, contact configuration, and projection configuration. Do not resume a Transformer run from an older transition-mode or MLP checkpoint.

### 2.8 TensorBoard

Start TensorBoard against one run directory:

```bash
nohup "$PWD/.venv312/bin/tensorboard" \
  --logdir "$TB" \
  --host 0.0.0.0 \
  --port 6006 \
  > "$TB/tensorboard.log" 2>&1 < /dev/null &
```

Open `http://localhost:6006` after forwarding port `6006` in VS Code's **Ports** panel. Use a different port if `6006` is already occupied.

To compare the original and resumed runs, point TensorBoard at their common parent directory:

```bash
nohup "$PWD/.venv312/bin/tensorboard" \
  --logdir "$PWD/outputs/tensorboard" \
  --host 0.0.0.0 \
  --port 6006 \
  > "$PWD/outputs/tensorboard/tensorboard.log" 2>&1 < /dev/null &
```

## 3. Data Generation CLI Reference

The generator is `examples/example_robot_nerd_train.py`.

| Option | Default | Meaning |
|---|---:|---|
| `--robot-id ID` | `so101` | Robot profile ID, such as `so101` or `franka_panda`. |
| `--training-config PATH_OR_ID` | robot ID | Training YAML path or profile name. |
| `--splits SPLIT ...` | `train validation test` | One or more of `train`, `validation`, `test`. `--num-trajectories` applies to each selected split. |
| `--generate-only` | off | Generate selected HDF5 files and skip model training. |
| `--skip-generation` | off | Use existing HDF5 files for the older integrated training path. It cannot be combined with `--generate-only`. |
| `--num-envs N` | `8` | Number of parallel simulation environments. More environments can improve throughput but require more GPU memory. |
| `--num-trajectories N` | automatic | Number of trajectories for each selected split. Must be positive when supplied. |
| `--generation-print-interval N` | `1000` | Print collection progress every N trajectories. |
| `--horizon N` | `200` | Number of transitions per trajectory. Must be long enough to contain valid 10-step windows. |
| `--seed N` | `42` | Base random seed. When multiple splits are generated in one invocation, split index is added to this seed. |
| `--default-pose` | off | Start environments from the robot's configured default pose. |
| `--action-scale VALUE` | `1.0` | Scale applied to generated random actions. Use the value expected by the training configuration. |
| `--action-hold-steps N` | `20` | Number of simulation steps for which each random-walk target is held. Must be positive. |
| `--train-dataset-path PATH` | `outputs/<robot>_train.hdf5` | Output/input path for the train split. |
| `--validation-dataset-path PATH` | `outputs/<robot>_validation.hdf5` | Output/input path for validation. |
| `--test-dataset-path PATH` | `outputs/<robot>_test.hdf5` | Output/input path for test. |
| `--dataset-path PATH` | none | Deprecated alias for `--train-dataset-path`. |
| `--render` | off | Enable OpenGL or Rerun rendering during generation. Rendering slows collection. |
| `--render-backend {opengl,rerun}` | `opengl` | Rendering backend. |
| `--render-splits SPLIT ...` | `train` | Splits to render when rendering is enabled. |
| `--rerun-view {3d,camera}` | `camera` | Rerun view. |
| `--grpc-port N` | `9876` | Rerun gRPC port. |
| `--web-port N` | `9090` | Rerun web port. |
| `--browser-host HOST` | `localhost` | Hostname used by the Rerun browser link. |
| `--env-spacing VALUE` | `0.0` | X/Z spacing between parallel robots for visualization. Must not be negative. |
| `--diagnostics` | off | Enable robot diagnostics during rendered generation. |
| `--keep-open` | off | Keep a rendered viewer alive after generation. Requires `--render`; exactly one rendered split is required. |

The following generator options belong to the older integrated training path. They are not needed when using the current `python -m training.train` command:

| Option | Default | Meaning |
|---|---:|---|
| `--epochs N` | `200` | Number of additional epochs for the integrated trainer. |
| `--target-epochs N` | none | Total epoch/update target when resuming the integrated trainer. |
| `--batch-size N` | `128` | Integrated trainer batch size. |
| `--learning-rate VALUE` | `1e-3` | Integrated trainer learning rate. |
| `--log-dir PATH` | none | Integrated trainer TensorBoard directory; omitted means disabled. |
| `--validation-interval N` | `10` | Integrated trainer validation interval. |
| `--validation-windows N` | `4096` | Maximum validation windows used for monitoring. |
| `--print-interval N` | `10` | Integrated trainer print interval. |
| `--checkpoint-interval N` | `1000` | Integrated trainer checkpoint interval. |
| `--resume-checkpoint PATH` | none | Resume a compatible integrated-trainer checkpoint. |
| `--checkpoint-path PATH` | `outputs/<robot>_nerd_model.pt` | Integrated-trainer checkpoint output path. |

## 4. Current Transformer Training CLI Reference

The current trainer is `training/train.py`, invoked as `python -m training.train`.

| Option | Default | Meaning |
|---|---:|---|
| `--config PATH_OR_ID` | required | Training YAML path or profile name. |
| `--train-dataset PATH` | required | Trajectory-mode training HDF5. |
| `--validation-dataset PATH` | required | Trajectory-mode validation HDF5. |
| `--test-dataset PATH` | none | Optional test HDF5 evaluated after training. |
| `--checkpoint PATH` | required | Final checkpoint path. The rolling path is derived by replacing `.pt` with `.latest.pt`. |
| `--steps N` | `1000` | Final optimizer step. On resume, this must be greater than or equal to the checkpoint's completed step. |
| `--seed N` | `42` | Sampling random seed. |
| `--batch-size N` | config value | Override `optimization.batch_size`. |
| `--learning-rate VALUE` | config value | Override `optimization.learning_rate`. |
| `--resume PATH` | none | Load model, optimizer, scheduler, normalization statistics, and completed step from a compatible checkpoint. |
| `--log-dir PATH` | none | TensorBoard event directory. Omit to disable TensorBoard logging. |
| `--validation-interval N` | `5000` | Run periodic validation every N optimizer steps. |
| `--validation-windows N` | all | Maximum validation windows used by periodic validation. |
| `--checkpoint-interval N` | `10000` | Save the rolling checkpoint every N optimizer steps. |
| `--print-interval N` | `1000` | Print training progress every N optimizer steps. Validation steps are printed too. |
| `--rollout-horizon N` | `0` | Optional autoregressive rollout length after training. `0` disables rollout evaluation. |
| `--rollout-windows N` | config/all | Maximum rollout windows when rollout evaluation is enabled. |

## 5. Training Configuration Reference

`--config` loads a YAML file and supports `extends`. The current Transformer schema contains these sections:

| YAML field | Purpose |
|---|---|
| `schema_version` | Configuration schema version. Current value is `1`. |
| `robot_id` | Robot identity used for compatibility checks. |
| `inputs.low_dim` | Model input names. Current SO-101 profile uses `states_embedding`, `joint_f`, and `self_contact`. |
| `sequence.length` | Number of consecutive transitions in each training window. Current value is `10`. |
| `network.encoder.low_dim.activation` | Activation for low-dimensional input encoders. |
| `network.encoder.low_dim.layer_sizes` | Hidden sizes for low-dimensional encoders. |
| `network.encoder.low_dim.layernorm` | Whether encoder outputs use layer normalization. |
| `network.model.mlp.activation` | Activation in the model MLP. |
| `network.model.mlp.layer_sizes` | Model MLP hidden sizes. |
| `network.model.mlp.layernorm` | Whether the model MLP uses layer normalization. |
| `network.transformer.n_layer` | Transformer block count. |
| `network.transformer.n_head` | Attention head count. |
| `network.transformer.n_embd` | Transformer embedding width. |
| `network.transformer.block_size` | Maximum sequence length supported by the Transformer. It must cover `sequence.length`. |
| `network.transformer.bias` | Whether Transformer linear layers use bias. |
| `network.transformer.dropout` | Transformer dropout probability. |
| `network.normalize_input` | Enable input normalization. |
| `network.normalize_output` | Enable target/output normalization. |
| `network.output_tanh` | Apply a tanh output bound. |
| `optimization.learning_rate` | Initial learning rate. |
| `optimization.learning_rate_end` | Final learning rate for a scheduled run. |
| `optimization.lr_schedule` | `constant`, `linear`, or `cosine`. |
| `optimization.batch_size` | Default batch size. |
| `optimization.num_workers` | DataLoader worker count. `0` uses direct GPU indexing. |
| `optimization.gradient_norm` | Optional gradient clipping limit. |
| `dataset.action_scale` | Dataset-generation action scale documented by the robot profile. |
| `dataset.action_hold_steps` | Dataset-generation action hold duration. |
| `contact.enabled` | Enable contact features and contact-aware generation. |
| `contact.backend` | Contact backend, such as `mujoco_warp` or `none`. |
| `contact.max_contacts` | Maximum contacts represented per step. |
| `contact.features` | Contact feature names, such as `active`, `separation`, `normal`, and `position`. |
| `contact.reject_separation_below` | Reject transitions with contacts below this separation threshold. |
| `contact.reject_nonfinite` | Reject non-finite contact values. |
| `projection.enabled` | Enable dynamics projection/correction. |
| `projection.solver_steps` | Number of projection solver steps. |

Changing any of `schema_version`, `inputs`, `sequence`, `network`, `contact`, or `projection` makes an existing current-format checkpoint incompatible by design.

## 6. Useful Checks

Check that the files exist and are non-empty:

```bash
ls -lh "$OUT"/*.hdf5 "$OUT"/*.pt
```

Check that a run is active:

```bash
ps -eo pid,ppid,stat,etime,%cpu,%mem,cmd \
  | grep -E '[t]raining.train|[t]ensorboard'
```

Check the latest training output without attaching to the process:

```bash
tail -n 50 "$OUT/train.log"
```

A successful current Transformer run ends with `train_loss=...`, `validation_mse=...`, and optionally `test_mse=...`. A successful resume log also contains `Resuming from checkpoint: ...` before the first training step.

## 7. Common Mistakes

- **Using `--steps 50000` for 50,000 additional steps:** if the checkpoint is at 10,000, use `--steps 60000`.
- **Resuming from `model.pt` while training is still running:** use the most recent `model.latest.pt` checkpoint.
- **Writing a new run into the old TensorBoard directory:** choose a new `--log-dir` for a separate run.
- **Changing the config during resume:** keep the same input list, sequence length, network, contact settings, and projection settings.
- **Mixing data-generation settings between splits:** use the same robot, config, horizon, action scale, and action hold settings; change only the split, trajectory count, and seed as intended.
- **Regenerating data accidentally:** `--generate-only` overwrites its target file. Use distinct paths for experiments you need to preserve.
- **Expecting `Ctrl+C` on `tail -f` to stop `nohup` training:** it stops only `tail`; inspect the training PID separately.
- **Confusing checkpoint formats:** checkpoints from the older integrated example are not interchangeable with checkpoints from `training.train`.
