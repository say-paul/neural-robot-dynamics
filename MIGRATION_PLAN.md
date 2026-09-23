# Main To SO-101 Training Migration Plan

## Goal

Port the reusable sequence-learning capabilities from `main` into the Newton
robot branch while keeping robot-specific physical data and training choices in
configuration files. SO-101 is the first migration profile, not a special code
path.

## Ownership Boundaries

### Shared Framework

- Configuration loading and validation
- Windowed trajectory datasets
- Input and target normalization
- Transformer training, validation, evaluation, checkpointing, and logging
- Generic contact feature schema and backend adapters
- Optional hybrid physics projection after a neural prediction

### Robot Specifications

`robot_specs/` remains the source of physical facts:

- Asset source, joint layout, limits, actuators, and default pose
- Collision enablement, geometry groups, and excluded link pairs
- Solver settings required by the physical backend

### Training Profiles

`configs/training/` contains learning choices:

- Model architecture and selected input features
- Contact representation and slot count
- Dataset sampling policy and loss weights
- Optimizer, schedule, checkpoint, and evaluation settings
- Hybrid projection mode

Suggested layout:

```text
configs/
  training/
    base_transformer.yaml
    robots/
      so101.yaml
      franka_panda.yaml
```

## Migration Phases

1. Capture the `main` trainer contracts.
   Document the dataset schema, normalization, sequence window semantics,
   checkpoint metadata, validation, and evaluation behavior to preserve.

2. Define generic configuration.
   Add base and robot-overlay training configs. Validate configuration before
   creating an environment or allocating a dataset.

3. Port reusable sequence infrastructure.
   Adapt the `main` sequence trainer, trajectory dataset, optimizer schedule,
   gradient clipping, TensorBoard logging, checkpointing, and evaluation to the
   Newton environment interface.

4. Add a generic contact feature interface.
   Define a versioned fixed-width schema with padding and deterministic slot
   ordering. Implement backend adapters for MuJoCo-Warp and the legacy Warp
   contact source without exposing backend-specific objects to the trainer.

5. Configure SO-101.
   Add collision metadata to its robot specification and a training profile for
   state, action, and self-contact inputs. Do not add SO-101 identifiers, joint
   counts, or geometry names to shared Python modules.

6. Add hybrid physical projection.
   Support an optional rollout mode that performs neural prediction, collision
   detection, a short physics projection, and then records the resolved state
   in the sequence history.

7. Validate progressively.
   Run configuration and schema tests, tiny dataset generation, train/validation
   and test smoke runs, checkpoint reload, neural rollout, and hybrid Rerun
   rollout before launching full generation or training.

## Acceptance Criteria

- A new robot requires only a robot specification and training overlay.
- Dataset and checkpoint metadata reject incompatible feature schemas.
- Metrics distinguish free-space, contact, and penetration transitions.
- Hybrid rollout uses the physical backend to resolve collision constraints.
- The full workflow starts only after all smoke gates pass.