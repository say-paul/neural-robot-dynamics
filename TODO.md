Migration status from main:

- [x] Scheduler and TensorBoard
- [x] Resume-safe checkpoints
- [x] Dataset/schema compatibility validation
- [~] DataLoader worker pipeline: replaced the old full-array copy path with a
	device-resident trajectory cache; worker-backed loading remains optional
	future work for datasets that do not fit on the selected device.
- [x] Generic rollout evaluation
- [ ] Hybrid physics projection
- [x] Contact-rich dataset generation
- [~] Full training workflow: generation, logging, periodic checkpoints,
	resume, validation, test evaluation, and optional rollout metrics are now
	wired; smoke-test gating and projection remain future work.