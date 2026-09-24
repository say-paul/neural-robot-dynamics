Migration status from main:

- [x] Scheduler and TensorBoard
- [x] Resume-safe checkpoints
- [x] Dataset/schema compatibility validation
- [x] DataLoader worker pipeline: optional worker-prefetched trajectory windows
	are enabled with `optimization.num_workers`; the device-resident cache
	remains the default fast path.
- [x] Generic rollout evaluation
- [ ] Hybrid physics projection
- [x] Contact-rich dataset generation
- [~] Full training workflow: generation, logging, periodic checkpoints,
	resume, validation, test evaluation, and optional rollout metrics are now
	wired; smoke-test gating and projection remain future work.