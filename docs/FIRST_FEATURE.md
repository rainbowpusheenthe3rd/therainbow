# First feature — DDPM on 2-D toy data (pure numpy)

A complete, tiny diffusion model with **no framework** — so the mechanism is fully visible and
runs in seconds on a CPU. This is Phase 1 of the [roadmap](../ROADMAP.md).

## Goal
Learn to sample from a 2-D distribution (start with **two moons** / a spiral) by denoising, and
*show* the process: a point cloud of static resolving into the target shape.

## Scope
- **Data:** a 2-D toy dataset (`sklearn.datasets.make_moons`, or a hand-rolled spiral). No images
  yet — 2-D keeps every tensor plottable.
- **Forward process:** a fixed **β-schedule** (linear or cosine) with the closed-form
  `q(x_t | x_0)` so any noise level is one step to compute. Verify empirically that `x_T` is ~N(0, I).
- **Model:** a small **MLP** `εθ(x_t, t)` predicting the noise (time `t` embedded via sinusoidal
  features). Pure numpy forward + backward, or `numpy` with hand-written gradients for a 2–3 layer net.
- **Training:** the standard simple objective — MSE between true ε and `εθ(x_t, t)` over random `t`.
- **Sampling:** ancestral DDPM sampling `x_T → x_0`; capture frames to animate the reverse process.

## Deliverables
- `src/therainbow/ddpm2d.py` — schedule, `q_sample`, the MLP, train loop, sampler (readable, commented).
- `scripts/demo_ddpm2d.py` — trains, samples, and writes:
  - `assets/forward_noising.png` (data → static across `t`),
  - `assets/reverse_sampling.gif|png` (static → recovered shape).
- `tests/test_ddpm2d.py`:
  - forward process drives variance toward 1 / mean toward 0 at large `t`;
  - after a short train, sampled points' mean/covariance are close to the data's (a loose,
    seed-fixed tolerance — correctness, not benchmark).

## Success = you can explain, from your own code
- why `q(x_t|x_0)` has that closed form,
- what the model actually predicts and why ε-prediction is convenient,
- how sampling walks noise back to data one step at a time.

## Not in this feature
UNets, images, guidance, Rust — those are Phases 2–4. Keep this one small and legible.
