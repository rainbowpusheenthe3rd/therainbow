# therainbow roadmap

Build diffusion from first principles, in ascending complexity — each phase runnable and
understandable on its own before the next. Python for clarity; Rust where speed matters.

## Phase 1 — DDPM on toy data (pure numpy) 📍 first
Understand the *process* with no deep-learning framework in the way.
- Forward process: add Gaussian noise on a fixed schedule, `x_0 → x_T`.
- Reverse process: a tiny learned denoiser (even linear/MLP) on 2-D toy distributions (e.g. a
  spiral, two moons) so training runs in seconds on a CPU.
- Deliverable: a noising→denoising animation + a sampled point cloud that matches the data.
- Spec: [`docs/FIRST_FEATURE.md`](docs/FIRST_FEATURE.md).

## Phase 2 — Score / ε-prediction with a real denoiser (PyTorch)
- Swap the toy denoiser for a small **UNet** (images) / transformer (tokens).
- ε-prediction vs score-matching vs v-prediction — implement one, explain the others.
- Train on a small image set (MNIST/CIFAR-scale) that fits on a laptop GPU.

## Phase 3 — Samplers & guidance
- DDPM → **DDIM** → a few-step sampler; make the steps↔quality trade-off visible.
- **Classifier-free guidance** implemented and ablated (why it sharpens, what it costs).

## Phase 4 — Rust for the hot loops
- Reimplement the sampling loop + noise schedule in **Rust** (PyO3 bindings); benchmark vs numpy.
- Then borrow the **applied** patterns from the LTX-2 optimisation work — fp8 quantisation, LoRA
  adaptation, VRAM-aware batching — to bridge from "toy, correct" to "real, fast".

## Non-goals (for now)
- Beating production models. The point is comprehension; performance comes from the Rust phase and
  the applied borrow, not from re-training a foundation model.
