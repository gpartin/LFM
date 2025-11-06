# Wave Algorithm Results (2025-11-05)

Scenario: 128^3 grid, 300 steps, Gaussian wave packet (σ=20, A=0.01, λ≈32), GPU backend.

Baseline (full update):
- Time: 2.193 s ± 0.089 s
- Drift: 1.94e-04 (Gate 0 PASS)
- Active fraction: 91.4%

Algorithm 1 — Active Field Mask (|E| threshold + halo):
- Time: 3.152 s ± 0.052 s
- Drift: 1.94e-04 (PASS)
- Active: 94.9%
- Speedup vs baseline: 0.76x (slower)

Algorithm 2 — Gradient-Adaptive (|∇E| + |E| + halo):
- Time: 4.016 s ± 0.073 s
- Drift: 1.94e-04 (PASS)
- Active: 98.3%
- Speedup vs baseline: 0.55x (slower)

Algorithm 3 — Wavefront Prediction (|E| or |∂E/∂t|; expand ~c·dt; persistent mask):
- Time: 3.249 s ± 0.217 s
- Drift: 1.94e-04 (PASS)
- Active: 92.3%
- Speedup vs baseline: 0.68x (slower)

Interpretation:
- The packet is dense (mean ~91% active), leaving little headroom for masking.
- Overheads (mask building, dilation, conditional updates) outweigh savings.
- Algorithm 2 inflates the mask the most (≈98%), causing the largest slowdown.

Recommendations:
- Create sparsity: larger grid (256^3) with same packet (σ=20) or narrower packet (σ=8–12) if drift stays < 5e-4.
- Parameter sweeps:
  - Alg1: threshold ∈ {2e-6, 5e-6}, halo ∈ {1,2}
  - Alg2: grad_thresh × {2,5}, halo ∈ {1,2}
  - Alg3: safety ∈ {1.0, 1.5}, persistence ∈ {False, True}
- Explore tile/block masks (e.g., 8^3 or 16^3 tiles) to reduce overhead.

Artifacts:
- Summaries in `results/`: baseline_wave_summary.json, algorithm1_summary.json, algorithm2_summary.json, algorithm3_summary.json
- Comparison: `results/comparison_wave_algorithms.md`
