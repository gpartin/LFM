# Benchmarks (workspace/performance/benchmarks)

Purpose
- Establish baseline performance characteristics and bottlenecks for key workloads.
- Provide stable references against which optimizations are measured.

Typical contents
- `{test_name}_baseline.py` — minimal reference implementation for timing.
- `{test_name}_profile.py` — profiling harness (CPU/GPU, memory, I/O).
- `{test_name}_report.md` — findings with line-level hotspots and metrics.

How this folder feeds other documentation and processes
- Optimizations depend on these baselines for validation and speedup measurement.
- Performance results may be summarized in reports and uploads when relevant.
- Links from `performance/README.md` describe the end-to-end workflow and tooling.

Standards
- Multiple runs with mean ± std; include hardware and environment details.
- Keep baselines simple and faithful to validated physics; do not mix code changes into baselines.

Related docs
- Performance overview: `performance/README.md`
- Optimizations: `performance/optimizations/`
- Tests (optional performance checks): `tests/performance/`
