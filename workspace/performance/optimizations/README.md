# Optimizations (workspace/performance/optimizations)

Purpose
- Host candidate performance improvements with clear validation and benchmarking.

Directory pattern
```
optimizations/
  {name}/
    implementation.py   # optimized code
    validation.py       # asserts numerical equivalence vs baseline
    benchmark.py        # reports speedup distributions
    report.md           # methodology, results, hardware, trade-offs
```

How this folder feeds other documentation and processes
- Each optimization cites its baseline in `performance/benchmarks/` and must pass validation first.
- Stable, validated optimizations may be referenced by physics tests once correctness is proven.
- Summaries can be included in public releases (`uploads/`) when performance is a reported result.

Standards
- “Validate first, then benchmark.”
- Report mean ± std over multiple runs; document hardware and environment.
- Keep trade-offs explicit (memory, precision, portability).

Related docs
- Performance overview: `performance/README.md`
- Benchmarks: `performance/benchmarks/`
- Project summary: `PERFORMANCE_OPTIMIZATIONS_README.md`
