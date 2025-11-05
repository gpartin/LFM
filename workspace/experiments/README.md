# LFM Experiments Directory

This directory contains exploratory physics investigations using the Lattice Field Model. Experiments follow a **three-gates promotion pipeline** from initial discovery to validated scientific claims.

## Philosophy: "Test and See"

The LFM framework enables rapid exploration of emergent physics phenomena. This directory structure supports:
- **Rapid iteration** on new physics ideas
- **Systematic validation** of promising results
- **IP protection** through documented discovery timeline
- **Scientific rigor** through reproducibility gates

## Directory Structure

```
experiments/
├── README.md              # This file
├── {investigation_name}/  # Active exploratory investigations
│   ├── script.py         # Experiment code
│   ├── notes.md          # Research notes, observations
│   ├── results/          # Raw outputs
│   └── analysis/         # Analysis scripts and plots
├── candidates/           # Experiments promoted to Gate 1
│   └── {name}/           # Same structure as above
└── archive/              # Completed or abandoned work
    └── {name}/           # Historical record
```

## Three-Gates Promotion Model

### Gate 0: Experiments (exploratory)
**Location**: `experiments/{investigation_name}/`

**Purpose**: Rapid exploration and discovery

**Criteria**:
- Executable script documenting the investigation
- Research notes explaining motivation and observations
- Raw results showing the phenomenon

**Activities**:
- Try new parameter regimes
- Explore unexpected behaviors
- Generate hypotheses
- Document surprises and anomalies

**Promotion Tool**: `python tools/promote_experiment.py {investigation_name}`

---

### Gate 1: Candidates (reproducible)
**Location**: `experiments/candidates/{investigation_name}/`

**Purpose**: Confirm reproducibility before investing validation effort

**Promotion Checklist**:
- [ ] Script runs without manual intervention
- [ ] Results reproduce within acceptable variance (< 5% typically)
- [ ] Dependencies documented (requirements.txt or environment.yml)
- [ ] Random seeds controlled or statistical variance quantified
- [ ] Output files generated consistently
- [ ] Computation time documented (for performance tracking)

**Activities**:
- Run on different machines/GPUs
- Test with different random seeds
- Verify numerical stability
- Document computational requirements

**Promotion Tool**: `python tools/promote_candidate.py {investigation_name}`

---

### Gate 2: Validated Tests (scientifically validated)
**Location**: `tests/tier{N}/{domain}/test_{name}.py`

**Purpose**: Formal validation as scientific claim

**Promotion Checklist**:
- [ ] Theoretical basis documented (equations, derivations)
- [ ] Parameter choices justified (not cherry-picked)
- [ ] Alternative explanations considered and ruled out
- [ ] Error analysis performed (numerical vs physical effects)
- [ ] Convergence tests completed (dx, dt independence)
- [ ] Boundary condition sensitivity checked
- [ ] Results compared to known limits or analytic solutions
- [ ] pytest-compatible test with assertions
- [ ] Docstrings explaining physics and validation criteria
- [ ] Discovery timestamped in `docs/discoveries/discoveries.json`

**Activities**:
- Write formal test with pytest
- Document theoretical foundation
- Perform sensitivity analysis
- Compare to literature/known results
- Submit for internal review

**Promotion Tool**: `python tools/validate_test.py {test_name}`

---

### Gate 3: Publication Ready (peer-reviewable)
**Location**: `tests/tier{N}/{domain}/test_{name}.py` (with publication flag)

**Purpose**: Ready for external scrutiny

**Criteria**:
- [ ] All Gate 2 criteria met
- [ ] Figures publication-quality
- [ ] Documentation at paper-draft level
- [ ] Independent verification by second researcher
- [ ] Code review completed
- [ ] Performance benchmarked
- [ ] Failure modes understood
- [ ] Limitations explicitly documented

**Deliverables**:
- Paper draft or technical report
- Reproducibility package
- Zenodo/arXiv upload-ready archive

---

## Workflow Examples

### Starting a New Investigation

```bash
# 1. Create investigation directory
mkdir experiments/circular_orbit_investigation
cd experiments/circular_orbit_investigation

# 2. Create script and notes
touch script.py notes.md
mkdir results analysis

# 3. Explore and document
python script.py
# Update notes.md with observations

# 4. When reproducible, promote to candidate
cd ../..
python tools/promote_experiment.py circular_orbit_investigation
```

### Promoting a Candidate to Validated Test

```bash
# 1. Verify reproducibility
python tools/check_promotion_readiness.py circular_orbit_investigation

# 2. If ready, promote to test
python tools/promote_candidate.py circular_orbit_investigation

# 3. This creates:
#    - tests/tier2/gravity/test_circular_orbits.py (pytest template)
#    - Entry in docs/discoveries/discoveries.json
#    - Timestamp for IP protection
```

## IP Protection Strategy

Every promotion through the gates creates timestamped records:

1. **Initial Discovery**: Experiment creation date
2. **Reproducibility Confirmed**: Gate 1 promotion timestamp
3. **Scientific Validation**: Gate 2 promotion timestamp + discovery.json entry
4. **Publication**: Gate 3 timestamp + public release date

This timeline provides evidence of conception and reduction-to-practice for patent applications.

## Best Practices

### Experiment Naming
- Use descriptive names: `circular_orbit_investigation`, not `test1`
- Include physics domain: `em_wave_dispersion`, `gravity_tidal_effects`
- Date prefixes optional: `2025-11-05_photon_drag_test`

### Notes Documentation
- Document **why** you're trying this
- Record **unexpected** observations
- Note **parameter choices** and reasoning
- Track **failed attempts** (learn from them!)

### Results Organization
```
results/
├── raw/              # Unprocessed simulation outputs
├── processed/        # Analyzed data
└── figures/          # Plots and visualizations
```

### Code Quality Evolution
- **Gate 0 (Experiments)**: Messy is OK, prioritize speed
- **Gate 1 (Candidates)**: Clean enough to run elsewhere
- **Gate 2 (Tests)**: Professional quality, documented
- **Gate 3 (Publication)**: Exemplary, tutorial-worthy

## Separate Track: Performance Work

Performance optimizations follow a different path (see `performance/README.md`):
- Benchmarks establish baseline performance
- Optimizations improve speed/memory without changing physics
- Performance tests validate correctness after optimization

**Key Distinction**: Physics validation and performance optimization are independent concerns. Don't conflate "faster" with "more valid."

## Tools Reference

- `tools/promote_experiment.py {name}` - Move experiment to candidates/
- `tools/promote_candidate.py {name}` - Create test skeleton from candidate
- `tools/check_promotion_readiness.py {name}` - Verify promotion criteria
- `tools/archive_experiment.py {name}` - Move to archive/ with notes

## Questions?

This is a living process. If the gates feel too rigid or too loose, adjust them. The goal is to **support discovery while maintaining rigor**, not to create bureaucracy.

---

**Remember**: Every major physics breakthrough starts as a weird experiment. The goal of this structure is to help you recognize which weird experiments are worth pursuing to publication.
