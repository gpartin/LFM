# Contributing to LFM (Lattice Field Medium)

Thank you for your interest in contributing to the Lattice Field Medium project! This document outlines our contribution process, intellectual property policies, and community standards.

---

## 🚨 Important: Read Before Contributing

**This project is released under CC BY-NC-ND 4.0 (Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International) for research artifacts.** This license has specific implications for contributions:

- **Non-Commercial**: Contributions are for academic/research purposes only
- **No Derivatives**: The project does not accept modifications to core theoretical content
- **Attribution Required**: All contributors are credited appropriately

**By contributing, you agree to these terms and assign copyright to the project maintainer while retaining attribution rights.**

---

## 📋 What We Accept

### ✅ Welcomed Contributions

- **Bug Reports**: Issues, crashes, incorrect physics behavior
- **Documentation Improvements**: Typos, clarifications, examples
- **Test Coverage**: New validation tests, edge cases, performance benchmarks
- **Tooling Enhancements**: Build scripts, CI/CD, visualization tools
- **Platform Support**: Cross-platform compatibility fixes (Windows/Linux/macOS)
- **Performance Optimizations**: GPU acceleration, algorithm improvements (with validation)
- **Code Quality**: Refactoring for clarity, type hints, error handling

### ❌ Not Accepting

- **Theoretical Modifications**: Changes to core physics equations or hypothesis
- **Alternative Implementations**: "Clean room" rewrites or competing frameworks
- **Commercial Features**: Proprietary extensions or paid-tier functionality
- **Scope Creep**: Features outside physics validation and reproducibility

---

## 🤝 Contribution Process

### 1. Open an Issue First

Before investing time in code, **open an issue** to discuss:
- What you want to contribute and why
- How it aligns with project goals
- Potential implementation approach

**Rationale**: Avoids wasted effort on contributions that may not fit project scope.

### 2. Fork & Branch

```bash
# Fork the repository on GitHub
git clone https://github.com/YOUR_USERNAME/LFM.git
cd LFM
git checkout -b feature/your-contribution-name
```

### 3. Follow Code Standards

**Python Code:**
- UTF-8 encoding mandatory: `# -*- coding: utf-8 -*-`
- Type hints for function signatures
- Google-style docstrings
- PEP 8 compliance (use `black` for formatting)
- Add/update tests for new functionality

**Documentation:**
- Clear, concise language
- Code examples where applicable
- Cross-reference related docs

**Commits:**
- Descriptive messages: "Fix energy conservation drift in Tier 3 tests"
- Not: "Update file" or "Fix bug"

### 4. Test Thoroughly

**Before submitting:**
```bash
# Run validation suite (physics correctness)
cd workspace/src
python run_parallel_suite.py --fast

# Run website tests (if applicable)
cd workspace/website
npm test

# Check encoding compliance
cd workspace
python tools/check_encoding_compliance.py
```

### 5. Submit Pull Request

**PR Template:**
```markdown
## Summary
Brief description of what this PR does.

## Motivation
Why is this change needed? Which issue does it close?

## Changes
- List of specific changes
- Files modified and why

## Testing
How was this tested? Include test results.

## Checklist
- [ ] Code follows project style guidelines
- [ ] All tests pass
- [ ] Documentation updated (if applicable)
- [ ] Commit messages are descriptive
- [ ] I agree to license terms (see below)
```

### 6. Code Review

Maintainers will review for:
- Correctness (does it work as intended?)
- Physics validity (does it preserve theoretical integrity?)
- Code quality (readable, maintainable, tested?)
- License compliance (proper headers, no proprietary code?)

**Expect iteration**: Feedback is normal and constructive.

---

## ⚖️ License & IP Terms

### Inbound = Outbound Rule

**Your contributions are licensed under the same terms as the project:**
- Research artifacts: CC BY-NC-ND 4.0
- Website code: MIT License (where applicable)

**This means:**
- You grant the project maintainer a perpetual, irrevocable license to use your contribution
- You retain copyright and attribution rights
- Your contribution cannot be relicensed to a more permissive license without your consent
- Maintainers may offer **commercial licenses separately** (see below)

### Developer Certificate of Origin (DCO)

All contributors must sign off commits using DCO:

```bash
git commit -s -m "Your commit message"
```

This certifies you have the right to submit the contribution under the project's license. Full DCO text: https://developercertificate.org/

**What signing off means:**
- You wrote the code or have permission to contribute it
- You understand it will be licensed under project terms
- You are not violating any employer agreements or other contracts

### Commercial Relicensing Path

**Scenario**: A company wants to use LFM (including your contributions) commercially.

**Process**:
1. Company contacts `licensing@emergentphysicslab.com` (see `COMMERCIAL_LICENSE_REQUEST.md`)
2. Maintainers negotiate commercial license terms
3. Contributors are notified and must consent (or their code is excluded)
4. Revenue sharing may be offered (negotiable, not guaranteed)

**Your Rights**:
- You can decline commercial relicensing of your contribution
- You retain attribution in all commercial uses
- You cannot unilaterally offer commercial licenses (only maintainers can)

**Important**: By default, commercial use is **prohibited**. Maintainers handle all commercial licensing separately from the open research license.

---

## 🛡️ Code of Conduct

We are committed to providing a welcoming, professional environment for all contributors.

### Expected Behavior

- **Respectful Communication**: Constructive feedback, no personal attacks
- **Scientific Integrity**: Honest reporting of results, reproducibility emphasis
- **Collaboration**: Help others, share knowledge, give credit
- **Openness**: Transparent decision-making, public discussions

### Unacceptable Behavior

- Harassment, discrimination, or hostile language
- Plagiarism or misrepresentation of work
- Spamming, trolling, or off-topic discussions
- Unauthorized commercial use or IP violations

### Enforcement

Violations will result in warnings, temporary bans, or permanent exclusion at maintainer discretion. Report issues to `security@emergentphysicslab.com`.

---

## 🔐 Security Disclosures

**Found a security vulnerability?**

**Do NOT open a public issue.** Instead:

1. Email `security@emergentphysicslab.com` with:
   - Description of the vulnerability
   - Steps to reproduce
   - Potential impact
   - Suggested fix (if known)

2. Allow 90 days for remediation before public disclosure

3. Receive credit in security acknowledgments (if desired)

**We take security seriously** and will respond promptly to responsible disclosures.

---

## 📞 Questions & Contact

- **General Questions**: Open a GitHub Discussion or Issue
- **Research Inquiries**: latticefieldmediumresearch@gmail.com
- **Commercial Licensing**: licensing@emergentphysicslab.com
- **Security Issues**: security@emergentphysicslab.com

---

## 🙏 Attribution & Acknowledgment

All contributors are credited in:
- Repository CONTRIBUTORS.md file (to be created when first contribution accepted)
- Project documentation and publications (where applicable)
- Zenodo/OSF upload packages (contributor metadata)

**Your work matters** and will be recognized appropriately.

---

## 📜 Legal Summary

By contributing, you certify that:

1. ✅ You have the legal right to contribute the work
2. ✅ You agree to license your contribution under CC BY-NC-ND 4.0 (research) or MIT (website code)
3. ✅ You understand the maintainer may offer commercial licenses separately (with your consent required for your contributions)
4. ✅ You accept the Developer Certificate of Origin (DCO) terms
5. ✅ You will not submit proprietary, patented, or encumbered code
6. ✅ You agree to the project's Code of Conduct

**If you cannot agree to these terms, please do not contribute.**

---

## 🚀 Ready to Contribute?

1. Read this document thoroughly
2. Check existing issues for "good first issue" or "help wanted" tags
3. Open an issue to discuss your planned contribution
4. Fork, branch, code, test, and submit a PR
5. Engage constructively during code review

**Thank you for helping advance open physics research! 🎉**

---

**Last Updated**: 2025-11-08  
**Version**: 1.0  
**Maintainer**: Greg D. Partin | latticefieldmediumresearch@gmail.com
