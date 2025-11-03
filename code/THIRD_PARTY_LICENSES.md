# Third-Party Licenses and Attribution

<!-- Copyright (c) 2025 Greg D. Partin. All rights reserved. -->
<!-- Licensed under CC BY-NC-ND 4.0 (Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International). -->
<!-- See LICENSE file in project root for full license text. -->
<!-- Commercial use prohibited without explicit written permission. -->
<!-- Contact: latticefieldmediumresearch@gmail.com -->


This document provides attribution and license information for all third-party open-source libraries used in the Lattice Field Medium (LFM) project.

For how to cite these libraries in academic work, see docs/REFERENCES.md (with BibTeX in docs/references.bib).

**LFM License:** Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International (CC BY-NC-ND 4.0)  
**Copyright:** (c) 2025 Greg D. Partin

---

## Overview of Third-Party Dependencies

The LFM project depends on the following open-source libraries. Each library retains its original license. The LFM CC BY-NC-ND 4.0 license applies only to LFM-authored code, NOT to these dependencies.

All dependencies listed below use **permissive licenses** that are compatible with non-commercial use and redistribution.

---

## 1. NumPy

**Description:** Fundamental package for scientific computing with Python  
**Version Required:** >= 1.24.0  
**License:** BSD 3-Clause License  
**Homepage:** https://numpy.org/  
**Copyright:** Copyright (c) 2005-2023, NumPy Developers

### License Summary
NumPy is licensed under the BSD 3-Clause "New" or "Revised" License, a permissive open-source license. It allows:
- ✅ Commercial and non-commercial use
- ✅ Modification and distribution
- ✅ Private use

**Full License Text:** https://numpy.org/doc/stable/license.html

### BSD 3-Clause License (Simplified)
```
Redistribution and use in source and binary forms, with or without modification, 
are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice
2. Redistributions in binary form must reproduce the above copyright notice
3. Neither the name of NumPy nor the names of contributors may be used to 
   endorse or promote products derived from this software without specific 
   prior written permission

THIS SOFTWARE IS PROVIDED "AS IS" WITHOUT WARRANTY OF ANY KIND.
```

---

## 2. Matplotlib

**Description:** Comprehensive library for creating static, animated, and interactive visualizations  
**Version Required:** >= 3.7.0  
**License:** PSF-based License (Python Software Foundation style)  
**Homepage:** https://matplotlib.org/  
**Copyright:** Copyright (c) 2002-2023, John D. Hunter, Michael Droettboom, and the Matplotlib development team

### License Summary
Matplotlib uses a license based on the Python Software Foundation license. It is permissive and allows:
- ✅ Commercial and non-commercial use
- ✅ Modification and distribution
- ✅ Private use

**Full License Text:** https://matplotlib.org/stable/users/project/license.html

### Key Terms
Matplotlib's license is very permissive and compatible with both open-source and proprietary projects. No copyleft requirements.

---

## 3. SciPy

**Description:** Fundamental algorithms for scientific computing in Python  
**Version Required:** >= 1.10.0  
**License:** BSD 3-Clause License  
**Homepage:** https://scipy.org/  
**Copyright:** Copyright (c) 2001-2023, SciPy Developers

### License Summary
SciPy uses the BSD 3-Clause License (same as NumPy). It allows:
- ✅ Commercial and non-commercial use
- ✅ Modification and distribution
- ✅ Private use

**Full License Text:** https://scipy.org/scipylib/license.html

### BSD 3-Clause License
Same terms as NumPy above. Very permissive, no copyleft.

---

## 4. h5py

**Description:** Pythonic interface to the HDF5 binary data format  
**Version Required:** >= 3.8.0  
**License:** BSD 3-Clause License  
**Homepage:** https://www.h5py.org/  
**Copyright:** Copyright (c) 2008-2023, h5py contributors

### License Summary
h5py uses the BSD 3-Clause License. It allows:
- ✅ Commercial and non-commercial use
- ✅ Modification and distribution
- ✅ Private use

**Full License Text:** https://github.com/h5py/h5py/blob/master/LICENSE

### Note on HDF5 Dependency
h5py wraps the HDF5 C library, which is licensed under a BSD-style license from The HDF Group. Both are permissive licenses.

---

## 5. pytest

**Description:** Testing framework for Python  
**Version Required:** >= 7.3.0  
**License:** MIT License  
**Homepage:** https://pytest.org/  
**Copyright:** Copyright (c) 2004-2023, Holger Krekel and pytest-dev team

### License Summary
pytest uses the MIT License, one of the most permissive open-source licenses. It allows:
- ✅ Commercial and non-commercial use
- ✅ Modification and distribution
- ✅ Private use
- ✅ Sublicensing

**Full License Text:** https://github.com/pytest-dev/pytest/blob/main/LICENSE

### MIT License (Simplified)
```
Permission is hereby granted, free of charge, to any person obtaining a copy of 
this software, to deal in the Software without restriction, including without 
limitation the rights to use, copy, modify, merge, publish, distribute, 
sublicense, and/or sell copies of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND.
```

**Note:** pytest is used only for development/testing and is not required for end-user execution of LFM.

---

## 6. CuPy (Optional)

**Description:** NumPy/SciPy-compatible Array Library for GPU-accelerated Computing with Python  
**Version Required:** >= 12.0.0 (cupy-cuda12x)  
**License:** MIT License  
**Homepage:** https://cupy.dev/  
**Copyright:** Copyright (c) 2015-2023, Preferred Networks, Inc. and Preferred Infrastructure, Inc.

### License Summary
CuPy uses the MIT License. It allows:
- ✅ Commercial and non-commercial use
- ✅ Modification and distribution
- ✅ Private use
- ✅ Sublicensing

**Full License Text:** https://github.com/cupy/cupy/blob/main/LICENSE

### Note on CUDA Dependency
CuPy requires NVIDIA CUDA Toolkit, which has its own license (NVIDIA CUDA EULA). CUDA is proprietary but free for development and use. Review NVIDIA's license if redistributing.

**Note:** CuPy is an optional dependency for GPU acceleration. LFM runs without it using CPU-only NumPy.

---

## 7. Development Dependencies (Optional)

The following are used only in development and are not required for end users:

### pytest-cov
- **License:** MIT License
- **Use:** Test coverage reporting
- **Required:** No (development only)

### black
- **License:** MIT License
- **Use:** Code formatting
- **Required:** No (development only)

### flake8
- **License:** MIT License
- **Use:** Code linting
- **Required:** No (development only)

### mypy
- **License:** MIT License
- **Use:** Static type checking
- **Required:** No (development only)

---

## License Compatibility Analysis

### Summary Table

| Library | License | Commercial Use | Copyleft | Compatible with LFM |
|---------|---------|----------------|----------|---------------------|
| NumPy | BSD 3-Clause | ✅ Yes | ❌ No | ✅ Yes |
| Matplotlib | PSF-based | ✅ Yes | ❌ No | ✅ Yes |
| SciPy | BSD 3-Clause | ✅ Yes | ❌ No | ✅ Yes |
| h5py | BSD 3-Clause | ✅ Yes | ❌ No | ✅ Yes |
| pytest | MIT | ✅ Yes | ❌ No | ✅ Yes |
| CuPy | MIT | ✅ Yes | ❌ No | ✅ Yes |

### Compatibility Statement

**All third-party dependencies are compatible with the LFM CC BY-NC-ND 4.0 license.**

The permissive licenses (BSD, MIT, PSF) used by these libraries allow LFM to:
1. Use them in a non-commercial project (✅)
2. Redistribute them with LFM (✅)
3. Apply a more restrictive license (CC BY-NC-ND 4.0) to LFM's own code (✅)

**Important:** LFM's CC BY-NC-ND 4.0 license does NOT change the licenses of these dependencies. They remain under their original permissive licenses.

---

## Attribution Requirements

When distributing LFM or citing it in academic work, you should acknowledge:

1. **LFM itself:** Greg D. Partin, CC BY-NC-ND 4.0, 2025
2. **NumPy:** Harris, C.R., Millman, K.J., van der Walt, S.J. et al. (2020). Array programming with NumPy. Nature, 585, 357–362.
3. **Matplotlib:** Hunter, J. D. (2007). Matplotlib: A 2D graphics environment. Computing in Science & Engineering, 9(3), 90-95.
4. **SciPy:** Virtanen, P. et al. (2020). SciPy 1.0: fundamental algorithms for scientific computing in Python. Nature Methods, 17, 261–272.

---

## Obtaining Dependency Source Code

All dependencies are open source and available:

- **NumPy:** https://github.com/numpy/numpy
- **Matplotlib:** https://github.com/matplotlib/matplotlib
- **SciPy:** https://github.com/scipy/scipy
- **h5py:** https://github.com/h5py/h5py
- **pytest:** https://github.com/pytest-dev/pytest
- **CuPy:** https://github.com/cupy/cupy

---

## No Warranty

The third-party libraries are provided "AS IS" without warranty under their respective licenses. See each library's license for detailed warranty disclaimers.

---

## Questions?

For questions about:
- **LFM licensing:** Contact latticefieldmediumresearch@gmail.com
- **Third-party library licenses:** Contact the respective project maintainers (see homepages above)

---

**Last Updated:** November 1, 2025  
**LFM Version:** 1.0  
**Document Maintainer:** Greg D. Partin
