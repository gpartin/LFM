# References and Software Citations

This project builds upon the scientific Python ecosystem. If you publish work using LFM, please:

1. Cite LFM itself (see README “Citation” or the repository’s DOI badge).
2. Cite the core software you used (NumPy, SciPy, Matplotlib, etc.).
3. Cite any optional accelerators or tools (e.g., CuPy) that you relied on.

Below are recommended citations with copy‑paste BibTeX where available.

---

## Core Scientific Python Stack

- NumPy

  Harris, C. R., Millman, K. J., van der Walt, S. J., et al. (2020). Array programming with NumPy. Nature, 585, 357–362. doi:10.1038/s41586-020-2649-2

  ```bibtex
  @article{harris2020array,
    title={Array programming with NumPy},
    author={Harris, Charles R and Millman, K Jarrod and van der Walt, St{\'e}fan J and others},
    journal={Nature},
    volume={585},
    number={7825},
    pages={357--362},
    year={2020},
    publisher={Nature Publishing Group},
    doi={10.1038/s41586-020-2649-2}
  }
  ```

- SciPy

  Virtanen, P., Gommers, R., Oliphant, T. E., et al. (2020). SciPy 1.0: fundamental algorithms for scientific computing in Python. Nature Methods, 17, 261–272. doi:10.1038/s41592-019-0686-2

  ```bibtex
  @article{virtanen2020scipy,
    title={SciPy 1.0: fundamental algorithms for scientific computing in Python},
    author={Virtanen, Pauli and Gommers, Ralf and Oliphant, Travis E and others},
    journal={Nature Methods},
    volume={17},
    pages={261--272},
    year={2020},
    doi={10.1038/s41592-019-0686-2}
  }
  ```

- Matplotlib

  Hunter, J. D. (2007). Matplotlib: A 2D graphics environment. Computing in Science & Engineering, 9(3), 90–95. doi:10.1109/MCSE.2007.55

  ```bibtex
  @article{hunter2007matplotlib,
    title={Matplotlib: A 2D graphics environment},
    author={Hunter, John D},
    journal={Computing in Science \& Engineering},
    volume={9},
    number={3},
    pages={90--95},
    year={2007},
    publisher={IEEE},
    doi={10.1109/MCSE.2007.55}
  }
  ```

- h5py

  The h5py Developers. h5py — HDF5 for Python. https://www.h5py.org/

- pytest

  pytest — a framework for testing Python. https://docs.pytest.org/

---

## Optional GPU Acceleration

- CuPy

  Okuta, R., Unno, Y., Nishino, D., Hido, S., & Loomis, C. (2017). CuPy: A NumPy-Compatible Library for NVIDIA GPU Calculations. Proceedings of the Workshop on Machine Learning Systems (LearningSys), NIPS 2017. https://cupy.dev/

---

## Domain Literature (Foundational Physics)

### Klein-Gordon Equation (Foundational)

LFM builds upon the Klein-Gordon equation first developed by Oskar Klein and Walter Gordon in 1926:

- **Klein, Oskar (1926)**. Quantentheorie und fünfdimensionale Relativitätstheorie. *Zeitschrift für Physik*, 37(12), 895-906.

  ```bibtex
  @article{klein1926,
    title={Quantentheorie und fünfdimensionale Relativitätstheorie},
    author={Klein, Oskar},
    journal={Zeitschrift für Physik},
    volume={37},
    number={12},
    pages={895--906},
    year={1926},
    publisher={Springer},
    doi={10.1007/BF01397481}
  }
  ```

- **Gordon, Walter (1926)**. Der Comptoneffekt nach der Schrödingerschen Theorie. *Zeitschrift für Physik*, 40(1-2), 117-133.

  ```bibtex
  @article{gordon1926,
    title={Der Comptoneffekt nach der Schrödingerschen Theorie},
    author={Gordon, Walter},
    journal={Zeitschrift für Physik},
    volume={40},
    number={1-2},
    pages={117--133},
    year={1926},
    publisher={Springer},
    doi={10.1007/BF01390840}
  }
  ```

### Additional Relevant Literature

If your work draws on specific physical models, derivations, or prior art related to:
- Klein–Gordon equation discretizations and numerical methods
- CFL stability analyses for wave equations  
- Gravity analogues and optical metrics references
- Lattice field theory computational approaches

Consider maintaining a project-wide `docs/references.bib` and citing from there.
