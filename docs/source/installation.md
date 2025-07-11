# Install

JAX-AM supports Linux and macOS. Create a new conda environment and run

```bash
git clone https://github.com/tianjuxue/jax-am.git
cd jax-am
pip install .
```

>**Note**: JAX-AM depends on [petsc4py](https://www.mcs.anl.gov/petsc/petsc4py-current/docs/usrman/index.html). We have found difficulty installing `petsc4py` with `pip` on certain platforms. Installing `petsc4py` with `conda` is therefore recommended (see [here](https://anaconda.org/conda-forge/petsc4py)).
