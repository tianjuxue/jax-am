#!/bin/bash
pip install --upgrade pip
pip install --upgrade "jax[cuda]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
pip install gmsh
pip install meshio
pip install orix
pip install matplotlib
pip install scipy
# conda install cuda -c nvidia
