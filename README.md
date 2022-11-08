<p align="middle">
  <img src="docs/materials/logo.png" width="200" />
</p>

A GPU-accelerated differentiable simulation toolbox for additive manufacturing (AM) based on [JAX](https://github.com/google/jax). 

# JAX-AM

[![Doc](https://img.shields.io/readthedocs/jax-am)](https://jax-am.readthedocs.io/en/latest/)
![PyPI](https://img.shields.io/pypi/v/jax-am) ![Github Star](https://img.shields.io/github/stars/tianjuxue/jax-am)
![Github Fork](https://img.shields.io/github/forks/tianjuxue/jax-am)
![License](https://img.shields.io/github/license/tianjuxue/jax-am)

JAX-AM is a collection of several numerical tools, currently including __Discrete Element Method (DEM)__, __Computational Fluid Dynamics (CFD)__, __Phase Field Method (PFM)__ and __Finite Element Method (FEM)__, that cover the analysis of the __Process-Structure-Property__ relationship in AM. 

Our vision is to share with the AM community a __free, open-source__ (under the GPL-3.0 License) software that facilitates the relevant computational research. In the JAX ecosystem, we hope to emphasize the potential of JAX for scientific computing. At the same time, AI-enabled research in AM can be made easy with JAX-AM. 

:fire: <u>***Join us for the development of JAX-AM!***</u> :rocket:

## Discrete Element Method (DEM)

DEM simulation can be used for simulating powder dynamics in metal AM.

<p align="middle">
  <img src="docs/materials/dem.gif" width="400" />
</p>
<p align="middle">
    <em >Free falling of 64,000 spherical particles.</em>
</p>


## Computational Fluid Dynamics (CFD)

CFD helps to understand the AM process by solving the (incompressible) Navier-Stokes equations for velocity, pressure and temperature.

<p align="middle">
  <img src="docs/materials/melt_pool_dynamics.png" width="400" />
</p>
<p align="middle">
    <em >Melt pool dynamics.</em>
</p>


## Phase Field Method (PFM)

PFM models the grain development that is critical to form the structure of the as-built sample.

<p align="middle">
  <img src="docs/materials/single_track_eta_DNS.gif" width="600" />
</p>
<p align="middle">
    <em >Microstructure evolution.</em>
</p>

<p align="middle">
  <img src="docs/materials/solidification_isotropic.gif" width="300" />
  <img src="docs/materials/solidification_anisotropic.gif" width="300" /> 
</p>
<p align="middle">
    <em >Directional solidification with isotropic (left) and anisotropic (right) grain growth.</em>
</p>


## Finite Element Method (FEM)

FEM is a powerful tool for thermal-mechanical analysis in AM. We support the following features

- 2D quadrilateral/triangle elements
- 3D hexahedron/tetrahedron elements
- First and second order elements
- Dirichlet/Neumann/Cauchy/periodic boundary conditions
- Linear and nonlinear analysis including
  - Heat equation
  - Linear elasticity
  - Hyperelasticity
  - Plasticity
- Differentiable simulation for solving inverse/design problems __without__ human deriving sensitivities, e.g.,
  - Toplogy optimization
  - Optimal thermal control


<p align="middle">
  <img src="docs/materials/ded.gif" width="600" />
</p>
<p align="middle">
    <em >Thermal profile in direct energy deposition.</em>
</p>

<p align="middle">
  <img src="docs/materials/von_mises.png" width="400" />
</p>
<p align="middle">
    <em >Linear static analysis of a bracket.</em>
</p>


<p align="middle">
  <img src="docs/materials/to.gif" width="600" />
</p>
<p align="middle">
    <em >Topology optimization with differentiable simulation.</em>
</p>

## Documentation

Please see the [web documentation](https://jax-am.readthedocs.io/en/latest/) for the installation and use of this project.


## License

This project is licensed under the GNU General Public License v3 - see the [LICENSE](https://www.gnu.org/licenses/) for details.
