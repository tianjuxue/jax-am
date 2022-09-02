import matplotlib.pyplot as plt
import numpy as np
import os

# Latex style plot
plt.rcParams.update({
    "text.latex.preamble": r"\usepackage{amsmath}",
    "text.usetex": True,
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica"]})


def plot_plastic_stress_strain():
    problem_name = "plasticity"
    plasticity_path = f"src/fem/applications/fem_examples/data/"
    disps_path = os.path.join(plasticity_path, 'numpy', problem_name, 'fenicsx/disps.npy')
    fenicsx_avg_stresses_path = os.path.join(plasticity_path, 'numpy', problem_name, 'fenicsx/avg_stresses.npy')
    jax_fem_avg_stresses_path = os.path.join(plasticity_path, 'numpy', problem_name, 'jax_fem/avg_stresses.npy')
    fenicsx_avg_stresses = np.load(fenicsx_avg_stresses_path)
    jax_fem_avg_stresses = np.load(jax_fem_avg_stresses_path)
    disps = np.load(disps_path)

    fig = plt.figure(figsize=(8, 6)) 
    plt.plot(disps, fenicsx_avg_stresses, label='FEniCSx', color='blue', linestyle="-", linewidth=2)
    plt.plot(disps, jax_fem_avg_stresses, label='JAX-FEM', color='red', marker='o', markersize=8, linestyle='None')    
    plt.xlabel(r'Displacement of top surface', fontsize=20)
    plt.ylabel(r'Volume averaged stress (z-z)', fontsize=20)
    plt.tick_params(labelsize=18)
    plt.legend(fontsize=20, frameon=False)
    plt.savefig(os.path.join(plasticity_path, 'pdf/plasticity_stress_strain.pdf'), bbox_inches='tight')


if __name__ == '__main__':
    plot_plastic_stress_strain()
    # plt.show()
