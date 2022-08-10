import matplotlib.pyplot as plt
import numpy as np

# Latex style plot
plt.rcParams.update({
    "text.latex.preamble": r"\usepackage{amsmath}",
    "text.usetex": True,
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica"]})


def modify_vtu_file(input_file_path, output_file_path):
    """Convert version 2.2 of vtu file to version 1.0
    meshio does not accept version 2.2, raising error of
    meshio._exceptions.ReadError: Unknown VTU file version '2.2'.
    """
    fin = open(input_file_path, "r")
    fout = open(output_file_path, "w")
    for line in fin:
        fout.write(line.replace('<VTKFile type="UnstructuredGrid" version="2.2">', '<VTKFile type="UnstructuredGrid" version="1.0">'))
    fin.close()
    fout.close()

def plot_plastic_stress_strain():
    problem_name = 'plasticity'
    disps = np.load(f"src/fem/tests/plasticity/fenicsx/disps.npy")
    fenicsx_avg_stresses = np.load(f"src/fem/tests/{problem_name}/fenicsx/avg_stresses.npy")
    jax_fem_avg_stresses = np.load(f"src/fem/tests/{problem_name}/jax_fem/avg_stresses.npy")

    fig = plt.figure(figsize=(8, 6)) 
    plt.plot(disps, fenicsx_avg_stresses, label='FEniCSx', color='blue', linestyle="-", linewidth=2)
    plt.plot(disps, jax_fem_avg_stresses, label='JAX-FEM', color='red', marker='o', markersize=8, linestyle='None')    
    plt.xlabel(r'Displacement of top surface', fontsize=20)
    plt.ylabel(r'Volume averaged stress (z-z)', fontsize=20)
    plt.tick_params(labelsize=18)
    plt.legend(fontsize=20, frameon=False)
    # plt.savefig(f'data/pdf/multi_layer_vol.pdf', bbox_inches='tight')


if __name__ == '__main__':
    plot_plastic_stress_strain()
    plt.show()



