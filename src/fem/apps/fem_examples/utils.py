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
    problem_names = ["linear_elasticity", "hyperelasticity", "plasticity"]
    data_path = f"src/fem/apps/fem_examples/data/"
    y_lables = [r'Force on top surface [N]', r'Force on top surface [N]', r'Volume averaged stress (z-z) [MPa]']
    ratios = [1e-3, 1e-3, 1.]

    for i in range(len(problem_names)):
        disps_path = os.path.join(data_path, 'numpy', problem_names[i], 'fenicsx/disps.npy')
        fenicsx_forces_path = os.path.join(data_path, 'numpy', problem_names[i], 'fenicsx/forces.npy')
        jax_fem_forces_path = os.path.join(data_path, 'numpy', problem_names[i], 'jax_fem/forces.npy')
        fenicsx_forces = np.load(fenicsx_forces_path)
        jax_fem_forces = np.load(jax_fem_forces_path)
        disps = np.load(disps_path)
        fig = plt.figure(figsize=(8, 6)) 
        plt.plot(disps, fenicsx_forces*ratios[i], label='FEniCSx', color='blue', linestyle="-", linewidth=2)
        plt.plot(disps, jax_fem_forces*ratios[i], label='JAX-FEM', color='red', marker='o', markersize=8, linestyle='None') 
        plt.xlabel(r'Displacement of top surface [mm]', fontsize=20)
        plt.ylabel(y_lables[i], fontsize=20)
        plt.tick_params(labelsize=18)
        plt.legend(fontsize=20, frameon=False)
        plt.savefig(os.path.join(data_path, f'pdf/{problem_names[i]}_stress_strain.pdf'), bbox_inches='tight')



def plot_performance():
    data_path = f"src/fem/apps/fem_examples/data/"
    abaqus_time = np.loadtxt(os.path.join(data_path, f"txt/abaqus_fem_time.txt"))
    fenicsx_time_np_1 = np.loadtxt(os.path.join(data_path, f"txt/fenicsx_fem_time_mpi_np_1.txt"))
    fenicsx_time_np_2 = np.loadtxt(os.path.join(data_path, f"txt/fenicsx_fem_time_mpi_np_2.txt"))
    fenicsx_time_np_4 = np.loadtxt(os.path.join(data_path, f"txt/fenicsx_fem_time_mpi_np_4.txt"))
    jax_time_cpu = np.loadtxt(os.path.join(data_path, f"txt/jax_fem_cpu_time.txt"))  
    jax_time_gpu = np.loadtxt(os.path.join(data_path, f"txt/jax_fem_gpu_time.txt"))  
    cpu_dofs = np.loadtxt(os.path.join(data_path, f"txt/jax_fem_cpu_dof.txt"))   
    gpu_dofs = np.loadtxt(os.path.join(data_path, f"txt/jax_fem_gpu_dof.txt"))   

    # tick_labels = ['25x25x25', '50x50x50', '100x100x100']
    # labels = ['fenicsx-np-1', 'fenicsx-np-2', 'fenicsx-np-4', 'jax-cpu', 'jax-gpu']
    # colors = ['orange', 'purple', 'green', 'red', 'blue']
    # markers = ['^', '^', '^', 'o', 'o']


    plt.figure(figsize=(12, 9))
    # plt.figure()
    # plt_tmp = np.arange(len(problem[0])) + 1
    # for j, p in enumerate(problem):
    #     plt.plot(plt_tmp, p, linestyle='-', marker=markers[j], markersize=10, linewidth=2, color=colors[j], label=labels[j])

    plt.plot(gpu_dofs[1:], abaqus_time[1:], linestyle='-', marker='s', markersize=12, linewidth=2, color='orange', label='Abaqus CPU')
    plt.plot(cpu_dofs[1:], fenicsx_time_np_1[1:], linestyle='-', marker='^', markersize=12, linewidth=2, color='purple', label='FEniCSx CPU MPI 1')
    plt.plot(cpu_dofs[1:], fenicsx_time_np_2[1:], linestyle='-', marker='^', markersize=12, linewidth=2, color='green', label='FEniCSx CPU MPI 2')
    plt.plot(cpu_dofs[1:], fenicsx_time_np_4[1:], linestyle='-', marker='^', markersize=12, linewidth=2, color='pink', label='FEniCSx CPU MPI 4')
    plt.plot(cpu_dofs[1:], jax_time_cpu[1:], linestyle='-', marker='o', markersize=12, linewidth=2, color='blue', label='JAX-FEM CPU')
    plt.plot(gpu_dofs[1:], jax_time_gpu[1:], linestyle='-', marker='o', markersize=12, linewidth=2, color='red', label='JAX-FEM GPU')


    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel("Number of DOFs", fontsize=20)
    plt.ylabel("Wall time [s]", fontsize=20)
    plt.tick_params(labelsize=20)
    ax = plt.gca()
    # ax.get_xaxis().set_tick_params(which='minor', size=0)
    # plt.xticks(plt_tmp, tick_labels)
    plt.tick_params(labelsize=20)
    plt.legend(fontsize=20, frameon=False)   
 
    plt.savefig(os.path.join(data_path, f'pdf/performance.pdf'), bbox_inches='tight')

if __name__ == '__main__':
    # plot_plastic_stress_strain()
    plot_performance()
    plt.show()
