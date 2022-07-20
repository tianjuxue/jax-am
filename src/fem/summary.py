import numpy as np
import matplotlib.pyplot as plt

# Latex style plot
plt.rcParams.update({
    "text.latex.preamble": r"\usepackage{amsmath}",
    "text.usetex": True,
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica"]})


def run():

    fenicsx_time_np_1 = np.loadtxt(f"post-processing/txt/fenicsx_fem_time_mpi_np_1.txt")
    fenicsx_time_np_2 = np.loadtxt(f"post-processing/txt/fenicsx_fem_time_mpi_np_2.txt")
    fenicsx_time_np_4 = np.loadtxt(f"post-processing/txt/fenicsx_fem_time_mpi_np_4.txt")
    jax_time_cpu = np.loadtxt(f"post-processing/txt/jax_fem_cpu_time.txt")
    jax_time_gpu = np.loadtxt(f"post-processing/txt/jax_fem_gpu_time.txt")

    wall_time = [fenicsx_time_np_1, fenicsx_time_np_2, fenicsx_time_np_4, jax_time_cpu, jax_time_gpu]

    linear_elasticity = []
    linear_poisson = []
    nonlinear_poisson = []

    problems = [linear_elasticity, linear_poisson, nonlinear_poisson]


    for i, problem in enumerate(problems):
        for j, wt in enumerate(wall_time):
            problem.append(wt[i])

    problems = np.array(problems)


    tick_labels = ['25x25x25', '50x50x50', '100x100x100']
    labels = ['fenicsx-np-1', 'fenicsx-np-2', 'fenicsx-np-4', 'jax-cpu', 'jax-gpu']
    colors = ['orange', 'purple', 'green', 'red', 'blue']
    markers = ['^', '^', '^', 'o', 'o']
    problem_names = ['linear_elasticity', 'linear_poisson', 'nonlinear_poisson']


    for i, problem in enumerate(problems):
        plt.figure(figsize=(8, 6))
        # plt.figure()
        plt_tmp = np.arange(len(problem[0])) + 1
        for j, p in enumerate(problem):
            plt.plot(plt_tmp, p, linestyle='-', marker=markers[j], markersize=10, linewidth=2, color=colors[j], label=labels[j])
        plt.xscale('log')
        plt.yscale('log')
        plt.xlabel("mesh size", fontsize=20)
        plt.ylabel("wall time [s]", fontsize=20)
        plt.tick_params(labelsize=18)
        ax = plt.gca()
        ax.get_xaxis().set_tick_params(which='minor', size=0)
        plt.xticks(plt_tmp, tick_labels)
        plt.tick_params(labelsize=18)
        plt.legend(fontsize=18, frameon=False)   
        plt.savefig(f"post-processing/pdf/{problem_names[i]}", bbox_inches='tight')


if __name__ == "__main__":
    run()
    plt.show()