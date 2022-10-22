# import os
# import jax
# import jax.numpy as np
# import numpy as onp
# from functools import partial
# from jax_am.phase_field.integrator import MultiVarSolver
# from jax_am.phase_field.utils import Field
# from jax_am.phase_field.yaml_parse import args

# os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# def set_params():
#     args['case'] = 'fd_solidification'
#     args['num_grains'] = 10000
#     args['domain_x'] = 1.
#     args['domain_y'] = 0.01
#     args['domain_z'] = 1.
#     args['write_sol_interval'] = 1000
#     args['laser_path']['time'][-1] = 0.0050

#     # If args['anisotropy'] = 0., isotropic evolution.
#     args['anisotropy'] = 0.15


# def neper_domain():
#     set_params()
#     neper_path = os.path.join(args['root_path'], f"data/neper/{args['case']}")
#     os.makedirs(neper_path, exist_ok=True)
#     os.system(f'''neper -T -n {args['num_grains']} -id 1 -regularization 0 -domain "cube({args['domain_x']},\
#                {args['domain_y']},{args['domain_z']})" \
#                 -o {neper_path}/domain -format tess,obj,ori''')
#     os.system(f"neper -T -loadtess {neper_path}/domain.tess -statcell x,y,z,vol,facelist -statface x,y,z,area")
#     os.system(f"neper -M -rcl 1 -elttype hex -faset faces {neper_path}/domain.tess")


# # @partial(jax.jit, static_argnums=(1,))
# def get_T_quench(t, polycrystal):
#     '''
#     Given spatial coordinates and t, we prescribe the value of T.
#     '''
#     centroids = polycrystal.centroids
#     z = centroids[:, 2]
#     vel = 200.
#     thermal_grad = 500.
#     cooling_rate = thermal_grad * vel
#     t_total = args['domain_z'] / vel
#     T = args['T_melt'] + thermal_grad * z - cooling_rate * t
#     return T[:, None]


# def run():
#     set_params()
#     solver = MultiVarSolver(get_T_quench)
#     solver.solve()


# if __name__ == "__main__":
#     # neper_domain()
#     run()
