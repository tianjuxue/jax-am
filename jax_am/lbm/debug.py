import jax
import jax.numpy as np

def to_id_xyz(lattice_id):
    # E.g., lattice_id = N**2 + 2N + 3 will be converted to (1, 2, 3)
    id_z = lattice_id % N
    lattice_id = lattice_id // N
    id_y = lattice_id % N
    id_x = lattice_id // N    
    return id_x, id_y, id_z 

def extract_3x3x3_grid(lattice_id, value_tensor):
    id_x, id_y, id_z = to_id_xyz(lattice_id)
    grid_index = np.ix_(np.array([(id_x - 1) % N, id_x, (id_x + 1) % N]), 
                        np.array([(id_y - 1) % N, id_y, (id_y + 1) % N]),
                        np.array([(id_z - 1) % N, id_z, (id_z + 1) % N]))
    grid_values = value_tensor[grid_index]
    return grid_values


def extract_7x3x3_grid(lattice_id, values):
    id_x, id_y, id_z = to_id_xyz(lattice_id)
    grid_index = np.ix_(np.array([(id_x - 3) % Nx, (id_x - 2) % Nx, (id_x - 1) % Nx, id_x, (id_x + 1) % Nx, (id_x + 2) % Nx, (id_x + 3) % Nx]), 
                        np.array([(id_y - 1) % Ny, id_y, (id_y + 1) % Ny]),
                        np.array([(id_z - 1) % Nz, id_z, (id_z + 1) % Nz]))
    grid_values = values[grid_index]
    return grid_values

def memory_test0(lattice_id, value_tensor):
    value_tensor_local = extract_3x3x3_grid(lattice_id, value_tensor) # (3, 3, 3, dof)
    return np.sum(value_tensor_local)

memory_test0_vmap = jax.jit(jax.vmap(memory_test0, in_axes=(0, None)))

def memory_test1(lattice_id, value_tensor):
    value_tensor_local = extract_3x3x3_grid(lattice_id, value_tensor) # (3, 3, 3, dof)
    vel = np.ones((3, 3, 3, dof, 1))
    u_local = value_tensor_local[:, :, :, :, None] * vel # (3, 3, 3, dof, 2)
    return np.sum(u_local)

memory_test1_vmap = jax.jit(jax.vmap(memory_test1, in_axes=(0, None)))

def memory_test2(lattice_id, value_tensor):
    value_tensor_local = extract_3x3x3_grid(lattice_id, value_tensor) # (3, 3, 3, dof)
    vel = np.ones((3, 3, 3, dof, 2))
    u_local = value_tensor_local[:, :, :, :, None] * vel # (3, 3, 3, dof, 1)
    return np.sum(u_local)

memory_test2_vmap = jax.jit(jax.vmap(memory_test2, in_axes=(0, None)))

def memory_test4(lattice_id, value_tensor):
    value_tensor_local = extract_7x3x3_grid(lattice_id, value_tensor) # (7, 3, 3, dof)
    return np.sum(value_tensor_local, axis=0)

memory_test4_vmap = jax.jit(jax.vmap(memory_test4, in_axes=(0, None)))

N = 300
Nx, Ny, Nz = N, N, N
dof = 19

key = jax.random.PRNGKey(0)
value_tensor = jax.random.normal(key, (N, N, N, dof))

result4 = memory_test4_vmap(np.arange(N*N*N), value_tensor)
print(f"max result1 = {np.max(result4)}")
# print(jax.make_jaxpr(memory_test0_vmap)(np.arange(N*N*N), value_tensor))

# result2 = memory_test2_vmap(np.arange(N*N*N), value_tensor)

