import numpy as onp
import numpy.testing as onptest
import jax
import jax.numpy as np
import meshio
import unittest
from src.fem.jax_fem import Mesh, LinearElasticity, solver, save_sol
from src.fem.tests.utils import modify_vtu_file


class Test(unittest.TestCase):
    """Test linear elasticity with cylinder mesh
    """

    # Define some class variables shared by all tests
    fenicsx_vtu_path_raw = "src/fem/tests/linear_elasticity_cylinder/fenicsx/sol_p0_000000.vtu"
    fenicsx_vtu_path = "src/fem/tests/linear_elasticity_cylinder/fenicsx/sol.vtu"
    modify_vtu_file(fenicsx_vtu_path_raw, fenicsx_vtu_path)
    fenicsx_vtu = meshio.read(fenicsx_vtu_path)
    cells = fenicsx_vtu.cells_dict['VTK_LAGRANGE_HEXAHEDRON8'] # 'hexahedron'
    points = fenicsx_vtu.points
    mesh = Mesh(points, cells)
    R = 5.
    H = 10.

    # @unittest.skip("Temporarily skip")
    def test_solve_problem(self):
        """Compare FEniCSx solution with JAX-FEM
        """
        def top(point):
            return np.isclose(point[2], Test.H, atol=1e-5)

        def bottom(point):
            return np.isclose(point[2], 0., atol=1e-5)

        def dirichlet_val(point):
            return 0.

        def neumann_val(point):
            return np.array([1., 0., 0.])

        def body_force(point):
            return np.array([0.1*point[0], 0.2*point[1], 0.3*point[2]])

        location_fns = [bottom, bottom, bottom]
        value_fns = [dirichlet_val, dirichlet_val, dirichlet_val]
        vecs = [0, 1, 2]
        dirichlet_bc_info = [location_fns, vecs, value_fns]

        neumann_bc_info = [[top], [neumann_val]]

        problem = LinearElasticity('linear_elasticity_cylinder', Test.mesh, dirichlet_bc_info, neumann_bc_info, body_force)
        sol = solver(problem)

        jax_vtu_path = f"src/fem/tests/linear_elasticity_cylinder/jax_fem/sol.vtu"
        save_sol(problem, sol, jax_vtu_path)
        jax_fem_vtu = meshio.read(jax_vtu_path)

        fenicsx_sol = Test.fenicsx_vtu.point_data['sol']
        jax_fem_sol = jax_fem_vtu.point_data['sol']

        print(f"Solution absolute value differs by {np.max(np.absolute(jax_fem_sol - fenicsx_sol))} between FEniCSx and JAX-FEM")
  
        onptest.assert_array_almost_equal(fenicsx_sol, jax_fem_sol, decimal=4)


    # @unittest.skip("Temporarily skip")
    def test_surface_integral(self):
        """Compute the top surface area of the cylinder with FEniCSx and JAX-FEM
        """
        fenicsx_surface_area = np.load(f"src/fem/tests/linear_elasticity_cylinder/fenicsx/surface_area.npy")
        def top(point):
            return np.isclose(point[2], Test.H, atol=1e-5)

        def neumann_val(point):
            return np.array([0., 0., 0.]) 

        neumann_bc_info = [[top], [neumann_val]]
        problem = LinearElasticity('linear_elasticity_cylinder', Test.mesh, None, neumann_bc_info)
        dofs = np.zeros((len(problem.mesh.points), problem.vec))

        jax_fem_area = problem.surface_integral(top, None, dofs)[0]

        # boundary_inds_list, _ = problem.Neuman_boundary_conditions()
        # boundary_inds = boundary_inds_list[0]
        # jax_fem_area = np.sum(problem.face_scale[boundary_inds[:, 0], boundary_inds[:, 1]])
        print(f"Circle area is {np.pi*Test.R**2}")
        print(f"FEniCSx computes approximate area to be {fenicsx_surface_area}")
        print(f"JAX-FEM computes approximate area to be {jax_fem_area}")

        onptest.assert_almost_equal(fenicsx_surface_area, jax_fem_area, decimal=4)


if __name__ == '__main__':
    unittest.main()
