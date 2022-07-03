from abc import ABC, abstractmethod
 

class ODESolver(ABC):
    def __init__(self, polycrystal):
        self.polycrystal = polycrystal

    @abstractmethod
    def stepper(self, state_pre, t_crt, ode_params):
        """Specify one ODE integration step. Child class must override this function.

        Args:
           state_pre (tuple):  (sol_pre, t_pre) where sol_pre is the previous solution array, t_pre is time at previous step.
           t_crt (float): Time at current step.
           ode_params (pytree): ODE parameters.

        Returns:
           tuple: (sol_crt, t_crt) where sol_crt is the current solution array, t_crt is time at current step.
        """
        pass

    @abstractmethod
    def ini_cond(self):
        """Specify initial conditions. Child class must override this function.

        Returns:
            array: The initial condition array.
        """
        pass
