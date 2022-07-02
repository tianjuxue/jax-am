from abc import ABC, abstractmethod
 

class ODESolver(ABC):
    def __init__(self, polycrystal):
        self.polycrystal = polycrystal

    @abstractmethod
    def stepper(self, state_pre, t_crt, ode_params):
        '''
        Child class must override
        '''
        pass

    @abstractmethod
    def ini_cond(self):
        '''
        Child class must override
        '''
        pass
