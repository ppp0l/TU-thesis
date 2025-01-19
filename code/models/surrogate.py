from abc import ABC, abstractmethod

class Surrogate(ABC):
    @abstractmethod
    def fit(self, train_x, train_y, noise):
        pass

    @abstractmethod
    def predict(self, x):
        pass
    
    @abstractmethod
    def update(self, new_X, new_y, new_noise = None, updated_y = None, updated_noise = None):
        pass
    
    @abstractmethod
    def state_dict(self):
        pass

    @abstractmethod
    def load_state_dict(self, state_dict):
        pass

