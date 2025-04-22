import csv 
import numpy as np
import subprocess

model_path = "/home/numerik/pvillani/Project/lipschitz/code/kaskade/beam"
out_path = "kaskade/beam/output"

class Adaptive_beam :
    """
    Python interface for the adaptive FE model of the beam.
    """
    dout = 6

    def __init__(self,
                 data_path : str,
                 adaptive : bool = True ,
                 out_path : str = out_path ,
                 default_tol : float = 1.e-3,
                 ) :
        
        self.adaptive = adaptive

        if adaptive :
            self.eval_param = []


        else :
            self.default_tol = default_tol

        self.data_path = data_path
        self.out_path = out_path



    def predict(self, 
                material_parameters : np.ndarray,
                tolerance : np.ndarray,
                ) -> np.ndarray :
        n_pts = len(material_parameters)

        if not self.adaptive :
            # use default tolerance
            tolerance = np.full(n_pts, self.default_tol)
        
    def run(self, flags: str) -> None:
        """
        Run the adaptive beam model.
        """
        flags = "--tolx 0.001 --order 2 > out.txt"
        # run the adaptive beam model
        cmd = f"{model_path}/adaBeam {flags}"
        subprocess.run(cmd, shell=True)