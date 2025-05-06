import csv 
import numpy as np

import subprocess
import os

model_path = "/home/numerik/pvillani/Project/lipschitz/code/kaskade/beam"

def_scaling = {
    "out" : {
        "sum" : [0.015, 0.015, -1.e-7, 1.e-7],
        "scale" : [-200/3, -200/3, 1.3e6, -2.e6]
    },
    "in" : {
        "sum" : [-2.e11, -0.1],
        "scale" : [0.5e-11, 20/7]
    }
}



class Adaptive_beam :
    """
    Python interface for the adaptive FE model of the beam.
    """
    dout = 4
    dim = 2

    scale_in = def_scaling["in"]
    scale_out = def_scaling["out"]

    def __init__(self,
                 data_path : str,
                 adaptive : bool = True ,
                 default_tol : float = 1.e-3,
                 model_path : str = model_path,
                 mesh : str = None,
                 ) :
        
        self.adaptive = adaptive

        if adaptive :
            self.eval_param = []


        else :
            self.default_tol = default_tol

        self.data_path = data_path

        if os.path.exists(data_path + "/eval") :
            # remove the directory
            subprocess.run(f"rm -rf {data_path}/eval", shell=True)
            
        self.model_path = model_path

        self.mesh = mesh

    def scale_parameters(self, parameters : np.ndarray, inverse : bool = False) -> np.ndarray :
        """
        Scale the parameter space to unit square and reverse.
        """
        if inverse :
            # reverse scaling
            parameters = parameters / self.scale_in["scale"] - self.scale_in["sum"]
        else :
            # scale to unit square
            parameters = (parameters + self.scale_in["sum"]) * self.scale_in["scale"]

        return parameters
    
    def scale_responses(self, responses : np.ndarray, inverse : bool = False) -> np.ndarray :
        """
        Scale the response space to unit square and reverse.
        """
        if inverse :
            # reverse scaling
            responses = responses /self.scale_out["scale"] - self.scale_out["sum"]
        else :
            # scale to unit square
            responses = (responses + self.scale_out["sum"]) * self.scale_out["scale"]

        return responses


    def predict(self, 
                prediction_pts : np.ndarray,
                tols : np.ndarray,
                verbose : bool = False
                ) -> np.ndarray :
        n_pts = len(prediction_pts)
        tolerance = tols

        material_parameters = self.scale_parameters(prediction_pts, inverse = True)

        E = material_parameters[:, 0] # Young's modulus
        nu = material_parameters[:, 1] # Poisson's ratio

        data_path = self.data_path
        flags = f""

        if verbose :
            flags += f" --verbose true "

        if self.adaptive :
            flags += f"--save_mesh true "

            meshfiles = []

            for i in range(n_pts) :
                curr_dir = data_path + f"/eval/par{i}"
                if os.path.exists(curr_dir + "/mesh.vtu") :
                    # use the mesh from the previous run
                    meshfiles.append(curr_dir + "/mesh.vtu")
                else :
                    # create the directory
                    if not os.path.exists(curr_dir) :
                        os.makedirs(curr_dir)
                    # use the default mesh
                    meshfiles.append(data_path + "/default_cuboid.vtu")
        else :
            # use default tolerance
            tolerance = np.full(n_pts, self.default_tol)

            flags += f"--max_refinements 0 "

            meshfiles = [ data_path + f"/non_adaptive{self.default_tol}.vtu" for i in range(n_pts) ]

        if not self.mesh is None :
            meshfiles = [self.mesh for i in range(n_pts)]
        
        responses = np.zeros((n_pts, self.dout))
        error_levels = np.zeros(n_pts)
        self.residuals = []
        self.tolerances = []

        #loop over evaluation points
        for i in range(n_pts) :
            # set the material parameters
            runflags = flags + f" --E {E[i]} --nu {nu[i]} --tolx {tolerance[i]} "
            #data paths
            if self.adaptive :
                out_path = data_path + f"/eval/par{i}"
            else :
                out_path = data_path + "/eval"

            if not os.path.exists(out_path) :
                os.makedirs(out_path)

            runflags += f"--datapath {out_path} --mesh {meshfiles[i]} "

            responses[i], error_levels[i] = self.run(runflags, out_path, tolerance[i])

            
        return responses, error_levels

        
    def run(self, flags: str, out_path: str, default_tol: float) -> tuple:
        """
        Run the adaptive beam model.
        """
        # run the adaptive beam model
        cmd = f"{self.model_path}/adaBeam {flags}"
        
        subprocess.run(cmd, shell=True)

        # read the output
        try :
            with open(out_path + "/measurements.csv", "r") as f :
                reader = csv.reader(f)
                data = np.array(list(reader)).astype(float)
            measurements = data[0]
            measurements = self.scale_responses(measurements)

        except FileNotFoundError :
            print(f"Model evaluation did not succeed")
            print("flags : " + flags)
            raise RuntimeError("FE Model error")
        
        try :
            with open(out_path + "/residuals.csv", "r") as f :
                reader = csv.reader(f)
                
                residuals = list(reader)
            residuals = np.array(residuals).astype(float)
            residuals = residuals * self.scale_out["scale"] 

            with open(out_path + "/tolerances.csv", "r") as f :
                reader = csv.reader(f)

                data = list(reader)
                
            tolerance_levels =  np.array(data[0]).astype(float)

            error_level = tolerance_levels[-1]

            self.residuals.append(residuals.tolist())
            self.tolerances.append(tolerance_levels)
        except FileNotFoundError :
            print("Residuals file not found")
            error_level = default_tol

        return measurements, error_level
    
    def get_residuals(self):
        """
        Get the residuals from the last run.
        """
        return self.residuals, self.tolerances
                
