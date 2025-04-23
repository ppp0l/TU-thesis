import csv 
import numpy as np

import subprocess
import os

model_path = "/home/numerik/pvillani/Project/lipschitz/code/kaskade/beam"

class Adaptive_beam :
    """
    Python interface for the adaptive FE model of the beam.
    """
    dout = 5

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
        self.model_path = model_path

        self.mesh = mesh


    def predict(self, 
                material_parameters : np.ndarray,
                tolerance : np.ndarray,
                ) -> np.ndarray :
        n_pts = len(material_parameters)

        E = material_parameters[:, 0] # Young's modulus
        nu = material_parameters[:, 1] # Poisson's ratio

        # TODO : scale E and nu to the range of the model

        data_path = self.data_path
        flags = f""

        if self.adaptive :
            flags += f"--save_mesh true"

            meshfiles = []

            for i in range(n_pts) :
                curr_dir = data_path + f"/par{i}"
                if os.path.exists(curr_dir) :
                    # use the mesh from the previous run
                    meshfiles.append(curr_dir + "/mesh.vtu")
                else :
                    # create the directory
                    os.makedirs(curr_dir)
                    # use the default mesh
                    meshfiles.append(data_path + "/default_cuboid.vtu")
                    

        else :
            # use default tolerance
            tolerance = np.full(n_pts, self.default_tol)

            flags += f"--max_refinements 0"

            meshfiles = [ data_path + "/non_adaptive.vtu" for i in range(n_pts) ]

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
            out_path = data_path + f"/par{i}"
            runflags += f"--datapath {out_path} --mesh {meshfiles[i]}"

            if not os.path.exists(out_path) :
                os.makedirs(out_path)

            responses[i], error_levels[i] = self.run(runflags, out_path)
            
        return responses, error_levels

        
    def run(self, flags: str, out_path: str):
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
                measurements = data[0][:-1]
                error_level = data[0][-1]

                #TODO : scale
        except FileNotFoundError :
            print(f"Model evaluation did not succeed")
            print("flags : " + flags)
            raise RuntimeError("FE Model error")
        
        try :
            with open(out_path + "/residuals.csv", "r") as f :
                reader = csv.reader(f)
                
                residuals = list(reader)
                # TODO : scale

                self.residuals.append(residuals)

            with open(out_path + "/tolerances.csv", "r") as f :
                reader = csv.reader(f)
                
                tolerance_levels = list(reader)
                # TODO : scale
                self.tolerances.append(tolerance_levels)
        except FileNotFoundError :
            print("Residuals or tolerances file not found")

        return measurements, error_level
    
    def get_residuals(self):
        """
        Get the residuals from the last run.
        """
        return self.residuals, self.tolerances
                
