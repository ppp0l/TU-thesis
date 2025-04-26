SHELL := /bin/bash


env = AGP-env
OUTPATH = /data/numerik/people/pvillani/thesis/prova1

FLAGS = --path $(OUTPATH)

CONDA_ACTIVATE = source $$(conda info --base)/etc/profile.d/conda.sh ; conda activate ; conda activate $(env)
PYPATH = export PYTHONPATH=$(CURDIR)

CONDA_PY_SET = $(CONDA_ACTIVATE); $(PYPATH);


laplace_GP_config: experiments/3d-laplace/generate_configuration_GP.py
		$(CONDA_PY_SET) python experiments/3d-laplace/generate_configuration_GP.py $(FLAGS)

laplace_LR_config: experiments/3d-laplace/generate_configuration_LR.py
		$(CONDA_PY_SET) python experiments/3d-laplace/generate_configuration_LR.py $(FLAGS)

laplace_ALR: laplace_LR_config
		$(CONDA_ACTIVATE); $(PYPATH); for i in $$(seq 1 25) ; do \
		python experiments/run_ALR.py $(FLAGS) --dim 3; \
		done


laplace_GP: laplace_GP_config
		$(CONDA_ACTIVATE); $(PYPATH); for i in $$(seq 1 25) ; do \
		python experiments/run_AGP.py $(FLAGS) --dim 3; \
		python experiments/run_LHSGP.py $(FLAGS) --dim 3; \
		python experiments/run_posAdGP.py $(FLAGS) --dim 3; \
		python experiments/run_randGP.py $(FLAGS) --dim 3; \
		done
		
laplace_LR : laplace_LR_config
		$(CONDA_ACTIVATE); $(PYPATH); for i in $$(seq 1 25) ; do \
		python experiments/run_ALR.py $(FLAGS) --dim 3; \
		python experiments/run_LHSLR.py $(FLAGS) --dim 3; \
		python experiments/run_posAdLR.py $(FLAGS) --dim 3; \
		python experiments/run_randLR.py $(FLAGS) --dim 3; \
		done

missing_laplace : laplace_LR_config laplace_GP_config
		$(CONDA_ACTIVATE); $(PYPATH); for i in $$(seq 1 25) ; do \
		python experiments/run_ALR.py $(FLAGS); \
		done


diffusion_GP_config: experiments/6d-diffusion/generate_configuration_GP.py
		$(CONDA_PY_SET) python experiments/6d-diffusion/generate_configuration_GP.py $(FLAGS)

diffusion_LR_config: experiments/6d-diffusion/generate_configuration_LR.py
		$(CONDA_PY_SET) python experiments/6d-diffusion/generate_configuration_LR.py $(FLAGS)

diffusion_GP: diffusion_GP_config
		$(CONDA_ACTIVATE); $(PYPATH); for i in $$(seq 1 25) ; do \
		python experiments/run_AGP.py $(FLAGS) --dim 6; \
		python experiments/run_LHSGP.py $(FLAGS) --dim 6; \
		python experiments/run_posAdGP.py $(FLAGS) --dim 6; \
		python experiments/run_randGP.py $(FLAGS) --dim 6; \
		done
	
diffusion_LR : diffusion_LR_config
		$(CONDA_ACTIVATE); $(PYPATH); for i in $$(seq 1 25) ; do \
		python experiments/run_ALR.py $(FLAGS) --dim 6; \
		python experiments/run_LHSLR.py $(FLAGS) --dim 6; \
		python experiments/run_posAdLR.py $(FLAGS) --dim 6; \
		python experiments/run_randLR.py $(FLAGS) --dim 6; \
		done

FE_config_GP: experiments/2d-FE/generate_configuration_GP.py
		$(CONDA_PY_SET) python experiments/2d-FE/generate_configuration_GP.py $(FLAGS)
	
FE_config_LR: experiments/2d-FE/generate_configuration_LR.py
		$(CONDA_PY_SET) python experiments/2d-FE/generate_configuration_LR.py $(FLAGS)

FE_GP: FE_config_GP
		$(CONDA_ACTIVATE); $(PYPATH); for i in $$(seq 1 25) ; do \
		python experiments/run_AGP.py $(FLAGS) --dim 2; \
		python experiments/run_LHSGP.py $(FLAGS) --dim 2; \
		python experiments/run_posAdGP.py $(FLAGS) --dim 2; \
		python experiments/run_randGP.py $(FLAGS) --dim 2; \
		done

FE_LR: FE_config_LR
		$(CONDA_ACTIVATE); $(PYPATH); for i in $$(seq 1 25) ; do \
		python experiments/run_ALR.py $(FLAGS) --dim 2; \
		python experiments/run_LHSLR.py $(FLAGS) --dim 2; \
		python experiments/run_posAdLR.py $(FLAGS) --dim 2; \
		python experiments/run_randLR.py $(FLAGS) --dim 2; \
		done

