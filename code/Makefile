SHELL := /bin/bash

.DEFAULT_GOAL := laplace_exp


env = AGP-env
OUTPATH = /data/numerik/people/pvillani/thesis

FLAGS = --path $(OUTPATH)

CONDA_ACTIVATE = source $$(conda info --base)/etc/profile.d/conda.sh ; conda activate ; conda activate $(env)
PYPATH = export PYTHONPATH=$(CURDIR)

CONDA_PY_SET = $(CONDA_ACTIVATE); $(PYPATH);


laplace_config: experiments/3d-laplace/generate_configuration.py
		$(CONDA_PY_SET) python experiments/3d-laplace/generate_configuration.py $(FLAGS)

laplace_GP: laplace_config
		$(CONDA_ACTIVATE); $(PYPATH); for i in $$(seq 1 26) ; do \
		python experiments/run_AGP.py $(FLAGS) --dim 3; \
		python experiments/run_LHSGP.py $(FLAGS) --dim 3; \
		python experiments/run_posAdGP.py $(FLAGS) --dim 3; \
		python experiments/run_randGP.py $(FLAGS) --dim 3; \
		done
		
laplace_LR : laplace_config
		$(CONDA_ACTIVATE); $(PYPATH); for i in $$(seq 1 25) ; do \
		python experiments/run_ALR.py $(FLAGS) --dim 3; \
		python experiments/run_LHSLR.py $(FLAGS) --dim 3; \
		python experiments/run_posAdLR.py $(FLAGS) --dim 3; \
		python experiments/run_randLR.py $(FLAGS) --dim 3; \
		done

missing_laplace : laplace_config
		$(CONDA_ACTIVATE); $(PYPATH); for i in $$(seq 1 25) ; do \
		python experiments/run_ALR.py $(FLAGS); \
		done