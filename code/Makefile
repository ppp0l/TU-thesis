SHELL := /bin/bash

.DEFAULT_GOAL := laplace_exp


env = AGP-env
OUTPATH = /data/numerik/people/pvillani/thesis
DIM = 2

FLAGS = --path $(OUTPATH)

CONDA_ACTIVATE = source $$(conda info --base)/etc/profile.d/conda.sh ; conda activate ; conda activate $(env)
PYPATH = export PYTHONPATH=$(CURDIR)

CONDA_PY_SET = $(CONDA_ACTIVATE); $(PYPATH);


laplace_config: experiments/3d-laplace/generate_configuration.py
		$(CONDA_PY_SET) python experiments/3d-laplace/generate_configuration.py $(FLAGS)

laplace_exp: laplace_config
		$(CONDA_ACTIVATE); $(PYPATH); for i in $$(seq 1 26) ; do \
		python experiments/3d-laplace/run_AGP.py $(FLAGS); \
		python experiments/3d-laplace/LHSGP.py $(FLAGS); \
		python experiments/3d-laplace/run_posAdGP.py $(FLAGS); \
		python experiments/3d-laplace/run_randGP.py $(FLAGS); \
		done
