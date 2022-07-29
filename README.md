# gaussian-process-odes
Implementation of the work [Variational multiple shooting for Bayesian ODEs with Gaussian processes](https://arxiv.org/abs/2106.10905)


Installation
--------

This codebase utilizes `poetry` for dependency management. Instructions for `poetry` installation can be found [here](https://python-poetry.org/docs/master/#installation).
Python and package versions can be found in `pyproject.toml`. Once `poetry` is installed, necessary python dependencies can be installed by running the following command in the terminal: 

    > poetry install
Make sure to activate the virtual environment by running following command from the project root. 

    > source .venv/bin/activate 

Model specification and experiments 
------------
We propose two variants:
![](/assets/vdp_illustration.png)
1. GPODE - a Bayesian model to learn posteriors over unknown ODEs with Gaussian processes. A notebook illustrating the proposed model can be found at `notebooks/Learning-VDP-with-GPODE.ipynb`. The model can be trained on the VDP system and MoCap datasets as below:

         > python train_vdp_gpode.py
         > python train_mocap_gpode.py

![](/assets/shooting_illustration.png)
2. Shooting variant of GPODE - which splits the full ODE time integration into multiple short trajectories, hence beneficial while training on long training trajectories. Scripts for training the shooting variant of GPODE on the VDP and MoCap datasets are as below:

         > python train_vdp_gpode_shooting.py
         > python train_mocap_gpode_shooting.py

The hyperparameter configuration used for the experiments in the manuscript are specified as default values in the training scripts.
These scripts generate optimization traces, model diagnostics plots, and predictive performance scores in the `results/` directory. 

MoCap sequences
------------
Below video shows GPODE predictions for running sequences (subject 09) in the MoCap dataset.
![](/assets/mocap.gif)