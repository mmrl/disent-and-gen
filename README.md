# The Role of Disentanglement in Generalisation

[ICLR version](https://openreview.net/forum?id=qbH974jKUVy)

**Authors**: [Milton Llera Montero](https://github.com/miltonllera), [Casimir Ludwig](https://sites.google.com/site/casimirludwig/), [Rui Ponte Costa](https://neuralml.github.io/), [Gaurav Malhotra](https://research-information.bris.ac.uk/en/persons/gaurav-malhotra) and [Jeff Bowers](https://jeffbowers.blogs.bristol.ac.uk/).

**Abstract**: Combinatorial generalisation — the ability to understand and produce novel combinations of familiar elements — is a core capacity of human intelligence that current AI systems struggle with. Recently, it has been suggested that learning disentangled representations may help address this problem. It is claimed that such representations should be able to capture the compositional structure of the world which can then be combined to support combinatorial generalisation. In this study, we systematically tested how the degree of disentanglement affects various forms of generalisation, including two forms of combinatorial generalisation that varied in difficulty. We trained three classes of variational autoencoders (VAEs) on two datasets on an unsupervised task by excluding combinations of generative factors during training. At test time we ask the models to reconstruct the missing combinations in order to measure generalisation performance. Irrespective of the degree of disentanglement, we found that the models supported only weak combinatorial generalisation. We obtained the same outcome when we directly input perfectly disentangled representations as the latents, and when we tested a model on a more complex task that explicitly required independent generative factors to be controlled. While learning disentangled representations does improve interpretability and sample efficiency in some downstream tasks, our results suggest that they are not sufficient for supporting more difficult forms of generalisation.

---

This repo contains the code necessary to run the experiments for the article. The code was tested on Python 3.8 and PyTorch 1.7.

Currently the repo includes:

1. Three models: [Latent Gaussian Model](http://proceedings.mlr.press/v32/rezende14.html), the Ground-Truth Decoder that they are compared to, and a Composer that uses them for the composition task.
2. Four losses to train them: 1) [VAE](https://arxiv.org/abs/1312.6114), 2) [$\beta$-VAE](https://openreview.net/pdf?id=Sy2fzU9gl), 3) [CCI-VAE](https://arxiv.org/pdf/1804.03599.pdf%20) and 4) [FactorVAE](https://arxiv.org/pdf/1802.05983.pdf)
3. Three datasets to test the models on: 1) [dSprites](https://github.com/deepmind/dsprites-dataset), 2) [3DShapes](https://github.com/deepmind/3d-shapes), 3) [MPI3D](https://github.com/rr-learning/disentanglement_dataset) 

## Setting up the Conda environment

Running these experiments requires (among others) the following libraries installed:

* [PyTorch and Torchvision](https://pytorch.org/): Basic framework for Deep Learning models and training.
* [Ignite](https://github.com/pytorch/ignite): High-level framework to train models, eliminating the need for much boilerplate code.
* [Sacred](https://github.com/IDSIA/sacred): Libary used to define and run experiments in a systematic way.
* [Matplotlib](https://matplotlib.org/): For plotting.
* [Jupyter](https://jupyter.org/): To produce the plots.

We recommend using the provided [environment configuration file]() and intalling using:

```
conda env create -f torchlab-env.yml
```

## Directory structure

The repository is organized as follows:

```
data/
├── raw/
    ├── dsprites/
    ├── shapes3d/
    ├── mpi/
	├── mpi3d_real.npz 
├── sims/
    ├── disent/  # Runs will be added here, Sacred will asign names as integers of increasing value
    ├── composition/
    ├── decoders/
notebooks/
├── postproc.ipnyb  # Notebooks used to produce the plots/analyze the data.
├── transformers.ipynb
scripts/
├── configs/
    ├── cnnvae.py  # An example config file with VAE architectures.
├── ingredients/
    ├── models.py  # Example ingredient that wrapps model initalization
├── experiments/
    ├── disent.py  # Experiment script for training disentangled models
src/
├── analysis/  # These folders contain the actual datasets, losses, model classes etc.
├── dataset/
├── models/
├── training/
```

The data structure should be self explanatory for the most part. The main thing to note is that ``src`` contains code for models that are used throughout the experiments while the ingredients contain wrappers around these to initialize them from the configuration files. Simulation results will be saved in sims. The results of the analysis were stored in a new folder (``results``, not shown). We attempted to use models with the hightes disentanglement in our analysis.

Datasets should appear in a subfolder as shown above. Right now, there is not method for automatically downloading the data, but they can be found in their corresponding repos. Alternatively, altering the source file or passing the dataset root as a parameter can be used to look for the datasets in another location[^1].

The configuration folder has the different parameters combinations used in the experiments. Following these should allow someone to define new experiments easily. Just remember to add the configurations to the appropriate ingredient using ``ingredient.named_config(config_function/yaml_file)``.

## Running an experiment

To run an experiment you should execute one of the scripts from the scripts folder with the appropraite options. We use Sacred to run and track experimetns. You can check the online documentation to understand how it works. Below you is the general command used and more can be found in the ``bin`` folder.

```
cd ~/path/to/project/scripts/
python -m experiments.disent with dataset.<option> model.<option> training.<option>
```

Sacred allows passing parameters using keyword arguments. For example we can change the latent size and $\beta$ from the default values:

```
python -m experiments.disent with dataset.dsprites model.kim training.factor model.latent_size=50 training.loss.params.beta=10
```

## Attributions

This repository contains code used to analyze models which was based/copied form the ones the one found in [qedr](https://github.com/cianeastwood/qedr/tree/master/lib), and [disentangling-vae](https://github.com/YannDubs/disentangling-vae), which we also used to check our implementations. We would like to thank the authors for making them available.

## Acknowledgements

We would like to thank everyone who gave feedback on this research, especially the members of the [Mind and Machine Research Lab](https://mindandmachine.blogs.bristol.ac.uk/) and [Neural and Machine Learning Group](https://neuralml.github.io/).

This project has received funding from the European Research Council (ERC) under the European Union’s Horizon 2020 research and innovation programme (grant agreement No. 741134).


[^1]: I might add code to automatically download the datasets and create the folders, but only if I have time.
