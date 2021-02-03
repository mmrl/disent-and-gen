# !/bin/bash

# python -m experiments.composition with model.higgins

# python -m experiments.composition with model.burgess_v2

# python -m experiments.composition with model.burgess_v2 training.loss.params.beta=4

# python -m experiments.composition with model.burgess_v2 training.loss.params.beta=8


# python -m experiments.composition with model.higgins

# python -m experiments.composition with dataset.leave1out_comb model.burgess_v2

# python -m experiments.composition with model.burgess_v2 training.loss.params.beta=4

# python -m experiments.composition with model.burgess_v2 training.loss.params.beta=8


python -m experiments.composition with dataset.leave1out_trans model.burgess_v2

python -m experiments.composition with dataset.rotation_extrp model.burgess_v2