# !/bin/bash

# #======================== Dsprites ===================================================

# # Extrapolation
# python -m experiments.disent -n full-extrp with dataset.blank_side model.burgess_v2

# python -m experiments.disent -n full-extrp with dataset.blank_side model.burgess_v2 \
#                                                 training.loss.params.beta=8

# python -m experiments.disent -n full-extrp with dataset.blank_side model.higgins \
#                                                 training.loss.params.beta=8

# python -m experiments.disent -n full-extrp with dataset.blank_side model.burgess_v2 \
#                                                 training.loss.params.beta=12


# # Generalization by combinatorial interpolation along the translation dimension

# python -m experiments.disent -n dimcomp-tx with dataset.leave1out_trans model.burgess_v2

# python -m experiments.disent -n dimcomp-tx with dataset.leave1out_trans model.higgins \
#                                                 training.loss.params.beta=8

# python -m experiments.disent -n dimcomp-tx with dataset.leave1out_trans model.burgess_v2 \
#                                                 training.loss.params.beta=8 seed=716011605

# python -m experiments.disent -n dimcomp-tx with dataset.dsprites dataset.condition=recomb2range \
#                                                 dataset.variant=shape2tx model.burgess_v2 \
#                                                 training.loss.params.beta=12 seed=145395029

# # Combinatorial interpolation along the rotation dimension

# python -m experiments.disent -n dimcomp-rot with dataset.rotation_extrp model.burgess_v2

# python -m experiments.disent -n dimcomp-rot with dataset.rotation_extrp model.burgess_v2 \
#                                              training.loss.params.beta=8

# python -m experiments.disent -n dimcomp-rot with dataset.rotation_extrp model.higgins \
#                                              training.loss.params.beta=8

# python -m experiments.disent -n dimcomp-rot with dataset.rotation_extrp model.burgess_v2 \
#                                              training.loss.params.beta=12

# # Combinatorial interpolation along the scale dimension

# python -m experiments.disent -n dimcomp-scl with dataset.leave1out_scale model.higgins

# python -m experiments.disent -n dimcomp-scl with dataset.leave1out_scale model.burgess_v2

# python -m experiments.disent -n dimcomp-scl with dataset.leave1out_scale model.higgins \
#                                    training.loss.params.beta=8

# python -m experiments.disent -n dimcomp-scl with dataset.leave1out_scale model.burgess_v2 \
#                                   training.loss.params.beta=8

# python -m experiments.disent -n dimcomp-scl with dataset.leave1out_scale model.burgess_v2 \
#                                   training.loss.params.beta=12


# Local interpolation, hard
# python -m experiments.disent -n localinterp with dataset.leave1out_comb model.higgins

# python -m experiments.disent -n localinterp with dataset.leave1out_comb model.burgess_v2

# python -m experiments.disent -n localinterp with dataset.leave1out_comb model.burgess_v2 \
#                                    training.loss.params.beta=8

# python -m experiments.disent -n localinterp with dataset.leave1out_comb model.burgess_v2 \
#                                   training.loss.params.beta=8

# python -m experiments.disent -n localinterp with dataset.leave1out_comb model.burgess_v2 \
#                                    training.loss.params.beta=12

## Using FactorVAE

# python -m experiments.disent with dataset.dsprites dataset.condition=extrp \
#                                   dataset.variant=blank_side \
#                                   model.kim training.factor \
#                                   training.loss.params.gamma=50

# python -m experiments.disent with dataset.dsprites dataset.condition=recomb2range \
#                                   dataset.variant=shape2tx \
#                                   model.kim training.factor \
#                                   training.loss.params.gamma=50

# python -m experiments.disent with dataset.dsprites dataset.condition=recomb2element \
#                                   dataset.variant=leave1out \
#                                   model.kim training.factor \
#                                   training.loss.params.gamma=50


# python -m experiments.disent with dataset.dsprites dataset.condition=extrp \
#                                   dataset.variant=blank_side \
#                                   model.kim training.factor \
#                                   training.loss.params.gamma=20

# python -m experiments.disent with dataset.dsprites dataset.condition=recomb2range \
#                                   dataset.variant=shape2tx \
#                                   model.kim training.factor \
#                                   training.loss.params.gamma=20

# python -m experiments.disent with dataset.dsprites dataset.condition=recomb2element \
#                                   dataset.variant=leave1out \
#                                   model.kim training.factor \
#                                   training.loss.params.gamma=20


# python -m experiments.disent with dataset.dsprites dataset.condition=extrp \
#                                   dataset.variant=blank_side \
#                                   model.kim training.factor \
#                                   training.loss.params.gamma=100

# python -m experiments.disent with dataset.dsprites dataset.condition=recomb2range \
#                                   dataset.variant=shape2tx \
#                                   model.kim training.factor \
#                                   training.loss.params.gamma=100

# python -m experiments.disent with dataset.dsprites dataset.condition=recomb2element \
#                                   dataset.variant=leave1out \
#                                   model.kim training.factor \
#                                   training.loss.params.gamma=100

# #==================== SHAPES3D ==========================================

# MODELS='kim'
# # # BETAS='1 8 12'
# GAMMAS='20 50 100'

# # Extrapolation
# for MODELNAME in $MODELS
# do
#     for B in $BETAS
#     do
#         python -m experiments.disent -n shapes3d with dataset.shapes3d \
#                                         dataset.condition=extrp \
#                                         dataset.variant=fhue_gt50 \
#                                         model.$MODELNAME \
#                                         training.beta \
#                                         training.loss.params.beta=$B
#     done
#
#     for G in $GAMMAS
#     do
#         python -m experiments.disent -n shapes3d with dataset.shapes3d \
#                                         dataset.condition=extrp \
#                                         dataset.variant=fhue_gt50 \
#                                         model.$MODELNAME \
#                                         training.factor \
#                                         training.loss.params.gamma=$G
#     done
# done


# Recombination to range
# SHAPES_R2R_VARIANTS='shape2fhue shape2ohue'

# for MODELNAME in $MODELS
# do
#     for B in $BETAS
#     do
#         for V in $SHAPES_R2R_VARIANTS
#         do
#         python -m experiments.disent -n shapes3d with dataset.shapes3d \
#                                         dataset.condition=recomb2range \
#                                         dataset.variant=$V \
#                                         model.$MODELNAME \
#                                         training.beta \
#                                         training.loss.params.beta=$B
#         done
#     done
#
#     for G in $GAMMAS
#     do
#         for V in $SHAPES_R2R_VARIANTS
#         do
#             python -m experiments.disent -n shapes3d with dataset.shapes3d \
#                                             dataset.condition=recomb2range \
#                                             dataset.variant=$V \
#                                             model.$MODELNAME \
#                                             training.factor \
#                                             training.loss.params.gamma=$G

#         done
#     done
# done

# Recombination to element
# for MODELNAME in $MODELS
# do
#     for B in $BETAS
#     do
#         python -m experiments.disent -n shapes3d with dataset.shapes3d \
#                                         dataset.condition=recomb2element \
#                                         dataset.variant=leave1out \
#                                         model.$MODELNAME \
#                                         training.beta \
#                                         training.loss.params.beta=$B
#     done
#
#     for G in $GAMMAS
#     do
#         python -m experiments.disent -n shapes3d with dataset.shapes3d \
#                                         dataset.condition=recomb2element \
#                                         dataset.variant=leave1out \
#                                         model.$MODELNAME \
#                                         training.factor \
#                                         training.loss.params.gamma=$G
#     done
# done

#==================== MPI3D ==========================================

MODELS='kim'
BETAS='1 8 12'
GAMMAS='20 50 100'

# Extrapolation
MIP_EXTRP_VARIANTS='horz_gt20 objc_gt3'

for MODELNAME in $MODELS
do
    for B in $BETAS
    do
        for V in $MIP_EXTRP_VARIANTS
        do
        python -m experiments.disent -n mpi3d with dataset.mpi3d \
                                        dataset.condition=extrp \
                                        dataset.variant=$V \
                                        model.$MODELNAME \
                                        training.beta \
                                        training.loss.params.beta=$B
        done
    done

    for G in $GAMMAS
    do
        for V in $MIP_EXTRP_VARIANTS
        do
            python -m experiments.disent -n mpi3d with dataset.mpi3d \
                                            dataset.condition=extrp \
                                            dataset.variant=$V \
                                            model.$MODELNAME \
                                            training.factor \
                                            training.loss.params.gamma=$G
        done
    done
done


# Recombination to range
MPI_R2R_VARIANTS='cyl2horz objc2horz'

for MODELNAME in $MODELS
do
    for B in $BETAS
    do
        for V in $MPI_R2R_VARIANTS
        do
            python -m experiments.disent -n mpi3d with dataset.mpi3d \
                                            dataset.condition=recomb2range \
                                            dataset.variant=$V \
                                            model.$MODELNAME \
                                            training.beta \
                                            training.loss.params.beta=$B
        done
    done

    for G in $GAMMAS
    do
        for V in $MPI_R2R_VARIANTS
        do
            python -m experiments.disent -n mpi3d with dataset.mpi3d \
                                            dataset.condition=recomb2range \
                                            dataset.variant=$V \
                                            model.$MODELNAME \
                                            training.factor \
                                            training.loss.params.gamma=$G
        done
    done
done

# Recombination to element
for MODELNAME in $MODELS
do
    for B in $BETAS
    do
        python -m experiments.disent -n mpi3d with dataset.mpi3d \
                                        dataset.condition=recomb2element \
                                        dataset.variant=leave1out \
                                        model.$MODELNAME \
                                        training.beta \
                                        training.loss.params.beta=$B
    done

    for G in $GAMMAS
    do
        python -m experiments.disent -n mpi3d with dataset.mpi3d \
                                        dataset.condition=recomb2element \
                                        dataset.variant=leave1out \
                                        model.$MODELNAME \
                                        training.factor \
                                        training.loss.params.gamma=$G
    done
done
