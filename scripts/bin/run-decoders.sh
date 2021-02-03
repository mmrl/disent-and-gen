# !/bin/bash

# Full extrapolation
# python -m experiments.decoders with dataset.dsprites \
#                                     dataset.condition=extrp \
#                                     dataset.variant=blank_side \
#                                     model.higgins

# python -m experiments.decoders with dataset.blank_side_50 \
#                                     model.burgess training.recons_nll

# python -m experiments.decoders with dataset.blank_side_50 \
#                                     model.deep training.recons_nll


# # Dimension-wise compositional interpolation (translation)
# python -m experiments.decoders with dataset.dsprites \
#                                     dataset.condition=recomb2range \
#                                     dataset.variant=shape2tx \
#                                     model.higgins

# python -m experiments.decoders with dataset.leave1out_trans \
#                                     model.burgess training.recons_nll

# python -m experiments.decoders with dataset.leave1out_trans \
#                                     model.deep training.recons_nll


# # Dimension-wise compositional interpolation (scale)
# python -m experiments.decoders with dataset.leave1out_scale \
#                                     model.higgins training.recons_nll

# python -m experiments.decoders with dataset.leave1out_scale \
#                                     model.burgess training.recons_nll

# python -m experiments.decoders with dataset.leave1out_scale \
#                                     model.deep training.recons_nll


# # Range-wise compositional interpolation
# python -m experiments.decoders with dataset.leave1out_comb \
#                                     model.higgins training.recons_nll

# python -m experiments.decoders with dataset.leave1out_comb \
#                                     model.burgess training.recons_nll

# python -m experiments.decoders with dataset.dsprites \
#                                     dataset.condition=recomb2element \
#                                     dataset.variant=leave1out \
#                                     model.higgins


# SHAPES 3D

# Extrapolation
# python -m experiments.decoders -n dec-shapes3d with dataset.shapes3d dataset.condition=extrp \
#                                     dataset.variant=floor_hue_50 model.higgins

# python -m experiments.decoders -n dec-shapes3d with dataset.shapes3d dataset.condition=extrp \
#                                     dataset.variant=floor_hue_50 model.kim_bn

# python -m experiments.decoders -n dec-shapes3d with dataset.shapes3d dataset.condition=extrp \
#                                     dataset.variant=floor_hue_50 model.deep_bn

# # Recomb to range (object and wall hues to floor hue)
# python -m experiments.decoders -n dec-shapes3d with dataset.shapes3d dataset.condition=recomb2range \
#                                     dataset.variant=objh_wall_to_floor model.higgins

# python -m experiments.decoders -n dec-shapes3d with dataset.shapes3d dataset.condition=recomb2range \
#                                     dataset.variant=objh_wall_to_floor model.kim_bn

# python -m experiments.decoders -n dec-shapes3d with dataset.shapes3d dataset.condition=recomb2range \
#                                     dataset.variant=objh_wall_to_floor model.deep_bn


# # Recomb to range (shape to floor hue)
# python -m experiments.decoders -n dec-shapes3d with dataset.shapes3d dataset.condition=recomb2range \
#                                     dataset.variant=shape_to_floor model.higgins

# python -m experiments.decoders -n dec-shapes3d with dataset.shapes3d dataset.condition=recomb2range \
#                                     dataset.variant=shape_to_floor model.kim_bn

# python -m experiments.decoders -n dec-shapes3d with dataset.shapes3d dataset.condition=recomb2range \
#                                     dataset.variant=shape_to_floor model.deep_bn


# # Recomb to range (shape to object hue)
# python -m experiments.decoders -n dec-shapes3d with dataset.shapes3d dataset.condition=recomb2range \
#                                     dataset.variant=shape_to_objh model.higgins

# python -m experiments.decoders -n dec-shapes3d with dataset.shapes3d dataset.condition=recomb2range \
#                                     dataset.variant=shape_to_objh model.kim_bn

# python -m experiments.decoders -n dec-shapes3d with dataset.shapes3d dataset.condition=recomb2range \
#                                     dataset.variant=shape_to_objh model.deep_bn


# # # Recomb to element
# python -m experiments.decoders -n dec-shapes3d with dataset.shapes3d dataset.condition=recomb2element \
#                                     dataset.variant=leave1out model.higgins

# python -m experiments.decoders -n dec-shapes3d with dataset.shapes3d dataset.condition=recomb2element \
#                                     dataset.variant=leave1out model.kim_bn

# python -m experiments.decoders -n dec-shapes3d with dataset.shapes3d dataset.condition=recomb2element \
#                                     dataset.variant=leave1out model.deep_bn


# MPI decoders
MIP_EXTRP_VARIANTS='horz_gt20 objc_gt3'
MPI_R2R_VARIANTS='cyl2horz objc2horz'
MODELS='deep_bn'

for MODELNAME in $MODELS
do
    for VARIANT in $MIP_EXTRP_VARIANTS
    do
        python -m experiments.decoders -n dec-mip with dataset.mpi3d \
                                    dataset.condition=extrp \
                                    dataset.variant=$VARIANT model.$MODELNAME
    done
done

for MODELNAME in $MODELS
do
    for VARIANT in $MPI_R2R_VARIANTS
    do
        python -m experiments.decoders -n dec-mip with dataset.mpi3d \
                                    dataset.condition=recomb2range \
                                    dataset.variant=$VARIANT model.$MODELNAME
    done
done

# MODELS='higgins kim_bn deep_bn'

for MODELNAME in $MODELS
do
    python -m experiments.decoders -n dec-mip with dataset.mpi3d \
                                dataset.condition=recomb2element \
                                dataset.variant=leave1out model.$MODELNAME
done
