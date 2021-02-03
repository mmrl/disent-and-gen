"""
Data-splitting functions for each dataset.

These are the functions that exlcude combiantions from the datasets
in order to test different generalisation setttings. The splits are
organized in classes so they create different namespaces.

The general mechanism works by passing a condition and variant
parameter to the appropriate class. The splits are returned as index
values. That way the images and targets (when predicting a latent)
can be split in one call.

Each dataset contains a description of the generative factor names
and their values for quick referencing when adding more splits.
"""


import numpy as np
import abc


class DataSplit(abc.ABC):
    @classmethod
    def get_splits(cls, condition, variant):
        pass


class Shapes3D(DataSplit):
    """
    Boolean filters used to partition the Shapes3D dataset
    for each generalisation condition

    #=============================================================
    # Latent Dimension, Latent values
    #=============================================================
    # floor hue:        10 values linearly spaced in [0, 1]
    # wall hue:         10 values linearly spaced in [0, 1]
    # object hue:       10 values linearly spaced in [0, 1]
    # scale:            8 values linearly spaced in [0.75, 1.25]
    # shape:            4 values in [0, 1, 2, 3]
    # orientation:      15 values linearly spaced in [-30, 30]

    """
    fh, wh, oh, scl, shp, rot = 0, 1, 2, 3, 4, 5

    @classmethod
    def get_splits(cls, condition, variant):
        if condition == 'extrp':
            if variant == 'fhue_gt50':
                return cls.missing_fh_50()
        elif condition == 'recomb2range':
            if variant == 'ohue+whue2fhue':
                return cls.objh_and_wallh_to_floorh()
            elif variant == 'shape2ohue':
                return cls.shape_to_objh()
            elif variant == 'shape2fhue':
                return cls.shape_to_floor()
        elif condition == 'recomb2element':
            if variant == 'leave1out':
                return cls.leave1out()
        else:
            raise ValueError(
                'Unrecognized condition {} or variant {}'.format(condition, variant))

    # Extrapolation variants
    @classmethod
    def missing_fh_50(cls):
        def train_filter(latent_values, latent_classes):
            return latent_values[:, cls.fh] < 0.5

        def test_filter(latent_values, latent_classes):
            return ~train_filter(latent_values, latent_classes)

        return train_filter, test_filter

    # Recombination to range
    @classmethod
    def objh_and_wallh_to_floorh(cls):
        def test_filter(latent_values, latent_classes):
            return ((latent_values[:, cls.oh] >= 0.8) &
                    (latent_values[:, cls.wh] <= 0.2))

        def train_filter(latent_values, latent_classes):
            return ~test_filter(latent_values, latent_classes)

        return train_filter, test_filter

    @classmethod
    def shape_to_floor(cls):
        def test_filter(latent_values, latent_classes):
            return ((latent_values[:, cls.shp] == 3.0) &
                    (latent_values[:, cls.fh] >= 0.5))

        def train_filter(latent_values, latent_classes):
            return ~test_filter(latent_values, latent_classes)

        return train_filter, test_filter

    @classmethod
    def shape_to_objh(cls):
        def test_filter(latent_values, latent_classes):
            return ((latent_values[:, cls.shp] == 3.0) &
                    (latent_values[:, cls.oh] >= 0.5))

        def train_filter(latent_values, latent_classes):
            return ~test_filter(latent_values, latent_classes)

        return train_filter, test_filter

    # Recombination to element
    @classmethod
    def leave1out(cls):
        def test_filter(latent_values, latent_classes):
            return ((latent_values[:, cls.oh] >= 0.8) &
                    (latent_values[:, cls.wh] >= 0.8) &
                    (latent_values[:, cls.fh] >= 0.8) &
                    (latent_values[:, cls.scl] >= 1.1) &
                    (latent_values[:, cls.shp] == 1) &
                    (latent_values[:, cls.rot] > 20))

        def train_filter(latent_values, latent_classes):
            return ~test_filter(latent_values, latent_classes)

        return train_filter, test_filter



class Dsprites(DataSplit):
    """
    Boolean filters used to partition the Dsprites dataset
    for each generalisation condition

    #=============================================================
    # Latent Dimension, Latent values
    #=============================================================
    # Luminence       - 255
    # Shape           - Square, ellipse, heart
    # Scale           - [0.5, 1] split into 6 values
    # Angle           - [0, 2pi] split into 40 values
    # Translation X   - [0, 1] split into 32 values
    # Translation Y   - [0, 1] split into 32 values
    """
    shp, scl, rot, tx, ty = 0, 1, 2, 3, 4
    a90, a120, a180, a240 = np.pi / 2, 4 * np.pi / 3, np.pi, 2 * np.pi / 3

    @classmethod
    def get_splits(cls, condition, variant):
        if condition == 'extrp':
            if variant == 'blank_side':
                return cls.blank_side()
        elif condition == 'recomb2range':
            if variant == 'shape2tx':
                return cls.shape2tx()
        elif condition == 'recomb2element':
            if variant == 'leave1out':
                return cls.leave1out()
        else:
            raise ValueError(
                'Unrecognized condition {} or variant {}'.format(condition, variant))


    # Filters for blank right side condition
    @classmethod
    def blank_side(cls, ratio=0.5):
        def blank_side_train(latent_values, latent_classes):
            return (latent_values[:, cls.tx] < ratio)

        def blank_side_extrp(latent_values, latent_classes):
            return (latent_values[:, cls.tx] > ratio)

        return blank_side_train, blank_side_extrp

    # Leave one shape out along translation dimension
    @classmethod
    def shape2tx(cls):
        def shape2tx_train(latent_values, latent_classes):
            return ((latent_values[:, cls.shp] != 1) | (latent_values[:, cls.tx] < 0.5))

        def shape2tx_extrp(latent_values, latent_classes):
            return ((latent_values[:, cls.shp] == 1) &
                    (latent_values[:, cls.tx] > 0.5))

        return shape2tx_train, shape2tx_extrp

    # leave1out_comb
    @classmethod
    def leave1out(cls):
        def leave1out_comb_test(latent_values, latent_classes):
            return ((latent_classes[:, cls.shp] == 1) &
                    (latent_values[:, cls.scl] > 0.6) &
                    (latent_values[:, cls.rot] > 0.0) &
                    (latent_values[:, cls.rot] < cls.a120) &
                    (latent_values[:, cls.rot] > cls.a240) &
                    (latent_values[:, cls.tx] > 0.66) &
                    (latent_values[:, cls.ty] > 0.66))

        def leave1out_comb_train(latent_values, latent_classes):
            return ~leave1out_comb_test(latent_values, latent_classes)

        return leave1out_comb_train, leave1out_comb_test


##################################### MPI3D ###################################

def fix_size_and_camh(variant):
    def single_cam_and_size(latent_values):
        return ((latent_values[:, MPI3D.camh] == 1) &
                (latent_values[:, MPI3D.sz] == 1))

    def reduced(cls):
        mask1, mask2 = variant(cls)

        def train_mask(latent_values):
            return (mask1(latent_values) &
                    single_cam_and_size(latent_values))

        def test_mask(latent_values):
            return (mask2(latent_values) &
                    single_cam_and_size(latent_values))

        return train_mask, test_mask

    return reduced

class MPI3D(DataSplit):
    """
    # Boolean filters used to partition the MPI datasets
    # for each generalisation condition

    #===========================================================================================
    # Latent Dimension,    Latent values                                                 N vals
    #===========================================================================================
    # object color:        white=0, green=1, red=2, blue=3, brown=4, olive=5                6
    # object shape:        cone=0, cube=1, cylinder=2, hexagonal=3, pyramid=4, sphere=5     6
    # object size:         small=0, large=1                                                 2
    # camera height:       top=0, center=1, bottom=2                                        3
    # background color:    purple=0, sea green=1, salmon=2                                  3
    # horizontal axis:     40 values liearly spaced [0, 39]                                40
    # vertical axis:       40 values liearly spaced [0, 39]                                40
    """
    oc, shp, sz, camh, bkg, hx, vx = 0, 1, 2, 3, 4, 5, 6

    @classmethod
    def get_splits(cls, condition, variant):
        if condition == 'extrp':
            if variant == 'horz_gt20':
                return cls.exclude_horz_gt20()
            elif variant == 'objc_gt3':
                return cls.exclude_objc_gt3()
        elif condition == 'recomb2range':
            if variant == 'cyl2horz':
                return cls.cylinder2horz20()
            if variant == 'objc2horz':
                return cls.redobj2horz20()
        elif condition == 'recomb2element':
            if variant == 'leave1out':
                return cls.leave1out()
        else:
            raise ValueError(
                'Unrecognized condition {} or variant {}'.format(condition, variant))

    # Extrapolation
    @classmethod
    # @fix_size_and_camh
    def exclude_horz_gt20(cls):
        def train_mask(latent_values):
            return latent_values[:, cls.hx] < 20

        def test_mask(latent_values):
            return (latent_values[:, cls.hx] > 20)

        return train_mask, test_mask


    @classmethod
    # @fix_size_and_camh
    def exclude_objc_gt3(cls):
        def train_mask(latent_values):
            return latent_values[:, cls.oc] <= 3

        def test_mask(latent_values):
            return latent_values[:, cls.oc] > 3

        return train_mask, test_mask


    # Recombination to element
    @classmethod
    # @fix_size_and_camh
    def cylinder2horz20(cls):
        def test_mask(latent_values):
            return ((latent_values[:, cls.hx] < 20) &
                    (latent_values[:, cls.shp] == 2))
        def training_mask(latent_values):
            return ~test_mask(latent_values)

        return training_mask, test_mask


    @classmethod
    # @fix_size_and_camh
    def redobj2horz20(cls):
        def test_mask(latent_values):
            return ((latent_values[:, cls.hx] < 20) &
                    (latent_values[:, cls.oc] == 2))
        def training_mask(latent_values):
            return ~test_mask(latent_values)

        return training_mask, test_mask


    # Recombination to element
    @classmethod
    # @fix_size_and_camh
    def leave1out(cls):
        def test_mask(latent_values):
            return ((latent_values[:, cls.oc] == 5) &
                    (latent_values[:, cls.shp] == 2) &
                    (latent_values[:, cls.sz] == 1) &
                    (latent_values[:, cls.camh] == 1) &
                    (latent_values[:, cls.bkg] == 1) &
                    (latent_values[:, cls.hx] > 35) &
                    (latent_values[:, cls.vx] > 35))

        def train_mask(latent_values):
            return ~test_mask(latent_values)

        return train_mask, test_mask
