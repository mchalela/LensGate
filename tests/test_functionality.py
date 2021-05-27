
import numpy as np

import pytest

from lensgate import RadialSymmetryLens

from astropy.cosmology import Planck18


# ============================================================================
# TESTS
# ============================================================================

def test_required_args():
    """Check z and cosmo stored values.
    """

    class MyModel(RadialSymmetryLens):

        def __init__(self, z, cosmo):

            # required parameters
            self.z = z
            self.cosmo = cosmo

        def density(self):
            pass

    model = MyModel(0.2, Planck18)

    assert 0.2 == model.z
    assert Planck18 == model.cosmo


def test_missing_required_args():
    """Check z and cosmo are initialized with these exact names.
    """    

    class MyModel(RadialSymmetryLens):

        def __init__(self, z, cosmo):

            # required parameters
            self.z = z
            self.kosmo = cosmo  # other than cosmo

        def density(self):
            pass

    with pytest.raises(AttributeError):
        model = MyModel(0.2, Planck18)


def test_missing_density():
    """Check if density() method was defined.
    """

    class MyModel(RadialSymmetryLens):

        def __init__(self, z, cosmo):

            # required parameters
            self.z = z
            self.cosmo = cosmo

    with pytest.raises(TypeError):
        model = MyModel(0.2, Planck18)

