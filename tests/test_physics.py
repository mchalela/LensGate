import numpy as np

import pytest

from lensgate import RadialSymmetryLens

from astropy.cosmology import Planck18


# ============================================================================
# TEST NFW CONSISTENCY
# ============================================================================

@pytest.mark.skip
def test_nfw():
    """Check if lensing integrals correspond to the analytic expresions.
    """

    class NFW(RadialSymmetryLens):

        def __init__(self, z, cosmo):

            # required parameters
            self.z = z
            self.cosmo = cosmo

        def delta_c(self, delta, c200):
            cplus = 1 + c200
            dc = (delta/3.) * (c200**3 / (np.log(cplus) - c200/cplus))
            return dc

        def density(self, r, r200, c200):
            dc = self.delta_c(200)
            rs = r200/c200
            x = r/rs 
            rho = self.rhoc * dc / (x * (1 + x)**2)
            return rho

    nfw = NFW(0.2, Planck18)

    assert 0.2 == nfw.z
    assert Planck18 == nfw.cosmo