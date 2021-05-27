import numpy as np
from astropy.cosmology import FLRW
import astropy.units as u

from scipy.integrate import quad

from functools import cached_property

from abc import ABCMeta, abstractmethod

# ============================================================================
# CONSTANTS
# ============================================================================

SCALAR_TYPES = (int, float, np.integer, np.floating)
DEFAULT_RZ_LIM = 1e9  # in pc


# ============================================================================
# CONSTRUCTOR
# ============================================================================

# our version of ABCMeta with required attributes
class CustomMeta(ABCMeta):
    required_attributes = []

    def __call__(self, *args, **kwargs):
        obj = super(CustomMeta, self).__call__(*args, **kwargs)
        for attr_name in obj.required_attributes:
            if not hasattr(obj, attr_name):
                raise AttributeError(
                    f"Required attribute `{attr_name}` not set."
                )

        # this should not be validated here, but where?
        if not isinstance(obj.cosmo, FLRW):
            raise TypeError(f"Cosmology `{obj.cosmo}` not allowed.")
        return obj


# ============================================================================
# METACLASS
# ============================================================================


class RadialSymmetryLens(metaclass=CustomMeta):
    """
    Provide methods with the lensing integrals optimized for lens systems
    with radial symmetry.

    The user needs to define the __init__ method with the obligatory
    parameters z (redshift) and cosmo (cosmology), and the density
    method.
    """

    required_attributes = ["z", "cosmo"]

    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def density(self):
        pass

    @cached_property
    def rhoc(self):
        """Compute critical density at z, in units of (Msun.pc-3)"""
        rhoc_ = self.cosmo.critical_density(self.z)
        rhoc_ = rhoc_.to_value(u.solMass / u.pc ** 3)
        return rhoc_

    @cached_property
    def rhom(self):
        """Compute mean density at z, in units of (Msun.pc-3)"""
        rhom_ = self.cosmo.critical_density(self.z) * self.cosmo.Om(self.z)
        rhom_ = rhom_.to_value(u.solMass / u.pc ** 3)
        return rhom_

    def density2D(self, rz, rp, *args):
        """Decompose the density method variable 'r' in two components:
        rz as the distance in the line of sight direction and rp as the
        distance in the plane of the sky."""
        r = np.sqrt(rp ** 2 + rz ** 2)
        return self.density(r, *args)

    def _rz_lim(self, rp):
        if hasattr(self, "cutoff"):
            if rp < self.cutoff:
                rz = np.sqrt(self.cutoff ** 2 - rp ** 2)
            else:
                rz = 0.0
        else:
            rz = DEFAULT_RZ_LIM
        return rz

    def sigma(self, rp, *args):
        """Projected mass density along the line of sight."""
        r_sing = 1e2  # in pc
        if isinstance(rp, np.ndarray):
            sigma_singularity = [
                quad(
                    self.density2D,
                    -r_sing,
                    r_sing,
                    args=(rpi, *args),
                    limit=200,
                    points=[0.0],
                    epsabs=1e-4,
                )[0]
                for rpi in rp
            ]
            sigma_outside = [
                quad(
                    self.density2D,
                    r_sing,
                    self._rz_lim(rp),
                    args=(rpi, *args),
                    epsabs=1e-4,
                    limit=200,
                )[0]
                for rpi in rp
            ]
            sigma_ = np.array(sigma_singularity) + 2 * np.array(sigma_outside)
        elif isinstance(rp, SCALAR_TYPES):
            sigma_singularity = [
                quad(
                    self.density2D,
                    -r_sing,
                    r_sing,
                    args=(rp, *args),
                    limit=200,
                    points=[0.0],
                    epsabs=1e-4,
                )[0]
            ]
            sigma_outside = [
                quad(
                    self.density2D,
                    r_sing,
                    self._rz_lim(rp),
                    args=(rp, *args),
                    epsabs=1e-4,
                    limit=200,
                )[0]
            ]
            sigma_ = np.array(sigma_singularity) + 2 * np.array(sigma_outside)
        return np.array(sigma_)

    def radial_mean_sigma(self, rp, *args):
        """Mean projected density averaged at every point of radius rp.

        Note: Here the density method is assumed to be centered at the point
        of simmetry, so no actual integral over the 2*pi is computed as it is
        assumed constant."""
        return self.sigma(rp, *args)

    def inner_mean_sigma(self, rp, *args):
        """Mean projected density within a circle of radius rp."""
        integrand = lambda r: r * self.radial_mean_sigma(r, *args)
        ims = [
            quad(integrand, 0.0, rpi, limit=200, epsabs=1e-4)[0]
            / (0.5 * rpi ** 2)
            for rpi in rp
        ]
        return np.array(ims)

    def delta_sigma(self, rp, *args):
        """Projected density contrast computed as:
        inner_mean_sigma - radial_mean_sigma"""
        deltaSigma = self.inner_mean_sigma(rp, *args) - self.radial_mean_sigma(
            rp, *args
        )
        return deltaSigma
