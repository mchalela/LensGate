import numpy as np
from astropy.cosmology import Planck18, FlatLambdaCDM
import astropy.units as u
import astropy.constants as const

import matplotlib.pyplot as plt

from scipy import special
from scipy.integrate import quad, simps
from scipy.interpolate import CubicSpline

from functools import cached_property

from abc import ABCMeta, abstractmethod

# ============================================================================
# CONSTANTS
# ============================================================================

SCALAR_TYPES = (int, float, np.integer, np.floating)
DEFAULT_RZ_LIM = 1e8  # in pc


# ============================================================================
# CONSTRUCTOR
# ============================================================================

# our version of ABCMeta with required attributes
class CustomMeta(ABCMeta):
    required_attributes = []

    def __call__(self, *args, **kwargs):
        obj = super(CustomMeta, self).__call__(*args, **kwargs)
        for attr_name in obj.required_attributes:
            if not getattr(obj, attr_name):
                raise ValueError("required attribute (%s) not set" % attr_name)
        return obj


# ============================================================================
# METACLASS
# ============================================================================


class AxialSimetryLens(metaclass=CustomMeta):

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
        """Projected mass density along the line of sight of RhoLW12."""
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
        return self.sigma(rp, *args)

    def inner_mean_sigma(self, rp, *args):
        integrand = lambda r: r * self.radial_mean_sigma(r, *args)
        ims = [
            quad(integrand, 0.0, rpi, limit=200, epsabs=1e-4)[0]
            / (0.5 * rpi ** 2)
            for rpi in rp
        ]
        return np.array(ims)

    def delta_sigma(self, rp, *args):
        deltaSigma = self.inner_mean_sigma(rp, *args) - self.radial_mean_sigma(
            rp, *args
        )
        return deltaSigma
