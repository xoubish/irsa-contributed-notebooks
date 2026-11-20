"""Helper functions for ``euclid_highz_selection.md``.

The notebook keeps the data access and the science choices inline.
This module holds the routine pieces that would distract from them:
flux conversions, the normalization of the redshift probability distribution,
and the plotting of image stamps and spectral energy distributions (SEDs).
"""

import numpy as np
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.nddata import Cutout2D


# ---------------------------------------------------------------------------
# Photometry
# ---------------------------------------------------------------------------
def as_float_array(column):
    """Convert a table column to a plain float array.

    Parameters
    ----------
    column : astropy.table.Column, astropy.table.MaskedColumn or array-like
        Values to convert. Masked entries (for example the missing values of a catalog) become NaN.

    Returns
    -------
    numpy.ndarray
        Float array of the same length as ``column``.
    """
    return np.asarray(np.ma.filled(np.ma.asarray(column, dtype=float), np.nan), dtype=float)


def abmag(flux):
    """Convert a flux in μJy to an AB magnitude.

    Parameters
    ----------
    flux : array-like
        Flux in μJy.

    Returns
    -------
    numpy.ndarray
        AB magnitude, NaN where the flux is missing, zero or negative.
    """
    flux = np.atleast_1d(as_float_array(flux))
    mag = np.full(flux.shape, np.nan)
    positive = flux > 0
    mag[positive] = 23.9 - 2.5 * np.log10(flux[positive])
    return mag


def signal_to_noise(flux, error):
    """Signal-to-noise ratio of a flux measurement.

    Parameters
    ----------
    flux, error : array-like
        Flux and its uncertainty, in the same units.

    Returns
    -------
    numpy.ndarray
        ``flux / error``, NaN where either value is missing or the uncertainty is not positive.
    """
    flux, error = as_float_array(flux), as_float_array(error)
    snr = np.full(flux.shape, np.nan)
    valid = np.isfinite(flux) & np.isfinite(error) & (error > 0)
    snr[valid] = flux[valid] / error[valid]
    return snr


# ---------------------------------------------------------------------------
# Redshift probability distribution
# ---------------------------------------------------------------------------
def redshift_pdf(log_probability, z_grid):
    """Normalize the log-probabilities of a redshift grid to a probability density.

    Parameters
    ----------
    log_probability : numpy.ndarray
        Array of shape (n_sources, n_redshifts) with the natural log of the redshift probability of each source.
    z_grid : numpy.ndarray
        The redshift grid, of length n_redshifts.

    Returns
    -------
    numpy.ndarray
        Array of the same shape, with each row integrating to 1 over ``z_grid``.
    """
    # Subtract the maximum of each row first, so that the exponential does not underflow.
    probability = np.exp(log_probability - np.nanmax(log_probability, axis=1, keepdims=True))
    probability = np.nan_to_num(probability)
    return probability / np.trapezoid(probability, z_grid, axis=1)[:, None]


def probability_below(pdf, z_grid, z_max):
    """Integrate a redshift probability density from the start of the grid to a redshift.

    Parameters
    ----------
    pdf : numpy.ndarray
        Array of shape (n_sources, n_redshifts), normalized as by `redshift_pdf`.
    z_grid : numpy.ndarray
        The redshift grid.
    z_max : float
        Upper limit of the integral.

    Returns
    -------
    numpy.ndarray
        Probability that each source is at a redshift below ``z_max``.
    """
    below = z_grid < z_max
    return np.trapezoid(pdf[:, below], z_grid[below], axis=1)


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------
def show_stamp(ax, image, wcs, ra, dec, size_arcsec, label):
    """Show a square image stamp centered on a sky position.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes to draw on.
    image : numpy.ndarray
        The image to cut the stamp from.
    wcs : astropy.wcs.WCS
        World coordinate system of ``image``.
    ra, dec : float
        Center of the stamp in degrees.
    size_arcsec : float
        Side of the stamp in arcsec.
    label : str
        Text written in the top left corner, for example the band name.
    """
    center = SkyCoord(ra * u.deg, dec * u.deg)
    stamp = Cutout2D(image, center, size_arcsec * u.arcsec, wcs=wcs, mode='partial', fill_value=np.nan).data
    # Gray scale from the noise level: -2 to +6 times a robust standard deviation around the median.
    median = np.nanmedian(stamp)
    sigma = 1.4826 * np.nanmedian(np.abs(stamp - median))
    ax.imshow(stamp, origin='lower', cmap='gray_r', vmin=median - 2 * sigma, vmax=median + 6 * sigma)
    ax.text(0.05, 0.95, label, transform=ax.transAxes, va='top', fontsize=9,
            bbox={'facecolor': 'white', 'edgecolor': 'none', 'alpha': 0.7, 'pad': 1.5})
    ax.set_xticks([])
    ax.set_yticks([])


def plot_sed(ax, sed, color, label, show_data=True, snr_limit=2.0):
    """Plot the photometry of one source and a best-fit template from EAZY.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes to draw on.
    sed : dict
        Output of ``eazy.photoz.PhotoZ.show_fit(..., show_fnu=1, get_spec=True)``:
        wavelengths in Angstrom and fluxes in μJy.
    color : str
        Color of the template.
    label : str
        Legend label of the template.
    show_data : bool, optional
        Plot the measured fluxes. Set it to False when adding a second template to the same axes.
    snr_limit : float, optional
        Bands with a signal-to-noise ratio below this value are drawn as upper limits at ``snr_limit`` sigma.
    """
    wavelength = np.asarray(sed['templz']) / 1e4  # micron
    ax.plot(wavelength, np.asarray(sed['templf']), color=color, linewidth=1, alpha=0.8, label=label)
    # Open squares: the flux of the template in each band.
    ax.scatter(np.asarray(sed['pivot']) / 1e4, np.asarray(sed['model']), marker='s', s=28, facecolor='none',
               edgecolor=color, zorder=3)
    if not show_data:
        return
    pivot = np.asarray(sed['pivot']) / 1e4
    flux, error = np.asarray(sed['fobs'], dtype=float), np.asarray(sed['efobs'], dtype=float)
    valid = np.asarray(sed['valid'], dtype=bool) & (error > 0)
    detected = valid & (flux > snr_limit * error)
    ax.errorbar(pivot[detected], flux[detected], yerr=error[detected], fmt='o', color='black', markersize=4,
                capsize=2, zorder=4)
    limit = valid & ~detected
    ax.errorbar(pivot[limit], snr_limit * error[limit], yerr=0.3 * snr_limit * error[limit], uplims=True, fmt='none',
                color='black', zorder=4)
