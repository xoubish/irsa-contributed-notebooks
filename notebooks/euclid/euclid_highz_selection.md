---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.17.3
kernelspec:
  display_name: python3
  language: python
  name: python3
authors:
  - name: Shoubaneh Hemmati
---

# Selecting z > 6 galaxy candidates in Euclid DR1

## Overview

- Select bright galaxy candidates at redshift z > 6 in a Euclid Deep Field, from the Euclid DR1 MER catalog, read from IRSA's Parquet files in the cloud.
- Fit galaxy and brown dwarf templates to their photometry with EAZY, and keep the sources that are best fit by a galaxy at z ≥ 6.
- Inspect the images of a few candidates and add Spitzer/IRAC fluxes with forced photometry, to see how the mid-infrared changes their redshifts.

## Introduction

Galaxies at z > 6 are seen in the first billion years of the Universe.
The light they emit blueward of the Lyman-α line (1216 Å in the rest frame) is absorbed by neutral hydrogen between the galaxy and us, so their spectrum has a sharp break: there is flux redward of (1 + z) × 1216 Å and none blueward.
Galaxies selected with this break are called Lyman-break galaxies (LBGs).
At z > 6 the break is at wavelengths longer than 0.85 μm, so an LBG is detected in the Euclid near-infrared bands ($Y_E$, $J_E$, $H_E$) and not in the visible band ($I_E$): it is an $I_E$ dropout.

Euclid is not as deep as the Hubble or James Webb space telescopes, but it covers an area that is thousands of times larger.
It can therefore find the brightest and rarest z > 6 galaxies, which the smaller surveys miss.
The difficulty is that other sources are red in the same bands and far more numerous: cool stars and brown dwarfs of spectral types M, L and T in the Milky Way, galaxies at z = 1 to 4 that are dusty or old, and image artifacts.

This notebook follows [Euclid Collaboration: Allen et al. (2026)](https://arxiv.org/abs/2511.02926), who tested how to select z > 6 LBGs in the Euclid Deep Fields with simulated photometry, and applied the method to the Euclid Quick Release 1 (Q1) data.
Their main conclusions are the steps of this notebook:

- Start with signal-to-noise cuts: a detection in the near-infrared bands and no detection in $I_E$.
- Fit SED templates of galaxies over all bands from the optical to the mid-infrared, and templates of brown dwarfs, and keep the sources that are better fit by a galaxy at high redshift. This gave them a cleaner sample than cuts in color-color diagrams.
- Inspect the images of the candidates, because many are image artifacts.
- Use deep optical data to confirm the break, and Spitzer/IRAC data to separate the candidates from brown dwarfs and from galaxies at lower redshift.

We apply these steps to the Euclid DR1 data, which are deeper than Q1 in the Euclid Deep Fields.
The result is a list of candidates, not a sample for science: the last section lists what a real analysis needs in addition.

**Data used.**
- The Euclid DR1 MER catalog of the Euclid Deep Survey: fluxes in the Euclid bands and in the ground-based Subaru Hyper Suprime-Cam (HSC) bands $g$, $r$, $i$, $z$.
- The Euclid DR1 MER mosaics and the Cosmic Dawn Survey Spitzer/IRAC mosaics, for the image stamps and the forced photometry of a few candidates.

Euclid observes three Euclid Deep Fields: North (EDF-N), South (EDF-S), and Fornax (EDF-F).
We use a circle of 0.25 degree radius (0.2 square degrees) in EDF-N, the field with deep HSC data in DR1.

**Data access.** The catalog is read from the HATS Parquet files that IRSA keeps in an AWS S3 bucket. Access is free and needs no account. The selected rows are saved to a local file, so they are downloaded only once.
The images are read by the `euclid_phot` package of the [Euclid forced photometry](forced_photometry/README.md) notebooks, which downloads small cutouts from IRSA and keeps them in a local cache.
EAZY downloads its templates and filter curves (about 200 MB) the first time it is used.

**Workflow.** The notebook has these sections:
1. Get the $I_E$ dropouts from the catalog.
2. Remove flagged sources and sources detected in the optical.
3. Fit the SEDs with EAZY.
4. Select the z ≥ 6 candidates.
5. Compare with a color selection.
6. Inspect the candidates and add IRAC with forced photometry.

**Runtime.** About 10 minutes, and a few minutes more the first time, when EAZY and `euclid_phot` download their files. Most of the time is spent in the SED fits of Sections 3 and 6 and in the forced photometry of Section 6.

## Imports

`euclid_phot` and its dependency Tractor are not on PyPI.
The package is in the `forced_photometry` directory next to this notebook, and its install script installs both: uncomment the first two lines, and restart the kernel when the install has finished.
EAZY prints a message that it failed to import `dust_attenuation` each time it is loaded. This optional dependency is not used in this notebook, and the message can be ignored.

```{code-cell} ipython3
# Uncomment the next lines to install dependencies if needed.
# import sys
# !bash forced_photometry/scripts/install.sh --python "{sys.executable}"
# %pip install numpy==2.4.6 matplotlib==3.10.5 astropy==7.2.0 pyarrow==24.0.0 astropy-healpix==1.1.3 eazy==0.8.7
```

```{code-cell} ipython3
import contextlib
import io
import os
import sys
import warnings

import eazy
import eazy.photoz
import euclid_phot as ep
import matplotlib.pyplot as plt
import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.dataset
import pyarrow.fs
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.table import Table, join
from astropy_healpix import HEALPix

# The helper functions of this notebook are in code_src/euclid_highz_selection.py.
sys.path.append('code_src')
from euclid_highz_selection import (abmag, as_float_array, plot_sed, probability_below,  # noqa: E402
                                    redshift_pdf, show_stamp, signal_to_noise)

# The forced photometry in Section 6 reads the DR1 images.
ep.set_data_access('DR1')

%matplotlib inline
```

EAZY keeps its templates and filter curves in the directory of its Python package, and downloads them from GitHub if they are not there.
We use them for the SED fits of Section 3, and for the next figure.

```{code-cell} ipython3
if not os.path.exists(os.path.join(eazy.utils.DATA_PATH, 'templates')):
    eazy.fetch_eazy_photoz()
    eazy.set_data_path()

# EAZY writes and reads its input catalog here.
os.makedirs('output', exist_ok=True)
```

## What a z > 6 galaxy looks like in the Euclid bands

Before selecting anything, we plot the three kinds of sources that this notebook is about, with the filters we use.
The model spectra are from the template files of EAZY: a young galaxy with little dust, which we place at z = 7 and absorb blueward of Lyman-α with the model of the intergalactic medium of Inoue et al. (2014); an old and dusty galaxy, which we place at z = 2.3; and a brown dwarf with an effective temperature of 1200 K (a T dwarf), from the BT-Settl models.

```{code-cell} ipython3
def model_spectrum(kind):
    """Observed wavelength and flux density of one of three model sources.

    Parameters
    ----------
    kind : {'galaxy at z = 7', 'red galaxy at z = 2.3', 'brown dwarf (1200 K)'}
        The model source.

    Returns
    -------
    wavelength : numpy.ndarray
        Observed wavelength in μm.
    flux : numpy.ndarray
        Flux density per unit frequency, with a value of 1 at 1.77 μm (the H band).
    """
    with contextlib.redirect_stdout(io.StringIO()), warnings.catch_warnings():
        warnings.simplefilter('ignore')
        if kind == 'brown dwarf (1200 K)':
            stars = eazy.templates.load_phoenix_stars(add_carbon_star=False, sonora_dwarfs=False)
            template = next(star for star in stars if star.name.startswith('bt-settl_t01200'))
            wavelength, flux = template.wave, np.squeeze(template.flux_fnu())
        else:
            name, redshift = {'galaxy at z = 7': ('corr_sfhz_13_bin0_av0.01.fits', 7.0),
                              'red galaxy at z = 2.3': ('corr_sfhz_13_bin2_av0.50.fits', 2.3)}[kind]
            template = eazy.templates.Template(file=os.path.join(eazy.utils.DATA_PATH, 'templates/sfhz', name))
            wavelength = template.wave * (1 + redshift)
            # Absorption by the intergalactic medium blueward of Lyman-alpha.
            flux = np.squeeze(template.flux_fnu(z=redshift)) * eazy.igm.Inoue14().full_IGM(redshift, wavelength)
    wavelength = wavelength / 1e4  # Angstrom to micron
    return wavelength, flux / np.interp(1.77, wavelength, flux)
```

```{code-cell} ipython3
filter_file = eazy.filters.FilterFile(os.path.join(eazy.utils.DATA_PATH, 'filters', 'FILTER.RES.latest'))
# EAZY filter numbers and labels of the bands of this notebook. Filters 18 and 19 are IRAC channels 1 and 2.
PLOT_FILTERS = {314: 'g', 315: 'r', 316: 'i', 317: 'z', 410: '$I_E$', 411: '$Y_E$', 412: '$J_E$', 413: '$H_E$',
                18: '3.6 μm', 19: '4.5 μm'}

fig, ax = plt.subplots(figsize=(11, 4.8))
for kind, color in [('galaxy at z = 7', 'crimson'), ('red galaxy at z = 2.3', 'tab:orange'),
                    ('brown dwarf (1200 K)', 'tab:blue')]:
    wavelength, flux = model_spectrum(kind)
    ax.plot(wavelength, flux, color=color, linewidth=1, label=kind)

# Filter curves at the bottom of the figure, each scaled to the same height.
for number, label in PLOT_FILTERS.items():
    filter_curve = filter_file[number]
    wavelength, throughput = filter_curve.wave / 1e4, filter_curve.throughput / filter_curve.throughput.max()
    is_euclid = number >= 410
    ax.fill_between(wavelength, 2e-3, 2e-3 * 10**(0.9 * throughput), color='0.3' if is_euclid else '0.75',
                    alpha=0.5, linewidth=0)
    ax.text(filter_curve.pivot / 1e4, 2.2e-2 if is_euclid else 1.3e-3, label, ha='center', fontsize=8)
ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlim(0.38, 5.5)
ax.set_ylim(1e-3, 30)
ax.set_xticks([0.5, 1, 2, 4], ['0.5', '1', '2', '4'])
ax.set_xlabel('observed wavelength (μm)')
ax.set_ylabel('flux density, relative to $H_E$')
ax.legend(frameon=False, loc='upper left')
plt.show()
```

**Figure 1.** Model spectra of a young galaxy at z = 7 (red), an old and dusty galaxy at z = 2.3 (orange) and a brown dwarf (blue), all with the same flux in $H_E$.
The shaded curves at the bottom are the filters: the Euclid bands in dark gray, and the HSC bands ($g$, $r$, $i$, $z$) and the IRAC channels in light gray.
The galaxy at z = 7 has no flux blueward of 1 μm, and a flat spectrum with emission lines redward of it.
The other two are also a hundred times fainter in $I_E$ and in the optical than in $H_E$, so with noisy data they can pass as dropouts.
The two galaxies differ most at 3.6 and 4.5 μm: the red galaxy is several times brighter there than in $H_E$, and the galaxy at z = 7 is a little fainter.
This is why the IRAC bands matter in Section 6.
The brown dwarf differs in the shape of its spectrum, which has deep molecular absorption bands between the peaks in $Y_E$, $J_E$ and $H_E$; the model is coarsely sampled beyond 2.5 μm, which gives the straight segments there.

## Science choices

The cuts below follow Allen et al. (2026).
Section 3 of that paper selects sources with a signal-to-noise ratio (S/N) of at least 5 in one near-infrared band and below 2 in $I_E$.
We ask for S/N ≥ 5 in both $J_E$ and $H_E$: a source detected in two bands is less likely to be an image artifact, and a galaxy at z = 6 to 10 has flux in both.
This is the choice Allen et al. made when they applied the method to the Q1 data (their Section 6), where they also asked for no detection in $Y_E$ to select z > 8.
We leave $Y_E$ free, so that the selection starts at z = 6.

```{code-cell} ipython3
# Center (degrees) and radius (degrees) of the search circle in EDF-N.
FIELD_RA, FIELD_DEC, FIELD_RADIUS = 269.73, 66.02, 0.25

SNR_DETECTED = 5.0      # minimum S/N in J and in H
SNR_NOT_DETECTED = 2.0  # maximum S/N in the bands blueward of the break

# Cuts on the SED fits.
Z_HIGH = 6.0         # minimum photometric redshift of a candidate
Z_LOW = 5.0          # a candidate must have little probability below this redshift ...
P_LOW_MAX = 0.05     # ... at most this much

# Section 6: number of candidates to follow up, side of the forced photometry field and of the stamps (arcsec).
N_FOLLOWUP = 3
FOLLOWUP_SIZE = 20.0
STAMP_SIZE = 8.0
```

The MER catalog has several flux measurements per band.
We use the aperture fluxes with a diameter of 2 times the FWHM (`2fwhm_aper`): they are measured in the same aperture on images matched to the same PSF in all bands, including the ground-based ones, so they give consistent colors, and they exist for sources that are not detected in $I_E$.
They are not total fluxes.

For each band we list the catalog table that has it, and the number of its filter curve in the filter file of EAZY.

```{code-cell} ipython3
# band name in the catalog columns: (catalog that has the band, EAZY filter number, label)
BANDS = {
    'g_ext_hsc': ('main', 314, 'HSC g'),
    'r_ext_hsc': ('photometry', 315, 'HSC r'),
    'i2_ext_hsc': ('photometry', 316, 'HSC i'),
    'z_ext_hsc': ('main', 317, 'HSC z'),
    'vis': ('main', 410, '$I_E$'),
    'y': ('main', 411, '$Y_E$'),
    'j': ('main', 412, '$J_E$'),
    'h': ('main', 413, '$H_E$'),
}
# The bands that are entirely blueward of the Lyman-alpha break at z >= 6.
BLUE_BANDS = ['g_ext_hsc', 'r_ext_hsc', 'i2_ext_hsc', 'vis']
```

## 1. Get the $I_E$ dropouts from the catalog

The DR1 MER catalog of the Euclid Deep Survey is in two tables that share the `object_id` column.
`mer_catalogue_deep_survey` has the positions, the flags, and the photometry in the Euclid bands and in a first set of ground-based bands.
`mer_catalogue_deep_survey_photometry` has the photometry in the other external bands.
In EDF-N the HSC $g$ and $z$ fluxes are in the first table, and the HSC $r$ and $i$ fluxes in the second, so we need both.

IRSA serves these tables in two ways: with its TAP service, and as [HATS](https://hats.readthedocs.io/) catalogs, which are sets of Parquet files in an AWS S3 bucket.
We read the Parquet files directly.
This needs no service to run the search, and takes less than a minute for our circle.
The IRSA tutorial [Euclid DR1 HATS Catalogs: Introduction](https://caltech-ipac.github.io/irsa-tutorials/euclid-dr1-hats-intro/) describes the catalogs and other ways to read them.

```{code-cell} ipython3
s3 = pyarrow.fs.S3FileSystem(anonymous=True, region='us-east-1')
HATS_PREFIX = 'nasa-irsa-euclid-dr1/contributed/dr1/hats/reformatted_catalogs'
MAIN_CATALOG = 'mer_catalogue_deep_survey'
PHOTOMETRY_CATALOG = 'mer_catalogue_deep_survey_photometry'
```

A HATS catalog is split into files by position on the sky: each file has the sources of one HEALPix pixel (a partition), and the rows carry the index of the smaller HEALPix pixel of order 9 they are in (`_healpix_9`, pixels of about 0.1 degree).
To read a circle on the sky we therefore:

- list the order 9 pixels that overlap the circle;
- open only the partitions that contain these pixels;
- ask for the rows with one of these pixel indices, and with any other condition we have. PyArrow applies these conditions while it reads, so only the matching rows are downloaded.

The pixels cover a little more than the circle, so the exact cut on the distance from the center is done after reading.

```{code-cell} ipython3
def read_hats_circle(catalog, columns, ra, dec, radius, condition=None):
    """Read the rows of an IRSA Euclid DR1 HATS catalog that are in the HEALPix pixels overlapping a circle.

    Parameters
    ----------
    catalog : str
        Name of the catalog, for example ``'mer_catalogue_deep_survey'``.
    columns : list of str
        Columns to read.
    ra, dec, radius : float
        Center and radius of the circle in degrees.
    condition : pyarrow.compute.Expression, optional
        Extra condition on the rows.

    Returns
    -------
    astropy.table.Table
        The requested columns of the matching rows. Some rows are outside the circle, by up to about 0.3 degree.
    """
    catalog_path = f'{HATS_PREFIX}/{catalog}/euclid_dr1_reformatted_catalogs_{catalog}-hats'

    # Order 9 pixels (nested numbering) that overlap the circle. The margin of 0.1 degree is for the
    # sources close to the edge of a pixel.
    healpix = HEALPix(nside=2**9, order='nested')
    pixels = np.unique(healpix.cone_search_lonlat(ra * u.deg, dec * u.deg, (radius + 0.1) * u.deg))
    row_filter = pc.field('_healpix_9').isin(pa.array(pixels, pa.int64()))

    # Partitions that contain these pixels. In the nested numbering, the pixel of order n that contains
    # a pixel of order 9 is its index divided by 4**(9 - n).
    with s3.open_input_stream(f'{catalog_path}/partition_info.csv') as partition_file:
        partitions = Table.read(partition_file.read().decode(), format='ascii.csv')
    partition_filter = None
    for order, pixel in zip(partitions['Norder'], partitions['Npix']):
        if np.any(pixels // 4**(9 - order) == pixel):
            this_partition = (pc.field('Norder') == int(order)) & (pc.field('Npix') == int(pixel))
            partition_filter = this_partition if partition_filter is None else partition_filter | this_partition
    row_filter = row_filter & partition_filter
    if condition is not None:
        row_filter = row_filter & condition

    dataset = pyarrow.dataset.dataset(f'{catalog_path}/dataset', partitioning='hive', filesystem=s3,
                                      ignore_prefixes=['_', '.'])
    return Table.from_pandas(dataset.to_table(columns=columns, filter=row_filter).to_pandas())
```

The signal-to-noise cuts are conditions of the read, so we download about a hundred rows instead of all the sources in the circle.
A cut `flux < 2 * error` in $I_E$ also keeps the sources with a negative $I_E$ flux, which is what we want.

```{code-cell} ipython3
def flux_column(band):
    """PyArrow expressions for the aperture flux of a band and its uncertainty."""
    return pc.field(f'flux_{band}_2fwhm_aper'), pc.field(f'fluxerr_{band}_2fwhm_aper')


flux_j, error_j = flux_column('j')
flux_h, error_h = flux_column('h')
flux_vis, error_vis = flux_column('vis')
dropout_condition = ((flux_j >= pc.multiply(pc.scalar(SNR_DETECTED), error_j))
                     & (flux_h >= pc.multiply(pc.scalar(SNR_DETECTED), error_h))
                     & (flux_vis < pc.multiply(pc.scalar(SNR_NOT_DETECTED), error_vis)))

# Positions, the flag for a detection in the VIS image, the quality flags and the Galactic reddening.
main_columns = ['object_id', 'ra', 'dec', 'vis_det', 'spurious_flag', 'det_quality_flag',
                'flag_y', 'flag_j', 'flag_h', 'gal_ebv']
photometry_columns = ['object_id']
# The aperture flux and its uncertainty in each band.
for band, (catalog, _, _) in BANDS.items():
    band_columns = [f'flux_{band}_2fwhm_aper', f'fluxerr_{band}_2fwhm_aper']
    if catalog == 'main':
        main_columns += band_columns
    else:
        photometry_columns += band_columns
```

We read the dropouts from the first catalog and keep the ones inside the circle, then read the rows of the same sources from the second catalog and join the two.

The result is saved in the `data` directory and read from there the next time the notebook runs, so a later run does not need the network for this step.
The file name has the position and the cuts in it: when you change one of them, the catalogs are read again.
To force a new read, delete the file.

```{code-cell} ipython3
os.makedirs('data', exist_ok=True)
dropouts_file = (f'data/highz_dropouts_{FIELD_RA:g}_{FIELD_DEC:g}_{FIELD_RADIUS:g}'
                 f'_snr{SNR_DETECTED:g}_{SNR_NOT_DETECTED:g}.ecsv')
if os.path.exists(dropouts_file):
    dropouts = Table.read(dropouts_file)
else:
    dropouts = read_hats_circle(MAIN_CATALOG, main_columns, FIELD_RA, FIELD_DEC, FIELD_RADIUS, dropout_condition)
    center = SkyCoord(FIELD_RA * u.deg, FIELD_DEC * u.deg)
    separation = SkyCoord(dropouts['ra'], dropouts['dec'], unit='deg').separation(center)
    dropouts = dropouts[separation <= FIELD_RADIUS * u.deg]

    same_sources = pc.field('object_id').isin(pa.array(dropouts['object_id']))
    photometry = read_hats_circle(PHOTOMETRY_CATALOG, photometry_columns, FIELD_RA, FIELD_DEC, FIELD_RADIUS,
                                  same_sources)
    dropouts = join(dropouts, photometry, keys='object_id', join_type='left')
    dropouts.write(dropouts_file)
print(f'{len(dropouts)} sources detected in J and H and not detected in I_E')
```

We compute the S/N in each band, and print the 5σ depth that the median uncertainty corresponds to.
These depths are for the 2 FWHM aperture, and tell how strong a break the data can show.

```{code-cell} ipython3
snr = {band: signal_to_noise(dropouts[f'flux_{band}_2fwhm_aper'], dropouts[f'fluxerr_{band}_2fwhm_aper'])
       for band in BANDS}

for band, (_, _, label) in BANDS.items():
    median_error = np.nanmedian(as_float_array(dropouts[f'fluxerr_{band}_2fwhm_aper']))
    print(f'{label:6s}  5 sigma depth = {abmag(5 * median_error)[0]:.1f} AB')
```

The $I_E$ and HSC $g$, $r$, $i$ images are 1 to 2.5 magnitudes deeper than the near-infrared images.
A source at the near-infrared detection limit with no flux in these bands therefore has a break of more than 2 magnitudes, when the non-detections are taken at 2σ.
A brighter source has a stronger and more certain break.

## 2. Remove flagged sources and sources detected in the optical

The MER catalog has flags that mark sources with a problem.
We keep the sources with:

- `spurious_flag = 0`, so the source is not classified as spurious;
- `det_quality_flag = 0`, so the detection step raised no warning (for example a neighbor, a bright star, or the border of the image);
- `flag_y`, `flag_j` and `flag_h` all 0, so no pixel of the source is flagged in the near-infrared images.

```{code-cell} ipython3
good_flags = np.ones(len(dropouts), dtype=bool)
for flag in ['spurious_flag', 'det_quality_flag', 'flag_y', 'flag_j', 'flag_h']:
    good_flags &= as_float_array(dropouts[flag]) == 0
print(f'{good_flags.sum()} of {len(dropouts)} sources have good flags')
```

A galaxy at z ≥ 6 has no flux blueward of 0.85 μm, so it is not detected in the HSC $g$, $r$ and $i$ bands.
We ask for S/N < 2 in each of them.
We do not cut on the HSC $z$ band (0.85 to 0.93 μm): the break of a galaxy at z = 6 to 6.6 is inside this band, so such a galaxy can have $z$-band flux.
The SED fit of the next section uses the $z$-band flux.

```{code-cell} ipython3
# A missing flux gives a NaN S/N. "not (S/N >= 2)" keeps those sources.
not_in_optical = np.all([~(snr[band] >= SNR_NOT_DETECTED) for band in BLUE_BANDS], axis=0)
sample = dropouts[good_flags & not_in_optical]
snr = {band: values[good_flags & not_in_optical] for band, values in snr.items()}
print(f'{len(sample)} sources are also not detected in HSC g, r and i')
```

We look at the sample before fitting it: how bright the sources are, and how strong their breaks are.
The strength of the break is the color $I_E - H_E$.
The sources are not detected in $I_E$, so we use the 2σ upper limit on their $I_E$ flux, which gives a lower limit on the color.

```{code-cell} ipython3
mag_h_all = abmag(dropouts['flux_h_2fwhm_aper'])
mag_h_sample = abmag(sample['flux_h_2fwhm_aper'])
# 2 sigma limit on the I_E magnitude, and the lower limit on the color I_E - H_E.
limit_vis = abmag(SNR_NOT_DETECTED * as_float_array(sample['fluxerr_vis_2fwhm_aper']))
break_limit = limit_vis - mag_h_sample

fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
bins = np.arange(21.5, 25.75, 0.25)
axes[0].hist(mag_h_all, bins=bins, color='0.85', label=f'detected in J and H, not in $I_E$ ({len(dropouts)})')
axes[0].hist(mag_h_all[good_flags], bins=bins, color='0.6', label=f'good flags ({good_flags.sum()})')
axes[0].hist(mag_h_sample, bins=bins, color='crimson', label=f'not detected in HSC g, r, i ({len(sample)})')
axes[0].set_xlabel('$H_E$ magnitude (AB, 2 FWHM aperture)')
axes[0].set_ylabel('number of sources')
axes[0].legend(frameon=False, fontsize=8, loc='upper left')

axes[1].scatter(mag_h_sample, break_limit, marker='^', s=22, color='crimson')
axes[1].set_xlabel('$H_E$ magnitude (AB, 2 FWHM aperture)')
axes[1].set_ylabel('lower limit on $I_E - H_E$ (mag)')
fig.tight_layout()
plt.show()
```

**Figure 2.** The sample before the SED fits.
Left: $H_E$ magnitudes of the sources after each cut.
Right: lower limit on the color $I_E - H_E$ of the sources of the final sample, from the 2σ upper limit on their $I_E$ flux.
The limit on the break follows the $H_E$ magnitude, because the depth of the $I_E$ image is almost the same for all sources: the break is only known to be stronger than about 2.5 magnitudes for the faintest sources, and stronger than 4 magnitudes for the brightest.
The cut on the HSC bands removes more than half of the sources with good flags: they have some optical flux, so they are not at z > 6.

## 3. Fit the SEDs with EAZY

[EAZY](https://github.com/gbrammer/eazy-py) fits a combination of galaxy templates to the photometry of each source, on a grid of redshifts.
It returns the $\chi^2$ of the best fit at each redshift, which gives the probability distribution of the redshift, p(z).
We use the same settings as Allen et al. (2026):

- the `blue_sfhz_13` templates, which include galaxies with strong emission lines and blue colors, as expected at high redshift;
- redshifts from 0 to 15;
- no prior on the redshift. The default priors of EAZY give the probability of a redshift for a given magnitude, and would move these bright sources to low redshift.

One setting is different: our redshift grid has steps of 0.01 × (1 + z), the default of EAZY, and theirs has steps of 0.01.
The wider steps at high redshift make the fit five times faster.

The next function writes the fluxes in the format EAZY reads, and runs the fit.
After the galaxy fit it fits templates of stars to the same photometry, at redshift 0.
For the stars we use the BT-Settl model grid that EAZY downloads, with effective temperatures from 400 to 7000 K: it covers the M, L and T dwarfs.
Allen et al. used the Sonora models, which cover the same cool dwarfs.

```{code-cell} ipython3
def fit_seds(fluxes, errors, filter_numbers, name, ebv=0.0):
    """Fit galaxy and star templates to the photometry of a list of sources with EAZY.

    Parameters
    ----------
    fluxes, errors : dict
        Flux and flux uncertainty in μJy, as one array per band. Both have the same keys.
    filter_numbers : dict
        For each band, the number of its filter curve in the filter file of EAZY.
    name : str
        Name of the files written to the ``output`` directory.
    ebv : float, optional
        Galactic reddening E(B-V) of the field. EAZY corrects the galaxy fit for it.

    Returns
    -------
    eazy.photoz.PhotoZ
        The fit. Its attributes used in this notebook are ``zgrid``, ``zbest``, ``chi2_fit``, ``lnp``
        (galaxy templates) and ``star_chi2``, ``star_teff`` (star templates).
    """
    # EAZY reads the flux of filter number N from a column F<N> and its uncertainty from E<N>.
    # A value below -90 marks a missing measurement.
    catalog = Table({'id': np.arange(len(next(iter(fluxes.values()))))})
    for band, number in filter_numbers.items():
        flux, error = as_float_array(fluxes[band]), as_float_array(errors[band])
        missing = ~np.isfinite(flux) | ~np.isfinite(error) | (error <= 0)
        catalog[f'F{number}'] = np.where(missing, -99.0, flux)
        catalog[f'E{number}'] = np.where(missing, -99.0, error)
    catalog.write(f'output/{name}.fits', overwrite=True)
    with open(f'output/{name}.translate', 'w') as translate_file:
        for number in filter_numbers.values():
            translate_file.write(f'F{number} F{number}\nE{number} E{number}\n')

    parameters = {
        'CATALOG_FILE': f'output/{name}.fits',
        'MAIN_OUTPUT_FILE': f'output/{name}',
        'FILTERS_RES': os.path.join(eazy.utils.DATA_PATH, 'filters', 'FILTER.RES.latest'),
        'TEMPLATES_FILE': 'templates/sfhz/blue_sfhz_13.param',
        'Z_MIN': 0.01, 'Z_MAX': 15.0, 'Z_STEP': 0.01,
        'APPLY_PRIOR': 'n',
        'PRIOR_ABZP': 23.9,  # the fluxes are in μJy
        'MW_EBV': ebv, 'CAT_HAS_EXTCORR': 'n',
        'FIX_ZSPEC': 'n',
    }
    # EAZY prints many lines and progress bars. We hide them; errors are still raised.
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()), \
            warnings.catch_warnings():
        warnings.simplefilter('ignore')
        fit = eazy.photoz.PhotoZ(param_file=None, translate_file=f'output/{name}.translate', zeropoint_file=None,
                                 params=parameters, load_prior=False, load_products=False, n_proc=0)
        fit.fit_catalog(n_proc=0)
        # Star templates, for all filters between 0.3 and 5 μm.
        fit.fit_phoenix_stars(sonora=False, wave_lim=[3000, 5.0e4])
    return fit
```

We run it on the sample.
The reddening by dust in the Milky Way is small in this field, so we give EAZY the median value of the sample.

```{code-cell} ipython3
fluxes = {band: sample[f'flux_{band}_2fwhm_aper'] for band in BANDS}
errors = {band: sample[f'fluxerr_{band}_2fwhm_aper'] for band in BANDS}
filter_numbers = {band: number for band, (_, number, _) in BANDS.items()}
field_ebv = float(np.nanmedian(as_float_array(sample['gal_ebv'])))

fit = fit_seds(fluxes, errors, filter_numbers, 'highz_catalog', ebv=field_ebv)
```

From the fit we take four numbers per source:

- `z_phot`, the redshift with the highest probability;
- `chi2_galaxy`, the $\chi^2$ of the best galaxy fit;
- `chi2_star`, the $\chi^2$ of the best star template;
- `p_low`, the probability that the redshift is below 5, the integral of p(z) from 0 to 5.

```{code-cell} ipython3
def summarize_fit(fit):
    """Collect the numbers used to select candidates from an EAZY fit.

    Parameters
    ----------
    fit : eazy.photoz.PhotoZ
        Output of `fit_seds`.

    Returns
    -------
    astropy.table.Table
        One row per source with the columns ``z_phot``, ``chi2_galaxy``, ``chi2_star``,
        ``star_teff`` (effective temperature of the best star template, in K) and ``p_low``.
    """
    pdf = redshift_pdf(fit.lnp, fit.zgrid)
    best_star = np.argmin(fit.star_chi2, axis=1)
    return Table({
        'z_phot': np.asarray(fit.zbest, dtype=float),
        'chi2_galaxy': np.min(fit.chi2_fit, axis=1),
        'chi2_star': np.min(fit.star_chi2, axis=1),
        'star_teff': fit.star_teff[best_star],
        'p_low': probability_below(pdf, fit.zgrid, Z_LOW),
    })


results = summarize_fit(fit)
for column in results.colnames:
    sample[column] = results[column]
sample['mag_h'] = abmag(sample['flux_h_2fwhm_aper'])
```

## 4. Select the z ≥ 6 candidates

Allen et al. (2026) keep a source when its best-fit redshift is high and a galaxy fits it better than a star ($\chi^2_{\rm star} > \chi^2_{\rm galaxy}$).
They also mark the sources with more than 5% of their probability at z < 5 as having a second, low-redshift solution.
We use the three conditions as cuts:

```{code-cell} ipython3
is_high_z = sample['z_phot'] >= Z_HIGH
not_a_star = sample['chi2_star'] > sample['chi2_galaxy']
no_low_z_solution = sample['p_low'] <= P_LOW_MAX
sample['candidate'] = is_high_z & not_a_star & no_low_z_solution

print(f'{"Detected in J and H, not in I_E":45s} {len(dropouts):5d}')
print(f'{"Good flags":45s} {good_flags.sum():5d}')
print(f'{"Not detected in HSC g, r, i":45s} {len(sample):5d}')
print(f'{f"Best-fit redshift >= {Z_HIGH:g}":45s} {is_high_z.sum():5d}')
print(f'{"Galaxy fits better than a star":45s} {(is_high_z & not_a_star).sum():5d}')
print(f'{f"P(z < {Z_LOW:g}) <= {P_LOW_MAX:g}":45s} {sample["candidate"].sum():5d}')
```

About a dozen sources pass all cuts.
To judge this number we compare it with the number of galaxies that are expected.
Allen et al. (2026, their Table 3) expect about 270 galaxies at z = 6 to 8 in the 53 square degrees of the Euclid Deep Fields at a depth of 24.5 AB, which is about 5 per square degree.

```{code-cell} ipython3
area = np.pi * FIELD_RADIUS**2  # square degrees
n_bright = np.sum(sample['candidate'] & (sample['mag_h'] < 24.5))
print(f'{n_bright} candidates brighter than H = 24.5 in {area:.2f} square degrees: {n_bright / area:.0f} per square degree')
```

We find several times more candidates than the expected number of galaxies.
Our aperture magnitudes are also fainter than total magnitudes, which makes the difference larger still.
So most of the candidates are not galaxies at z > 6.
The rest of the notebook looks at what they can be.

The next figure shows what the fits found for all sources of the sample.

```{code-cell} ipython3
candidates = sample[sample['candidate']]
rejected_as_star = sample[is_high_z & ~not_a_star]

fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))

# Left: best-fit redshift of all sources and of the candidates.
bins = np.arange(0, 12.5, 0.5)
axes[0].hist(sample['z_phot'], bins=bins, color='0.75', label=f'all sources ({len(sample)})')
axes[0].hist(candidates['z_phot'], bins=bins, color='crimson', label=f'candidates ({len(candidates)})')
axes[0].axvline(Z_HIGH, color='black', linestyle='--', linewidth=0.8)
axes[0].set_xlabel('best-fit redshift')
axes[0].set_ylabel('number of sources')
axes[0].legend(frameon=False)

# Middle: chi2 of the best star template against chi2 of the best galaxy fit, for the sources at z_phot >= Z_HIGH.
high_z = sample[is_high_z]
axes[1].scatter(high_z['chi2_galaxy'], high_z['chi2_star'], s=14, c=np.where(high_z['candidate'], 'crimson', '0.6'))
limits = [1, 1.2 * max(high_z['chi2_galaxy'].max(), high_z['chi2_star'].max())]
axes[1].plot(limits, limits, color='black', linestyle='--', linewidth=0.8)
axes[1].set_xscale('log')
axes[1].set_yscale('log')
axes[1].set_xlabel(r'$\chi^2$ of the best galaxy fit')
axes[1].set_ylabel(r'$\chi^2$ of the best star template')

# Right: H magnitude against redshift of the candidates.
axes[2].scatter(candidates['z_phot'], candidates['mag_h'], s=18, color='crimson')
axes[2].invert_yaxis()
axes[2].set_xlabel('best-fit redshift')
axes[2].set_ylabel('$H_E$ magnitude (AB, 2 FWHM aperture)')
fig.tight_layout()
plt.show()
```

**Figure 3.** Results of the SED fits.
Left: best-fit redshift of all sources of the sample (gray) and of the candidates (red); the dashed line is the redshift cut.
Middle: $\chi^2$ of the best star template against the $\chi^2$ of the best galaxy fit, for the sources with a best-fit redshift of at least 6; sources below the dashed line are better fit by a star and are rejected.
Right: $H_E$ magnitude and redshift of the candidates.

The sample has two groups of sources: one best fit at z ≥ 6, and one best fit at z = 1 to 2.5, where the Balmer break of a red galaxy is between the optical and the near-infrared bands.
For most sources the star and the galaxy templates fit almost equally well (middle), so the comparison of the two $\chi^2$ removes only a few sources.
The fit has eight bands, and a dropout has flux in three or four of them: this is not enough to tell a cool dwarf from a galaxy at high redshift.
Several sources have a $\chi^2$ of a few tens or more, which is high for eight bands: neither kind of template describes them, and their photometry should be checked.
The candidates have $H_E$ magnitudes from about 23.4 to 25.2 (right).
A galaxy at z ≈ 7 with $H_E$ = 23.5 has an absolute UV magnitude of about −23.5, which is as luminous as the most extreme candidates of Allen et al. (2026).
Finding several candidates within half a magnitude of that, in a fifth of a square degree, is one more sign that most of them are not galaxies at z > 6.

## 5. Compare with a color selection

High-redshift galaxies were traditionally selected with cuts in a color-color diagram.
Allen et al. (2026) give such cuts for Euclid in their Appendix A.4.
For galaxies at z = 6 to 8, which drop out of the $z$ band, the cuts are:

$$z - Y_E > 1.14, \qquad Y_E - J_E < 0.5, \qquad z - Y_E > 4\,(Y_E - J_E) + 1 .$$

The first color measures the break, and the second asks for a blue color redward of the break, which removes brown dwarfs and red galaxies at lower redshift.
The cuts need S/N > 2 in $Y_E$.
For a source that is not detected in the $z$ band, we use its 2σ upper limit, which gives a lower limit on $z - Y_E$.

```{code-cell} ipython3
flux_z = as_float_array(sample['flux_z_ext_hsc_2fwhm_aper'])
limit_z = SNR_NOT_DETECTED * as_float_array(sample['fluxerr_z_ext_hsc_2fwhm_aper'])
z_is_limit = ~(flux_z > limit_z)

sample['z-Y'] = abmag(np.where(z_is_limit, limit_z, flux_z)) - abmag(sample['flux_y_2fwhm_aper'])
sample['Y-J'] = abmag(sample['flux_y_2fwhm_aper']) - abmag(sample['flux_j_2fwhm_aper'])
has_colors = (snr['y'] > 2) & np.isfinite(sample['z-Y'])

in_color_box = (has_colors & (sample['z-Y'] > 1.14) & (sample['Y-J'] < 0.5)
                & (sample['z-Y'] > 4 * sample['Y-J'] + 1))
print(f'{in_color_box.sum()} sources pass the color cuts, {(in_color_box & sample["candidate"]).sum()} of them '
      f'are SED candidates')
print(f'{(sample["candidate"] & ~in_color_box).sum()} SED candidates do not pass the color cuts '
      f'({(sample["candidate"] & ~has_colors).sum()} of them have S/N < 2 in Y)')
```

We plot the two colors of the sources with S/N > 2 in $Y_E$.

```{code-cell} ipython3
fig, ax = plt.subplots(figsize=(6.5, 5.5))
groups = [('other sources', ~sample['candidate'] & ~(is_high_z & ~not_a_star), '0.7'),
          ('z ≥ 6, better fit by a star', is_high_z & ~not_a_star, 'tab:blue'),
          ('SED candidates', sample['candidate'], 'crimson')]
for label, group, color in groups:
    for is_limit, marker in [(False, 'o'), (True, '^')]:
        shown = group & has_colors & (z_is_limit == is_limit)
        ax.scatter(sample['Y-J'][shown], sample['z-Y'][shown], s=18, marker=marker, color=color,
                   label=None if is_limit else label)

# The color cuts: the selected region is above the lower boundary and left of Y - J = 0.5.
y_minus_j = np.array([-1.5, 0.035, 0.5])
lower_boundary = np.maximum(1.14, 4 * y_minus_j + 1)
ax.fill_between(y_minus_j, lower_boundary, 6, color='gold', alpha=0.25, linewidth=0)
ax.plot(y_minus_j, lower_boundary, color='black', linewidth=1)
ax.plot([0.5, 0.5], [3, 6], color='black', linewidth=1)
ax.text(-1.4, 4.75, 'color selection of z = 6 to 8 galaxies\n(Allen et al. 2026)', va='top', fontsize=9)
ax.set_xlim(-1.5, 2.5)
ax.set_ylim(-1, 5)
ax.set_xlabel('$Y_E - J_E$ (mag)')
ax.set_ylabel('$z - Y_E$ (mag)')
ax.legend(frameon=False, loc='lower right')
plt.show()
```

**Figure 4.** Colors of the sources with S/N > 2 in $Y_E$.
Triangles are lower limits on $z - Y_E$, for sources that are not detected in the HSC $z$ band.
The shaded region is the color selection of Allen et al. (2026) for galaxies at z = 6 to 8: a strong break ($z - Y_E$, vertical axis) and a blue color redward of it ($Y_E - J_E$, horizontal axis).

No source of our small field passes the color cuts.
The SED candidates are outside the selected region because they are red in $Y_E - J_E$.
This is where brown dwarfs and red galaxies at lower redshift are, and the color cut is made to remove them.
A red $Y_E - J_E$ color can also be that of a galaxy at z > 7, for which the break is inside the $Y_E$ band, so the color cut removes these too: this is one reason why Allen et al. prefer the SED fit, which uses all bands.
But the SED fit can only separate the two kinds of sources when it has bands redward of $H_E$, and the catalog has none in this field.
Allen et al. find in their simulations that the contamination by brown dwarfs is much higher when the optical or the IRAC bands are missing (their Appendix A).

## 6. Inspect the candidates and add IRAC with forced photometry

Allen et al. (2026) inspected the images of their Q1 candidates and found that most were artifacts, for example persistence and ghosts in the near-infrared images.
They also show that Spitzer/IRAC fluxes at 3.6 and 4.5 μm are needed to separate high-redshift galaxies from brown dwarfs and from red galaxies at lower redshift: these sources are much brighter in IRAC than a young galaxy at z > 6.

The DR1 catalog has no IRAC fluxes in EDF-N.
We measure them with forced photometry, using the `euclid_phot` package: it fits a model of every source in a small field to the pixels of each image, with the positions and shapes fixed and only the fluxes free.
Sources that are blended in the IRAC images, which have a PSF of about 2 arcsec, are fit together.
See the [forced photometry notebooks](forced_photometry/README.md) for the method.

Two settings matter for dropouts:

- The source models are built on the $H_E$ image (`prior={'band': 'H', ...}`) and not on the VIS image, the default, because the candidates are not detected in VIS.
- The source list is the MER catalog (`'objects': 'mer'`), so the neighbors of each candidate are modeled too.

The IRAC images are the mosaics of the Cosmic Dawn Survey (`mosaic_backend='dawn'`), which cover about half of EDF-N.

We fit a small field of 20 arcsec around each candidate (`FOLLOWUP_SIZE`).
The Euclid Deep Fields are crowded: a field of this size has about 20 MER sources, and one of 40 arcsec has about 100.
The fit of a small field is fast and needs little memory.
It also has less risk that the shape of one source is fit badly: in a larger field we had one case in which the model of a source grew very large and the IRAC fit ran out of memory.
The price is in the IRAC fluxes.
Sources outside the field are not modeled, and the IRAC PSF spreads some of their light into the field, so the IRAC fluxes of our candidates are about 5% higher than in a fit of 40 arcsec.
The fluxes in the Euclid bands are the same in both.
This is accurate enough here, where the question is whether a candidate is one or two magnitudes brighter in IRAC than in $H_E$.

The function tries the fit with IRAC twice, because the archive services sometimes time out, and then without IRAC, for the positions that have no IRAC mosaic.

```{code-cell} ipython3
def measure_field(ra, dec):
    """Forced photometry of all MER sources in a small field, in the Euclid bands and IRAC channels 1 and 2.

    Parameters
    ----------
    ra, dec : float
        Center of the field in degrees.

    Returns
    -------
    euclid_phot.ForcedPhotometryResult or None
        The fit. If the position has no IRAC mosaic, the result has the Euclid bands only.
        None if all attempts failed.
    """
    with_irac = {'target_bands': {'euclid': ('VIS', 'Y', 'J'), 'irac': ('IRAC1', 'IRAC2')},
                 'irac_options': {'mosaic_backend': 'dawn'}}
    euclid_only = {'target_bands': {'euclid': ('VIS', 'Y', 'J')}}
    for attempt in (with_irac, with_irac, euclid_only):
        try:
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                return ep.run_forced_photometry(ra, dec, FOLLOWUP_SIZE, prior={'band': 'H', 'objects': 'mer'},
                                                n_workers=4, verbose=False, **attempt)
        except Exception as error:
            print(f"Fit at ({ra:.4f}, {dec:.4f}) with bands {attempt['target_bands']} failed: {error}")
    return None
```

Each field takes about half a minute. We follow up `N_FOLLOWUP` candidates: the ones at the highest redshifts.
Increase `N_FOLLOWUP` to look at more.

```{code-cell} ipython3
followup = candidates[np.argsort(candidates['z_phot'])[::-1][:N_FOLLOWUP]]
followup['object_id', 'ra', 'dec', 'z_phot', 'mag_h', 'chi2_galaxy', 'chi2_star'].pprint_all()

fields = [measure_field(source['ra'], source['dec']) for source in followup]
```

The result of each field is a table with one row per MER source.
We take the row of the candidate, and add its forced fluxes to the table of the candidates.

```{code-cell} ipython3
FORCED_BANDS = ['VIS', 'Y', 'J', 'H', 'IRAC1', 'IRAC2']
for band in FORCED_BANDS:
    followup[f'forced_flux_{band}'] = np.nan
    followup[f'forced_error_{band}'] = np.nan

for row, field in zip(followup, fields):
    photometry = field.to_table(ebv=None) if field is not None else None
    match = photometry[photometry['object_id'] == row['object_id']] if field is not None else []
    if len(match) == 0:
        # The fit failed, or it dropped the source, for example at the edge of a mask.
        print(f"{row['object_id']} has no forced photometry")
        continue
    for band in FORCED_BANDS:
        if f'flux_{band}_ujy' in match.colnames:
            row[f'forced_flux_{band}'] = match[f'flux_{band}_ujy'][0]
            row[f'forced_error_{band}'] = match[f'flux_err_{band}_ujy'][0]

if not np.any(np.isfinite(followup['forced_flux_H'])):
    raise RuntimeError('The forced photometry failed for all candidates: see the messages above.')

print('S/N of the forced fluxes')
print('object_id            ' + ''.join(f'{band:>8s}' for band in FORCED_BANDS))
for row in followup:
    values = ''.join(f"{row[f'forced_flux_{band}'] / row[f'forced_error_{band}']:8.1f}" for band in FORCED_BANDS)
    print(f"{row['object_id']}  {values}")
```

Now we fit the SEDs again, with the IRAC fluxes.
The forced fluxes are total fluxes of the source models, and the catalog fluxes of Section 3 are aperture fluxes, so the two cannot be mixed directly.
We use the forced fluxes for the Euclid and IRAC bands.
For the HSC bands we keep the catalog aperture fluxes and multiply them by the ratio of the forced and the aperture flux in $H_E$, which puts them on the scale of the total fluxes.

```{code-cell} ipython3
aperture_to_total = followup['forced_flux_H'] / as_float_array(followup['flux_h_2fwhm_aper'])

fluxes_irac, errors_irac = {}, {}
for band in ['g_ext_hsc', 'r_ext_hsc', 'i2_ext_hsc', 'z_ext_hsc']:
    fluxes_irac[band] = as_float_array(followup[f'flux_{band}_2fwhm_aper']) * aperture_to_total
    errors_irac[band] = as_float_array(followup[f'fluxerr_{band}_2fwhm_aper']) * aperture_to_total
for band, forced_band in [('vis', 'VIS'), ('y', 'Y'), ('j', 'J'), ('h', 'H'), ('irac1', 'IRAC1'), ('irac2', 'IRAC2')]:
    fluxes_irac[band] = followup[f'forced_flux_{forced_band}']
    errors_irac[band] = followup[f'forced_error_{forced_band}']

# Filters 18 and 19 of EAZY are IRAC channels 1 (3.6 μm) and 2 (4.5 μm).
filter_numbers_irac = dict(filter_numbers, irac1=18, irac2=19)
fit_irac = fit_seds(fluxes_irac, errors_irac, filter_numbers_irac, 'highz_followup', ebv=field_ebv)
results_irac = summarize_fit(fit_irac)
```

We also need the rows of these candidates in the first fit, to compare the two.

```{code-cell} ipython3
sample_index = {object_id: index for index, object_id in enumerate(sample['object_id'])}
first_fit_index = [sample_index[object_id] for object_id in followup['object_id']]

print(f"{'object_id':20s} {'':10s} {'z_phot':>7s} {'chi2_gal':>9s} {'chi2_star':>10s} {'P(z<5)':>7s}")
for index, row in enumerate(followup):
    print(f"{row['object_id']!s:20s} {'catalog':10s} {row['z_phot']:7.2f} {row['chi2_galaxy']:9.1f} "
          f"{row['chi2_star']:10.1f} {row['p_low']:7.2f}")
    with_irac = results_irac[index]
    print(f"{'':20s} {'with IRAC':10s} {with_irac['z_phot']:7.2f} {with_irac['chi2_galaxy']:9.1f} "
          f"{with_irac['chi2_star']:10.1f} {with_irac['p_low']:7.2f}")
```

For each candidate we plot image stamps in the Euclid and IRAC bands, the two SED fits, and the two p(z).

```{code-cell} ipython3
pdf_catalog = redshift_pdf(fit.lnp, fit.zgrid)
pdf_irac = redshift_pdf(fit_irac.lnp, fit_irac.zgrid)
stamp_labels = {'VIS': '$I_E$', 'Y': '$Y_E$', 'J': '$J_E$', 'H': '$H_E$', 'IRAC1': '3.6 μm', 'IRAC2': '4.5 μm'}

fig = plt.figure(figsize=(17, 3.1 * len(followup)), layout='constrained')
grid = fig.add_gridspec(len(followup), 10, width_ratios=[1] * 6 + [0.25, 3.2, 0.15, 2])
for row_index, (row, field) in enumerate(zip(followup, fields)):
    # Image stamps. A band that was not measured is left empty.
    for column, band in enumerate(FORCED_BANDS):
        ax = fig.add_subplot(grid[row_index, column])
        try:
            images = field.image_set(band)
        except (AttributeError, ValueError, KeyError):
            ax.axis('off')
            continue
        show_stamp(ax, images['data'], images['wcs'], row['ra'], row['dec'], STAMP_SIZE, stamp_labels[band])
        if column == 0:
            ax.set_ylabel(f"{row['object_id']}", fontsize=8)

    # SEDs: the catalog fit (blue) and the fit with IRAC (red, with its photometry).
    ax = fig.add_subplot(grid[row_index, 7])
    with contextlib.redirect_stdout(io.StringIO()), warnings.catch_warnings():
        warnings.simplefilter('ignore')
        sed_catalog = fit.show_fit(first_fit_index[row_index], id_is_idx=True, show_fnu=1, get_spec=True)
        sed_irac = fit_irac.show_fit(row_index, id_is_idx=True, show_fnu=1, get_spec=True)
    plot_sed(ax, sed_catalog, 'tab:blue', f"catalog bands, z = {row['z_phot']:.1f}", show_data=False)
    plot_sed(ax, sed_irac, 'crimson', f"with IRAC, z = {results_irac['z_phot'][row_index]:.1f}")
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlim(0.4, 6)
    ax.set_ylim(0.3 * np.nanmin(np.asarray(sed_irac['efobs'])), 5 * np.nanmax(np.asarray(sed_irac['model'])))
    ax.set_xticks([0.5, 1, 2, 4], ['0.5', '1', '2', '4'])
    ax.set_ylabel('flux (μJy)')
    ax.legend(frameon=False, fontsize=8, loc='upper left')

    # Redshift probability distributions.
    ax_pdf = fig.add_subplot(grid[row_index, 9])
    ax_pdf.plot(fit.zgrid, pdf_catalog[first_fit_index[row_index]], color='tab:blue')
    ax_pdf.plot(fit_irac.zgrid, pdf_irac[row_index], color='crimson')
    ax_pdf.set_yticks([])
    ax_pdf.set_ylabel('p(z)')
    if row_index == len(followup) - 1:
        ax.set_xlabel('observed wavelength (μm)')
        ax_pdf.set_xlabel('redshift')
plt.show()
```

**Figure 5.** The candidates that were followed up, one per row.
Left: stamps of 8 arcsec in the Euclid bands and in IRAC channels 1 and 2, centered on the candidate.
Middle: the photometry used in the fit with IRAC (black points; arrows are 2σ upper limits), with the best galaxy fit to the catalog bands (blue) and to the catalog bands and IRAC (red). The open squares are the fluxes of each fit in the bands.
Right: p(z) of the two fits, in the same colors.

The three sources are real: each is seen in $J_E$ and $H_E$ at the same position and is not an image artifact.
All three are also bright in IRAC, 1 to 2 magnitudes brighter than in $H_E$, and this changes the fits:

- The source in the second row is isolated. With IRAC its best fit moves to z ≈ 2.4, and only a small part of p(z) stays near z = 9. It is most likely a red galaxy at lower redshift.
- The source in the third row moves to z ≈ 2.2. Here the forced photometry also measures flux in $I_E$ and $Y_E$ at 7σ, which the small aperture of the catalog did not, so it is not a dropout.
- The source in the first row stays at z ≈ 8, but the galaxy fit is worse than before and needs a spectrum that rises steeply from $H_E$ to IRAC. Young galaxies at these redshifts are blue. A brighter neighbor is 2 arcsec away, and the two are blended in the IRAC images, so the IRAC flux of the candidate depends on how well the model separates them. Allen et al. report a possible contamination of this kind for one of their two Q1 candidates.

The star templates fit none of the three once IRAC is included ($\chi^2_{\rm star}$ of several hundred in the table above), so they are not brown dwarfs.
None of the three is a convincing z > 6 galaxy.
We chose them for their high redshifts, where contaminants are a larger fraction of the candidates, so this does not mean that all candidates are contaminants.

## Summary

- The DR1 MER catalog can be read directly from Parquet files in the cloud. In less than a minute this gives the sources in 0.2 square degrees of EDF-N that are detected in $J_E$ and $H_E$ and not in $I_E$. The catalog has the Euclid and HSC fluxes in matched apertures, which is what an SED fit needs.
- Of about 120 such sources, about a dozen are not detected in the optical, are best fit by a galaxy at z ≥ 6, and are fit worse by a star.
- This is several times more than the number of z > 6 galaxies expected at these magnitudes. With the Euclid and optical bands only, the fits cannot separate high-redshift galaxies from brown dwarfs and from red galaxies at lower redshift.
- Forced photometry on the IRAC mosaics adds the two bands that can. For the three candidates we followed up, the IRAC fluxes argue against z > 6.

This notebook shows the method, and its candidates are not a sample for science.
An analysis like that of Allen et al. (2026) needs in addition:

- image stamps and IRAC photometry of all candidates, not of three;
- the masks around bright stars and the checks for persistence and ghosts in the near-infrared images;
- total fluxes in all bands, measured in the same way;
- a second SED fitting code and more complete sets of brown dwarf templates, and stricter cuts on p(z);
- simulations of the completeness and the contamination of the selection, as in the first part of their paper;
- spectroscopy of the best candidates.

To go further, you can use a larger search circle (IRAC covers about half of EDF-N), follow up more candidates (`N_FOLLOWUP`), or ask for no detection in $Y_E$ to select z > 8 as Allen et al. did for Q1.

## References

**Method**

- Euclid Collaboration: Allen, N., Oesch, P. A., Bowler, R. A. A., et al. (2026). *Euclid Quick Data Release (Q1): Hunting for luminous z > 6 galaxies in the Euclid Deep Fields – forecasts and first bright detections*. A&A, 711, A25. [arXiv:2511.02926](https://arxiv.org/abs/2511.02926).

**Data**

- Euclid Collaboration: Romelli, E., et al. (2025). *Euclid Quick Data Release (Q1): From images to multiwavelength catalogues: the Euclid MERge Processing Function*. [arXiv:2503.15305](https://arxiv.org/abs/2503.15305).
- Euclid Collaboration: McPartland, C. J. R., et al. (2025). *Euclid preparation: The Cosmic Dawn Survey (DAWN) of the Euclid Deep and Auxiliary Fields*. A&A.

**Software**

- Brammer, G. B., van Dokkum, P. G., & Coppi, P. (2008). *EAZY: A Fast, Public Photometric Redshift Code*. ApJ, 686, 1503.
- Allard, F., Homeier, D., & Freytag, B. (2012). *Models of very-low-mass stars, brown dwarfs and exoplanets* (BT-Settl). Phil. Trans. R. Soc. A, 370, 2765.
- Lang, D., Hogg, D. W., & Mykytyn, D. (2016). *The Tractor: Probabilistic astronomical source detection and measurement*. [ascl:1604.008](https://ascl.net/1604.008).

## About This Notebook

**Last updated:** 12 November 2026

This notebook was written and contributed by its authors.
When last checked, it ran without error in an environment set up with the install instructions above.
However, it is not regularly tested or maintained.

To submit a bug report or ask a question, please open an [issue](https://github.com/Caltech-IPAC/irsa-contributed-notebooks/issues).

### Acknowledgements

This tutorial uses Euclid DR1 data products distributed by the NASA/IPAC Infrared Science Archive (IRSA), which is operated by the California Institute of Technology under contract with the National Aeronautics and Space Administration.
Euclid is a European Space Agency mission with contributions from NASA.
The Spitzer/IRAC mosaics are from the Cosmic Dawn Survey.

This notebook uses the Python packages NumPy, Matplotlib, Astropy, PyVO, eazy-py, Tractor and euclid_phot.

The AI assistant Claude (Anthropic) was used to draft the notebook.
