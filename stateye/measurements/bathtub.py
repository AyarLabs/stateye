import numpy as np
import matplotlib.pyplot as plt
from scipy.special import erfinv, erf
from typing import Union
from tqdm import tqdm
import math

"""
Functions for generating 2D bathtub plots from eye diagrams.
Uses a dual-dirac model for histograms filtered by data-pattern
(so DDJ should be factored out if the pattern length is long
enough).

Ayar Labs
Derek M. Kita
"""


def to_q_scale(ber: Union[np.ndarray, float], rho_t: float):
    return np.sqrt(2) * erfinv(1 - ber / rho_t)


def to_ber_scale(q: Union[np.ndarray, float], rho_t: float):
    return rho_t * (1 - erf(q / np.sqrt(2)))


def generate_vertical_bathtub(
    hist: np.ndarray,
    bathtub: np.ndarray,
    raw_bathtub: np.ndarray,
    pattern_indices: np.ndarray,
    pattern_counts: np.ndarray,
    data_values: np.ndarray,
    sensitivity: float,
    y_scale: np.ndarray,
) -> np.ndarray:
    bathtub[:] = 0.0  # re-initialize to zero
    raw_bathtub[:] = 0.0  # re-initialize to zero
    roll_idx = int(round((sensitivity / 2) / (y_scale[1] - y_scale[0])))
    for i, idx in tqdm(enumerate(pattern_indices)):
        hist2d = hist[:, :, idx]
        bit_value = data_values[idx]
        transition_density = pattern_counts[i] / np.sum(pattern_counts)
        for sidx in range(hist.shape[0]):
            if bit_value == 0:  # 0-level
                bathtub[sidx, :] += np.flip(
                    q_scale_fit(
                        input_scale=np.flip(y_scale),
                        hist1d=np.flip(hist2d[sidx, :]),
                        rho_t=transition_density,
                        mean_shift=sensitivity / 2,
                    )
                )
                cdf = np.cumsum(np.flip(hist2d[sidx, :]))
                cdf *= transition_density / np.max(cdf)
                raw_bathtub[sidx, :] += np.roll(np.flip(cdf), roll_idx)
            elif bit_value == 1:  # 1-level
                bathtub[sidx, :] += q_scale_fit(
                    input_scale=y_scale,
                    hist1d=hist2d[sidx, :],
                    rho_t=transition_density,
                    mean_shift=-sensitivity / 2,
                )
                cdf = np.cumsum(hist2d[sidx, :])
                cdf *= transition_density / np.max(cdf)
                raw_bathtub[sidx, :] += np.roll(cdf, -roll_idx)
            else:
                raise NotImplementedError("PAM4 not yet supported :(")


def q_scale_fit(
    input_scale: np.ndarray,
    hist1d: np.ndarray,
    rho_t: float,
    mean_shift: float,
) -> dict:
    mask = np.nonzero(hist1d)
    scale = input_scale[mask]
    cdf = np.cumsum(hist1d[mask])
    cdf *= rho_t / np.max(cdf)

    qv = to_q_scale(cdf, rho_t=0.5 * rho_t)

    mask = np.isfinite(qv)
    qv = qv[mask]
    cdf = cdf[mask]
    scale = scale[mask]

    if len(qv) <= 1:
        # No q-scale values could be found
        ber = np.ones(len(input_scale)) * rho_t
        if math.isclose(np.sum(hist1d), 0):
            # No data.  Just return no bit errors.
            return ber
        else:
            # Some data, just make a sharp cutoff.
            idx = np.where(hist1d != 0)[0][0]
            ber[:idx] = 0
            return ber
    else:
        # Fit, scale = qv*sigma + mean
        sigma, mean = np.polyfit(qv, scale, deg=1)

        dx = abs(input_scale[1] - input_scale[0])
        gaussian = np.exp(
            -((input_scale - (mean + mean_shift)) ** 2) / (2 * sigma**2)
        )
        gaussian /= abs(sigma) * np.sqrt(2 * np.pi)
        ber = rho_t * np.cumsum(gaussian) * dx
        return ber
