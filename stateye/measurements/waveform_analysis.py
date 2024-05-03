from .utilities import (
    get_wvf_pts,
    filter,
    compute_abs_edge_time,
    compute_overshoot,
    compute_undershoot,
)
import numpy as np
from .timer import timer
from typing import Tuple

"""
Functions for analyzing time series waveforms prior to adding them to the
2D eye histogram.
"""

# assumes rise value bit is 0, fall value bit is 1
OMA_FILTER_MAP = {
    "xp": {
        "fbefore_rise": [0,],
        "fafter_rise": [1, 1],
        "fbefore_fall": [1,],
        "fafter_fall": [0, 0]
    },
    "4140": {
        "fbefore_rise": [0]*3,
        "fafter_rise": [1]*4,
        "fbefore_fall": [1]*3,
        "fafter_fall": [0]*4
    },
    "8180": {
        "fbefore_rise": [0]*7,
        "fafter_rise": [1]*8,
        "fbefore_fall": [1]*7,
        "fafter_fall": [0]*8
    },
}

def nrz_waveform_analysis(
    time: np.ndarray,
    wvf: np.ndarray,
    sampling_indices: np.ndarray,
    initial_threshold: float,
    period: float,
) -> Tuple[dict, dict]:
    """
    Perform all standard waveform measurements for a NRZ signal
    """
    msmts = {}  # keep track of measured values
    counts = {}  # keep track of # of measurements observed for each value

    # Compute and store data amplitudes (wvf_s), and binary values (data)
    wvf_sampled = get_wvf_pts(sampling_indices, wvf)

    midpt_sampling_indices = (sampling_indices[1:] + sampling_indices[:-1]) / 2
    wvf_sampled_midpts = get_wvf_pts(midpt_sampling_indices, wvf)
    data = (wvf_sampled > initial_threshold).astype(int)
    compute_zero_at_crossing(wvf_sampled_midpts, data, msmts, counts)
    compute_one_at_crossing(wvf_sampled_midpts, data, msmts, counts)
    msmts["threshold"] = 0.5 * (msmts["one_level_xp"] + msmts["zero_level_xp"])
    counts["threshold"] = min([counts["zero_level_xp"], counts["one_level_xp"]])

    # Re-compute data, one-level, and zero-level with newer sampling threshold
    data = (np.array(wvf_sampled) > msmts["threshold"]).astype(int)
    compute_zero_at_crossing(wvf_sampled_midpts, data, msmts, counts)
    compute_one_at_crossing(wvf_sampled_midpts, data, msmts, counts)
    msmts["threshold"] = 0.5 * (msmts["one_level_xp"] + msmts["zero_level_xp"])
    counts["threshold"] = min([counts["zero_level_xp"], counts["one_level_xp"]])

    msmts["oma_xp"] = msmts["one_level_xp"] - msmts["zero_level_xp"]
    counts["oma_xp"] = counts["threshold"]
    msmts["extinction_ratio_xp"] = 10 * np.log10(msmts["one_level_xp"] / msmts["zero_level_xp"])
    counts["extinction_ratio_xp"] = counts["oma_xp"]

    msmts["average"], counts["average"] = np.mean(wvf), len(wvf)

    sps = period / (time[1] - time[0])
    compute_x1x0_oma(wvf_sampled, data, wvf, sampling_indices, sps, msmts, counts, pattern_length=8)
    compute_x1x0_oma(wvf_sampled, data, wvf, sampling_indices, sps, msmts, counts, pattern_length=4)
    compute_edge_statistics(msmts, counts, time, wvf, sampling_indices, data, period)

    return msmts, counts


@timer
def compute_zero_at_crossing(
    wvf_at_crossing: np.ndarray,
    data: np.ndarray,
    msmts: dict,
    counts: dict,
) -> None:
    fb = np.array([0], dtype=int)
    fa = np.array([], dtype=int)
    zero_matches = filter(0, fbefore=fb, fafter=fa, data=data)[1:]
    if np.sum(zero_matches) == 0:
        msmts["zero_level_xp"], counts["zero_level_xp"] = np.nan, 0
    else:
        msmts["zero_level_xp"] = np.dot(zero_matches, wvf_at_crossing) / np.sum(
            zero_matches
        )
        counts["zero_level_xp"] = np.sum(zero_matches)


@timer
def compute_one_at_crossing(
    wvf_at_crossing: np.ndarray,
    data: np.ndarray,
    msmts: dict,
    counts: dict,
) -> None:
    fb = np.array([1], dtype=int)
    fa = np.array([], dtype=int)
    one_matches = filter(1, fbefore=fb, fafter=fa, data=data)[1:]
    if np.sum(one_matches) == 0:
        msmts["one_level_xp"], counts["one_level_xp"] = np.nan, 0
    else:
        msmts["one_level_xp"] = np.dot(one_matches, wvf_at_crossing) / np.sum(one_matches)
        counts["one_level_xp"] = np.sum(one_matches)


@timer
def compute_x1x0_oma(
    wvf_sampled: np.ndarray,
    data: np.ndarray,
    wvf: np.ndarray, 
    sampling_indices: np.ndarray,
    sps: float,
    msmts: dict,
    counts: dict,
    pattern_length: int,
) -> None:
    """
    Compute the OMA for a given pattern length.  This shall be computed in a way consistent with
    Section 68.6.2 - Optical Modulation Amplitude (OMA) from IEEE 802.3

    'For the purposes of Clause 68, OMA is defined by the measurement method given in 52.9.5, 
    and as illustrated in Figure 68–4. The mean logic ONE and mean logic ZERO values are measured 
    over the center 20% of the two time intervals of the square wave. The OMA is the difference 
    between these two means.'

    From Section 52.9.5:
    'Measure the mean optical power P1 of the logic “1” as defined over the center 20% of the time 
    interval where the signal is in the high state. (See Figure 52–6.)'
    'Measure the mean optical power P0 of the logic “0” as defined over the center 20% of the time 
    interval where the signal is in the low state. (See Figure 52–6.)'

    From 58.7.6:
    OMA = P1 - P0
    ER = P1 / P0 (or 10 log10(P1/P0) in dB)
    Pmean = (P0 + P1)/2
    """
    pstr = f"{pattern_length}1{pattern_length}0"
    oma_key = "oma_" + pstr
    zero_key = "zero_level_" + pstr
    one_key = "one_level_" + pstr
    er_key = "extinction_ratio_" + pstr
    one_matches = filter(
        1,
        fbefore=np.array([1] * (pattern_length - 1), dtype=int),
        fafter=np.array([], dtype=int),
        data=data,
    )
    zero_matches = filter(
        0,
        fbefore=np.array([0] * (pattern_length - 1), dtype=int),
        fafter=np.array([], dtype=int),
        data=data,
    )
    one_matches = np.asarray(one_matches).astype(bool)
    zero_matches = np.asarray(zero_matches).astype(bool)
    wvf_sampled = np.asarray(wvf_sampled)

    # Find all sections of ones and zeros of length >= pattern_length.
    # First group into contiguous sections (results of filter() can overlap)
    def get_contiguous_segments(matches):
        segment_start, segment_stop = [], []
        for i, _ in enumerate(matches[1:-1]):
            if matches[i-1]==False and matches[i]==True:
                segment_start += [i]
            if matches[i]==True and matches[i+1]==False:
                segment_stop += [i]
        return segment_start, segment_stop, min([len(segment_start), len(segment_stop)])
    
    one_segment_starts, one_segment_stops, num_one_segments = get_contiguous_segments(one_matches)
    zero_segment_starts, zero_segment_stops, num_zero_segments = get_contiguous_segments(zero_matches)

    one_values = []
    for i in range(num_one_segments):
        start_idx = sampling_indices[one_segment_starts[i]] - (pattern_length - 0.5)*sps
        stop_idx = sampling_indices[one_segment_stops[i]] + 0.5*sps

        start_idx_center_20 = round(start_idx + 0.4 * (stop_idx - start_idx))
        stop_idx_center_20 = round(start_idx + 0.6 * (stop_idx - start_idx))
        one_values += wvf[start_idx_center_20:stop_idx_center_20].tolist()

    zero_values = []
    for i in range(num_zero_segments):
        start_idx = sampling_indices[zero_segment_starts[i]] - (pattern_length - 0.5)*sps
        stop_idx = sampling_indices[zero_segment_stops[i]] + 0.5*sps

        start_idx_center_20 = round(start_idx + 0.4 * (stop_idx - start_idx))
        stop_idx_center_20 = round(start_idx + 0.6 * (stop_idx - start_idx))
        zero_values += wvf[start_idx_center_20:stop_idx_center_20].tolist()

    if (num_one_segments== 0) or (num_zero_segments == 0):
        msmts[oma_key], counts[oma_key] = np.nan, 0
        msmts[one_key], counts[one_key] = np.nan, 0
        msmts[zero_key], counts[zero_key] = np.nan, 0
        msmts[er_key], counts[er_key] = np.nan, 0
    else:
        p0, p1 = np.mean(zero_values), np.mean(one_values)
        msmts[zero_key] = p0
        counts[zero_key] = num_zero_segments
        msmts[one_key] = p1
        counts[one_key] = num_one_segments
        msmts[oma_key] = p1 - p0
        counts[oma_key] = min([num_zero_segments, num_one_segments])
        if p0 == 0:
            msmts[er_key] = np.nan
        else:
            msmts[er_key] = 10 * np.log10(p1 / p0)
        counts[er_key] = counts[oma_key]


@timer
def compute_edge_statistics(
    msmts: dict,
    counts: dict,
    time: np.ndarray,
    wvf: np.ndarray,
    sampling_indices: np.ndarray,
    data: np.ndarray,
    period: float,
):
    for oma_type in ["xp", "4140", "8180"]:
        edge_matches_rise = filter(
            0, 
            fbefore=np.array(OMA_FILTER_MAP[oma_type]["fbefore_rise"], dtype=int), 
            fafter=np.array(OMA_FILTER_MAP[oma_type]["fafter_rise"], dtype=int), 
            data=data,
        )
        rising_edge_50, rising_counts = compute_abs_edge_time(
            pattern_matches=edge_matches_rise,
            threshold=msmts["threshold"],
            s_idx=sampling_indices,
            t=time,
            y=wvf,
            period=period,
        )
        edge_matches_fall = filter(
            1, 
            fbefore=np.array(OMA_FILTER_MAP[oma_type]["fbefore_fall"], dtype=int), 
            fafter=np.array(OMA_FILTER_MAP[oma_type]["fafter_fall"], dtype=int), 
            data=data,
        )
        falling_edge_50, falling_counts = compute_abs_edge_time(
            pattern_matches=edge_matches_fall,
            threshold=msmts["threshold"],
            s_idx=sampling_indices,
            t=time,
            y=wvf,
            period=period,
        )
        if (rising_counts == 0) or (falling_counts == 0):
            t_center = np.nan
            msmts[f"abs_rise_time_50_{oma_type}"], counts[f"abs_rise_time_50_{oma_type}"] = np.nan, 0
            msmts[f"abs_fall_time_50_{oma_type}"], counts[f"abs_fall_time_50_{oma_type}"] = np.nan, 0
            msmts[f"dcd_{oma_type}"], counts[f"dcd_{oma_type}"] = np.nan, 0
        else:
            t_center = (rising_edge_50 + falling_edge_50) / 2
            msmts[f"abs_rise_time_50_{oma_type}"] = rising_edge_50 - t_center
            counts[f"abs_rise_time_50_{oma_type}"] = rising_counts
            msmts[f"abs_fall_time_50_{oma_type}"] = falling_edge_50 - t_center
            counts[f"abs_fall_time_50_{oma_type}"] = falling_counts
            msmts[f"dcd_{oma_type}"] = rising_edge_50 - falling_edge_50
            counts[f"dcd_{oma_type}"] = min([rising_counts, falling_counts])

        for t in [10, 20, 80, 90]:
            cur_rising_edge, cur_rising_counts = compute_abs_edge_time(
                pattern_matches=edge_matches_rise,
                threshold=msmts[f"zero_level_{oma_type}"] + (t/100) * msmts[f"oma_{oma_type}"],
                s_idx=sampling_indices,
                t=time,
                y=wvf,
                period=period,
            )
            if (cur_rising_counts == 0) or np.isnan(t_center):
                msmts[f"abs_rise_time_{t}_{oma_type}"], counts[f"abs_rise_time_{t}_{oma_type}"] = np.nan, 0
            else:
                msmts[f"abs_rise_time_{t}_{oma_type}"] = cur_rising_edge - t_center
                counts[f"abs_rise_time_{t}_{oma_type}"] = cur_rising_counts

            cur_falling_edge, cur_falling_counts = compute_abs_edge_time(
                pattern_matches=edge_matches_fall,
                threshold=msmts[f"zero_level_{oma_type}"] + (t/100) * msmts[f"oma_{oma_type}"],
                s_idx=sampling_indices,
                t=time,
                y=wvf,
                period=period,
            )
            if (cur_falling_counts == 0) or np.isnan(t_center):
                msmts[f"abs_fall_time_{t}_{oma_type}"], counts[f"abs_fall_time_{t}_{oma_type}"] = np.nan, 0
            else:
                msmts[f"abs_fall_time_{t}_{oma_type}"] = cur_falling_edge - t_center
                counts[f"abs_fall_time_{t}_{oma_type}"] = cur_falling_counts
        
        for edge, tstart, tend in [("rise", 10, 90), ("rise", 20, 80), ("fall", 90, 10), ("fall", 80, 20)]:
            t1_n = f"abs_{edge}_time_{tstart}_{oma_type}"
            t2_n = f"abs_{edge}_time_{tend}_{oma_type}"
            msmt_name = f"{edge}_time_{tstart}-{tend}_{oma_type}"
            if (counts[t1_n] == 0) or (counts[t2_n] == 0):
                msmts[msmt_name], counts[msmt_name] = np.nan, 0
            else:
                msmts[msmt_name] = msmts[t2_n] - msmts[t1_n]
                counts[msmt_name] = min([counts[t1_n], counts[t2_n]])

        # Overshoot
        over_t, over_y, over_counts = compute_overshoot(
            pattern_matches=edge_matches_rise,
            s_idx=sampling_indices,
            t=time,
            y=wvf,
            period=period,
        )
        if (rising_counts == 0) or (over_counts == 0) or np.isnan(t_center):
            msmts[f"overshoot_percentage_{oma_type}"], counts[f"overshoot_percentage_{oma_type}"] = np.nan, 0
            msmts[f"abs_overshoot_time_{oma_type}"], counts[f"abs_overshoot_time_{oma_type}"] = np.nan, 0
        else:
            msmts[f"overshoot_percentage_{oma_type}"] = (
                100 * (over_y - msmts[f"one_level_{oma_type}"]) / msmts[f"oma_{oma_type}"]
            )
            counts[f"overshoot_percentage_{oma_type}"] = min([rising_counts, over_counts])
            msmts[f"abs_overshoot_time_{oma_type}"] = over_t - t_center
            counts[f"abs_overshoot_time_{oma_type}"] = min([rising_counts, over_counts])

        # Undershoot
        under_t, under_y, under_counts = compute_undershoot(
            pattern_matches=edge_matches_fall,
            s_idx=sampling_indices,
            t=time,
            y=wvf,
            period=period,
        )
        if (falling_counts == 0) or (under_counts == 0) or np.isnan(t_center):
            msmts[f"undershoot_percentage_{oma_type}"], counts[f"undershoot_percentage_{oma_type}"] = np.nan, 0
            msmts[f"abs_undershoot_time_{oma_type}"], counts[f"abs_undershoot_time_{oma_type}"] = np.nan, 0
        else:
            msmts[f"undershoot_percentage_{oma_type}"] = (
                100 * (msmts[f"zero_level_{oma_type}"] - under_y) / msmts[f"oma_{oma_type}"]
            )
            counts[f"undershoot_percentage_{oma_type}"] = min([rising_counts, under_counts])
            msmts[f"abs_undershoot_time_{oma_type}"] = under_t - t_center
            counts[f"abs_undershoot_time_{oma_type}"] = min([rising_counts, under_counts])