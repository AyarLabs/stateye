import numpy as np
import math
from scipy.signal import butter
from scipy.signal import freqs as gen_freqs


def generate_half_sine_data(
    datarate_gbps,
    samples_per_symbol,
    nbits,
    amplitude,
    rise_fall_time_sec,
    add_noise=False,
):
    sequence = np.random.choice([0, 1], nbits - 8*8)
    sequence = np.concatenate((sequence, np.tile(np.array([1]*8+[0]*8, dtype=int), 4)))
    wvf = np.repeat(amplitude * sequence, samples_per_symbol).astype(float)

    symbol_time = 1 / (datarate_gbps * 1e9)
    dt_sec = symbol_time / samples_per_symbol

    wvf = shape_edges_half_sine(
        wvf,
        rise_fall_time=rise_fall_time_sec,
        dt=dt_sec,
        samples_per_symbol=samples_per_symbol,
        symbol_time=symbol_time,
    )

    if add_noise:
        wvf += np.random.normal(
            scale=0.05 * amplitude, size=nbits * samples_per_symbol
        )  # Add some noise

    return wvf


def filter_waveform(wvf: np.ndarray, time: np.ndarray, bw_3db_Hz: float, npoles: int, filter_func = butter):
    freq = np.fft.rfftfreq(time.shape[-1], d=time[1] - time[0])
    b, a = filter_func(npoles, 2 * np.pi * bw_3db_Hz, "low", analog=True)
    w, h = gen_freqs(b, a, worN=2 * np.pi * freq)
    signal_filtered = np.fft.irfft(h * np.fft.rfft(wvf))
    return signal_filtered


def generate_data_with_filtered_noise(
    samples_per_symbol,
    nbits: float,
    amplitude: float,
    std: float,
    dt_sec: float,
    bw_3db_Hz: float,
    npoles: int,
):
    bits = np.random.choice([0.0, amplitude], nbits)
    wvf = np.repeat(bits, samples_per_symbol)
    time = dt_sec * np.arange(len(wvf))

    wvf += np.random.normal(loc=0, scale=std, size=len(wvf))

    return filter_waveform(wvf, time, bw_3db_Hz, npoles)


def generate_data_with_known_jitter(
    samples_per_symbol,
    nbits: float,
    random_jitter_std_ui: float,
    deterministic_jitter_ui: float,
):
    size = nbits // 4
    rj_idx = random_jitter_std_ui * samples_per_symbol
    t1 = 0.5 * samples_per_symbol
    t2 = 1.5 * samples_per_symbol
    dj_idx = deterministic_jitter_ui * samples_per_symbol
    ep1_1 = np.random.normal(loc=t1, scale=rj_idx, size=size)
    ep1_2 = np.random.normal(loc=t1 + dj_idx, scale=rj_idx, size=size)
    ep1 = np.concatenate((ep1_1, ep1_2))
    ep1 += np.arange(2 * size) * 2 * samples_per_symbol
    ep1[size:] += 1 * samples_per_symbol

    ep2_1 = np.random.normal(loc=t2 + dj_idx, scale=rj_idx, size=size)
    ep2_2 = np.random.normal(loc=t2, scale=rj_idx, size=size)
    ep2 = np.concatenate((ep2_1, ep2_2))
    ep2 += np.arange(2 * size) * 2 * samples_per_symbol
    ep2[size:] += 1 * samples_per_symbol

    edge_positions = np.column_stack((ep1, ep2))

    wvf = np.zeros((4 * size + 1) * samples_per_symbol)
    for (_e1, _e2) in edge_positions:
        idx1_l, idx1_u = math.floor(_e1), math.ceil(_e1)
        idx2_l, idx2_u = math.floor(_e2), math.ceil(_e2)

        wvf[idx1_u + 1 : idx2_l] = 1

        # rising edge
        if _e1 % 1 < 0.5:
            wvf[idx1_l] = 1 - 0.5 / (1 - (_e1 % 1))
            wvf[idx1_u] = 1
        else:
            wvf[idx1_l] = 0
            wvf[idx1_u] = 0.5 / (_e1 % 1)

        # falling edge
        if _e2 % 1 < 0.5:
            wvf[idx2_l] = 0.5 / (1 - (_e2 % 1))
            wvf[idx2_u] = 0
        else:
            wvf[idx2_l] = 1
            wvf[idx2_u] = 1 - 0.5 / (_e2 % 1)

    return wvf


def shape_edges_half_sine(
    wvf: np.ndarray,
    rise_fall_time: float,
    dt: float,
    samples_per_symbol: float,
    symbol_time: float,
) -> np.ndarray:
    """
    Replace all waveform edges with Half-Sine transitions with a specified rise/fall time (10-90 / 90-10)
    """
    spui = samples_per_symbol

    scale_factor = (rise_fall_time / dt) / (
        2 * np.arcsin(0.8)
    )  # 0.8 since we go up to +0.8 of sine height and -0.8 of sine dip
    edge_duration = scale_factor * np.pi
    if edge_duration * dt > (symbol_time):
        raise Exception(
            f"Warning! A rise_fall_time of {rise_fall_time} psec results in an edge duration of {edge_duration*dt} psec which is longer than the symbol time ({symbol_time})!"
        )

    t_full = np.arange(0, spui, 1) - spui // 2
    t_before = np.ones(sum(t_full < -edge_duration / 2.0))
    t_after = np.ones(sum(t_full > edge_duration / 2.0))
    t_edge = t_full[(t_full <= edge_duration / 2.0) & (t_full >= -edge_duration / 2.0)]
    Nedgesamples = len(t_edge)

    # Subsample waveform to get the data levels at each decision time
    wvf_subsampled = wvf[spui // 2 : -1 : spui]
    Nbits = len(wvf_subsampled)

    start_values = np.roll(wvf_subsampled, 1)
    end_values = wvf_subsampled

    # Convert to matrix of size N_bits x N_pre-edge_samples
    data_before_edge = np.dot(np.vstack(start_values), np.ones((1, len(t_before))))

    # Convert to matrix of size N_bits x N_post-edge_samples
    data_after_edge = np.dot(np.vstack(end_values), np.ones((1, len(t_after))))

    # Convert to matrices of size N_bits x N_edge_samples
    start_values = np.dot(np.vstack(start_values), np.ones((1, Nedgesamples)))
    end_values = np.dot(np.vstack(end_values), np.ones((1, Nedgesamples)))
    t_edge = np.dot(np.ones((Nbits, 1)), t_edge.reshape(1, Nedgesamples))

    # Compute all edge shapes at once via numpy matrix broadcasting
    wvf_chunks = (0.5 + np.sin(t_edge / scale_factor) * 0.5) * (
        end_values - start_values
    ) + start_values

    # Concatenate the matrices of data before, at, and after the edges
    edges = np.concatenate((data_before_edge, wvf_chunks, data_after_edge), axis=1)

    # Flatten to a single array, then remove constant phase shift
    wvf = edges.flatten()
    wvf = np.roll(wvf, -spui // 2)

    return wvf
