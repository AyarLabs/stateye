from stateye import IdealEye
from sample_signals.generate_signals import generate_data_with_filtered_noise
import numpy as np
from tqdm import tqdm

datarate_gbps = 32
samples_per_symbol = 64
symbol_time = 1 / (datarate_gbps * 1e9)
dt_sec = symbol_time / samples_per_symbol
bw_3db_GHz = 9

# Make a sample waveform
np.random.seed(123)
wvf = generate_data_with_filtered_noise(
    samples_per_symbol=samples_per_symbol,
    nbits=16352,
    amplitude=1,
    std=0.2,
    dt_sec=dt_sec,
    bw_3db_Hz=bw_3db_GHz * 1e9,
    npoles=1,
)

# Initialize eye
eye = IdealEye(
    datarate_gbps=datarate_gbps,
    dt_sec=dt_sec,
    scale_y=2.0,
)
for _ in tqdm(range(16)):
    eye.add_data(wvf, "mV")

"""
Current benchmarks on Macbook Pro 2019 with i9 processor:
4.96 sec for proces_data()
1.55 sec for ideal cdr lock
"""
