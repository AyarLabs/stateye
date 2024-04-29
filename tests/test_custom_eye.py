def test_custom_eye():
    from stateye import CustomEye
    from sample_signals.generate_signals import generate_half_sine_data, filter_waveform
    import numpy as np
    import matplotlib.pyplot as plt
    import math

    np.random.seed(1123124)

    datarate_gbps = 32
    samples_per_symbol = 32
    period_sec = 1 / (datarate_gbps * 1e9)
    dt_sec = period_sec / samples_per_symbol
    nbits = 10000

    # Initialize eye
    eye = CustomEye(
        datarate_gbps=datarate_gbps,
        dt_sec=dt_sec,
        scale_y=2.0,
    )

    # Make a sample waveform
    wvf = generate_half_sine_data(
        datarate_gbps=datarate_gbps,
        samples_per_symbol=samples_per_symbol,
        nbits=nbits,
        amplitude=3,
        rise_fall_time_sec=15e-12,
        add_noise=False,
    )

    # Create eyes with square-wave timing jitter.  Increasing the amp. should
    # close the eye width by an amount equal to the pk-pk jitter change
    st = np.arange(nbits) * period_sec + 0.7 * period_sec

    times = np.arange(len(wvf)) * dt_sec
    wvf = filter_waveform(wvf, times, bw_3db_Hz=20e9, npoles=1)

    st[1::2] += 2.0 * 1e-12
    st[::2] -= 2.0 * 1e-12

    eye.add_data(wvf, "mV", st, "sec")

    m = eye.get_measurements()
    width1 = m["inner_eye_width"]
    print(f"inner_eye_width = {width1}")

    # Initialize eye
    eye2 = CustomEye(
        datarate_gbps=datarate_gbps,
        dt_sec=dt_sec,
        scale_y=2.0,
    )

    st = np.arange(nbits) * period_sec + 0.7 * period_sec
    st[1::2] += 4.0 * 1e-12
    st[::2] -= 4.0 * 1e-12
    eye2.add_data(wvf, "mV", st, "sec")

    m = eye2.get_measurements()
    width2 = m["inner_eye_width"]
    print(f"inner_eye_width = {width2}")
    print(f"width1 - width2 = {width1 - width2}")

    assert math.isclose(width1 - width2, 4.0, rel_tol=1e-2)
