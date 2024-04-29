def test_ideal_eye():
    from stateye import IdealEye
    import numpy as np

    np.random.seed(1123124)

    datarate_gbps = 32
    samples_per_symbol = 32
    dt_sec = (31.25e-12) / samples_per_symbol

    # Initialize eye
    eye = IdealEye(
        datarate_gbps=datarate_gbps,
        dt_sec=dt_sec,
        scale_y=2.0,
    )

    # Make a sample waveform
    n = 100000
    time = np.arange(n) * dt_sec
    freq = datarate_gbps * 1e9 * 0.5
    sinewave = 0.5 + 0.5 * np.sin(2 * np.pi * time * freq)
    sinewave += np.random.normal(scale=0.1, size=n)  # Add some noise

    eye.add_data(sinewave, "mV")

    # eye.plot()
