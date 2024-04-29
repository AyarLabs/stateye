def test_ddj_separation():
    from stateye import IdealEye
    from sample_signals.generate_signals import generate_half_sine_data, filter_waveform
    import numpy as np
    import matplotlib.pyplot as plt

    np.random.seed(1123124)

    datarate_gbps = 32
    samples_per_symbol = 32
    symbol_time = 1 / (datarate_gbps * 1e9)
    dt_sec = symbol_time / samples_per_symbol

    # Initialize eye
    eye = IdealEye(
        datarate_gbps=datarate_gbps,
        dt_sec=dt_sec,
        scale_y=2.0,
        hdf5_path="eye.h5",
        num_bits_to_filter_on_before=2,
        num_bits_to_filter_on_after=1,
    )

    # Make a sample waveform
    wvf = generate_half_sine_data(
        datarate_gbps=datarate_gbps,
        samples_per_symbol=samples_per_symbol,
        nbits=10000,
        amplitude=3,
        rise_fall_time_sec=15e-12,
        add_noise=False,
    )
    times = np.arange(len(wvf)) * dt_sec
    wvf = filter_waveform(wvf, times, bw_3db_Hz=8e9, npoles=1)
    eye.add_data(wvf, "mW")

    # Add some noise
    y_values=np.arange(-5, 5, 0.025)
    sigma = 0.01
    noise_distribution = np.exp(-(y_values)**2 / (2*sigma))
    eye.add_vertical_noise(noise_distribution, y_values, y_values_unit="mW")

    # from pprint import pprint
    # pprint(eye.get_measurements())
    # eye.plot()

    # eye.plot_bathtub_cross_section(direction="horizontal", y_axis="ber")
    # eye.plot_bathtub_cross_section(direction="vertical", y_axis="ber")
    # eye.plot_bathtub_cross_section(direction="horizontal", y_axis="q-scale")
    # eye.plot_bathtub_cross_section(direction="vertical", y_axis="q-scale")

    # x_values=np.arange(-5, 5, 0.025)
    # noise_distribution=np.zeros(len(x_values))
    # noise_distribution[abs(x_values) < 3] = 1
    # eye.add_jitter_distribution(jitter_times_psec=x_values, jitter_distribution_values=noise_distribution)

    # eye.get_measurements()
    # eye.plot(ber_thresholds=[1e-12, 1e-15])
    # eye.plot_bathtub()
    # for dp in eye.data_patterns:
    #     eye.plot(pattern=dp)
    
    # Make sure there exists something in each histogram
    nz = eye.hist_data.shape[2]
    assert nz == 2**4
    for zi in range(eye.hist_data.shape[2]):
        assert np.sum(eye.hist_data[:,:,zi]) > 1