def test_ideal_cdr():
    from stateye import IdealEye
    from sample_signals.generate_signals import generate_data_with_filtered_noise
    import numpy as np
    import math

    datarate_gbps = 32
    samples_per_symbol = 64
    symbol_time = 1 / (datarate_gbps * 1e9)
    dt_sec = symbol_time / samples_per_symbol
    bw_3db_GHz = 9

    # Make a sample waveform
    np.random.seed(123)
    wvf = generate_data_with_filtered_noise(
        samples_per_symbol=samples_per_symbol,
        nbits=20000,
        amplitude=1,
        std=0.2,
        dt_sec=dt_sec,
        bw_3db_Hz=bw_3db_GHz * 1e9,
        npoles=1,
    )

    nb, na = 3, 1
    metric = "eye_height_1e-06"

    # Initialize eye
    eye = IdealEye(
        datarate_gbps=datarate_gbps,
        dt_sec=dt_sec,
        scale_y=2.0,
        sampling_offset_mode="adaptive",
        num_bits_to_filter_on_after=na,
        num_bits_to_filter_on_before=nb,
    )
    eye.add_data(wvf, "mV")
    sampling_offset_optimal = eye.sampling_offset
    assert math.isclose(eye.sampling_offset, 42.0, abs_tol=1)

    m = eye.get_measurements()
    eye_height = m[metric]

    # Check that the sampling offset is optimal
    eye = IdealEye(
        datarate_gbps=datarate_gbps,
        dt_sec=dt_sec,
        scale_y=2.0,
        sampling_offset_mode="adaptive",
        num_bits_to_filter_on_after=na,
        num_bits_to_filter_on_before=nb,
    )
    eye.add_data(wvf, "mV", sampling_offset=sampling_offset_optimal + 1)
    m = eye.get_measurements()
    eye_height2 = m[metric]

    eye = IdealEye(
        datarate_gbps=datarate_gbps,
        dt_sec=dt_sec,
        scale_y=2.0,
        sampling_offset_mode="adaptive",
        num_bits_to_filter_on_after=na,
        num_bits_to_filter_on_before=nb,
    )
    eye.add_data(wvf, "mV", sampling_offset=sampling_offset_optimal - 1)
    m = eye.get_measurements()
    eye_height3 = m[metric]

    print(f"eye_height = {eye_height}")
    print(f"eye_height_late = {eye_height2}")
    print(f"eye_height_early = {eye_height3}")

    assert eye_height > eye_height2
    assert eye_height > eye_height3


def test_ideal_cdr_half_ui_mode():
    from stateye import IdealEye
    from sample_signals.generate_signals import generate_data_with_filtered_noise
    import numpy as np
    import math

    datarate_gbps = 32
    samples_per_symbol = 64
    symbol_time = 1 / (datarate_gbps * 1e9)
    dt_sec = symbol_time / samples_per_symbol
    bw_3db_GHz = 9

    # Make a sample waveform
    np.random.seed(123)
    wvf = generate_data_with_filtered_noise(
        samples_per_symbol=samples_per_symbol,
        nbits=5000,
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
        sampling_offset_mode="half_ui",
    )
    eye.add_data(wvf, "mV")
    sampling_offset_optimal = eye.sampling_offset
    # print(f"eye.sampling_offset = {eye.sampling_offset}")
    assert math.isclose(eye.sampling_offset, samples_per_symbol / 2)
    # eye.plot()
