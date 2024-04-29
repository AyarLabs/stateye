def test_oma():
    from stateye import IdealEye
    from sample_signals.generate_signals import generate_half_sine_data
    import numpy as np
    import math

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
        num_bits_to_filter_on_after=0,
        num_bits_to_filter_on_before=0,
    )

    # Make a sample waveform
    oma = 1.0723
    wvf = generate_half_sine_data(
        datarate_gbps=datarate_gbps,
        samples_per_symbol=samples_per_symbol,
        nbits=10000,
        amplitude=oma,
        rise_fall_time_sec=15e-12,
        add_noise=True,
    )
    eye.add_data(wvf, "mV")

    for oma_type in ["xp", "4140", "8180"]:
        measured_oma = eye.get_measurements()[f"oma_{oma_type}"]
        assert math.isclose(oma, measured_oma, rel_tol=1e-2)


def test_rise_fall_time_10_90():
    from stateye import IdealEye
    from sample_signals.generate_signals import generate_half_sine_data
    import numpy as np
    import math

    np.random.seed(1123124)

    datarate_gbps = 32
    samples_per_symbol = 64
    symbol_time = 1 / (datarate_gbps * 1e9)
    dt_sec = symbol_time / samples_per_symbol

    # Initialize eye
    eye = IdealEye(
        datarate_gbps=datarate_gbps,
        dt_sec=dt_sec,
        scale_y=2.0,
        num_bits_to_filter_on_after=0,
        num_bits_to_filter_on_before=0,
    )
    print(f"dt_psec = {dt_sec * 1e12}")

    # Make a sample waveform
    rise_time = 15e-12
    wvf = generate_half_sine_data(
        datarate_gbps=datarate_gbps,
        samples_per_symbol=samples_per_symbol,
        nbits=10000,
        amplitude=1.0,
        rise_fall_time_sec=15e-12,
        add_noise=False,
    )
    eye.add_data(wvf, "mV")

    measurement = eye.get_measurements()
    for oma_type in ["xp", "4140", "8180"]: 
        # 10-90 edges
        measured_rise_time = measurement[f"rise_time_10-90_{oma_type}"]
        print(f"measured_rise_time = {measured_rise_time}")
        print(f"rise_time = {rise_time}")
        assert math.isclose(rise_time * 1e12, measured_rise_time, rel_tol=1e-3)

        measured_fall_time = measurement[f"fall_time_90-10_{oma_type}"]
        print(f"measured_fall_time = {measured_fall_time}")
        print(f"rise_time = {rise_time}")
        assert math.isclose(rise_time * 1e12, measured_fall_time, rel_tol=1e-3)

        # 20-80 edges
        correction_factor = (np.arcsin(0.4/0.5) - np.arcsin(-0.4/0.5))/(np.arcsin(0.3/0.5) - np.arcsin(-0.3/0.5))
        measured_rise_time = measurement[f"rise_time_20-80_{oma_type}"]
        print(f"measured_rise_time = {measured_rise_time}")
        print(f"rise_time = {rise_time * 1e12 / correction_factor}")
        assert math.isclose(rise_time * 1e12 / correction_factor, measured_rise_time, rel_tol=1e-3)

        measured_fall_time = eye.get_measurements()[f"fall_time_80-20_{oma_type}"]
        print(f"measured_fall_time = {measured_fall_time}")
        print(f"rise_time = {rise_time * 1e12 / correction_factor}")
        assert math.isclose(rise_time * 1e12 / correction_factor, measured_fall_time, rel_tol=1e-3)


def test_8180_oma():
    from stateye import IdealEye
    from sample_signals.generate_signals import generate_half_sine_data
    import numpy as np
    import math

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
        num_bits_to_filter_on_after=0,
        num_bits_to_filter_on_before=0,
    )

    # Make a sample waveform
    oma = 1.0723
    wvf = generate_half_sine_data(
        datarate_gbps=datarate_gbps,
        samples_per_symbol=samples_per_symbol,
        nbits=30000,
        amplitude=oma,
        rise_fall_time_sec=15e-12,
        add_noise=True,
    )
    eye.add_data(wvf, "mV")

    measured_oma = eye.get_measurements()["oma_8180"]
    assert math.isclose(oma, measured_oma, rel_tol=1e-2)


def test_4140_oma():
    from stateye import IdealEye
    from sample_signals.generate_signals import generate_half_sine_data
    import numpy as np
    import math

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
        num_bits_to_filter_on_after=0,
        num_bits_to_filter_on_before=0,
    )

    # Make a sample waveform
    oma = 1.0723
    wvf = generate_half_sine_data(
        datarate_gbps=datarate_gbps,
        samples_per_symbol=samples_per_symbol,
        nbits=30000,
        amplitude=oma,
        rise_fall_time_sec=15e-12,
        add_noise=True,
    )
    eye.add_data(wvf, "mV")

    measured_oma = eye.get_measurements()["oma_4140"]
    assert math.isclose(oma, measured_oma, rel_tol=1e-2)


def test_d_levs():
    from stateye import IdealEye
    from sample_signals.generate_signals import generate_data_with_filtered_noise
    import numpy as np

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
        num_bits_to_filter_on_after=0,
        num_bits_to_filter_on_before=2,
    )

    # Make a sample waveform
    wvf = generate_data_with_filtered_noise(
        samples_per_symbol=samples_per_symbol,
        nbits=30000,
        amplitude=1,
        std=0.1,
        dt_sec=dt_sec,
        bw_3db_Hz=10e9,
        npoles=1,
    )
    eye.add_data(wvf, "mV")

    dlm = [
        eye.get_dlev([1, 1], 1, []),
        eye.get_dlev([0, 1], 1, []),
        eye.get_dlev([0, 0], 1, []),
        eye.get_dlev([1, 1], 0, []),
        eye.get_dlev([1, 0], 0, []),
        eye.get_dlev([0, 0], 0, [])
    ]
    assert dlm[0] > dlm[1]
    assert dlm[1] > dlm[2]
    assert dlm[3] > dlm[4]
    assert dlm[4] > dlm[5]