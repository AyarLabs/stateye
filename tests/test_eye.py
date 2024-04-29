def test_basic_eye_functions():
    from stateye import IdealEye
    from sample_signals.generate_signals import generate_half_sine_data
    import numpy as np
    import matplotlib.pyplot as plt
    import copy

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
        dump_to_hdf5=True,
    )

    # Make a sample waveform
    wvf = generate_half_sine_data(
        datarate_gbps=datarate_gbps,
        samples_per_symbol=samples_per_symbol,
        nbits=10000,
        amplitude=3,
        rise_fall_time_sec=15e-12,
        add_noise=True,
    )
    eye.add_data(wvf, "mV")

    # Check that we can run all of the plotting utilities
    msmt_before = copy.deepcopy(eye.msmts)
    msmt_counts_before = copy.deepcopy(eye.get_measurement_counts())
    # print(f"msmt before = {msmt_before}")
    eye.plot(show=False)
    plt.close()

    # Check that hdf5 load/dumps work
    # Dump data to hdf5
    eye.dump_hdf5_dataset()

    # Create a new eye object and load the hdf5 data in
    eye2 = IdealEye(
        datarate_gbps=datarate_gbps,
        dt_sec=dt_sec,
        scale_y=2.0,
    )
    eye2.load_hdf5_dataset(filename="eye.h5")
    # print(f"msmt after = {eye2.msmts}")
    eye2.plot(show=False)
    msmt_after = eye2.get_measurements()

    # Check that measurements are preserved upon dumping/loading
    for key in msmt_before.keys():
        if hasattr(msmt_before[key], "__iter__"):  # lists, ndarrays, etc
            assert np.array_equal(
                np.array(msmt_before[key]), np.array(msmt_after[key]), equal_nan=True
            )
        else:
            if np.isnan(msmt_before[key]):
                assert np.isnan(msmt_after[key])
            else:
                assert msmt_before[key] == msmt_after[key]

    # Check that the measurement counts is preserved
    msmt_counts_after = eye2.get_measurement_counts()
    for key in msmt_counts_before.keys():
        if not hasattr(msmt_counts_before[key], "__iter__"):
            assert msmt_counts_before[key] == msmt_counts_after[key]
        else:
            for mv1, mv2 in zip(msmt_counts_before[key], msmt_counts_after[key]):
                assert mv1 == mv2

    # Test units
    msmt_units = eye2.get_measurement_units()

    assert len(msmt_units.keys()) == len(msmt_after.keys())