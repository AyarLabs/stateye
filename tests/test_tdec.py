def test_tdec():
    # Test accuracy of the TDEC calculation
    from stateye import IdealEye
    from sample_signals.generate_signals import generate_data_with_filtered_noise
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
        nx=1024,
        ny=2*2048,
        num_bits_to_filter_on_before=0,
        num_bits_to_filter_on_after=0,
    )

    # Make a sample waveform
    wvf = generate_data_with_filtered_noise(
        samples_per_symbol=samples_per_symbol,
        nbits=10000,
        amplitude=1,
        std=0,
        dt_sec=dt_sec,
        bw_3db_Hz=8e9,
        npoles=1,
    )
    eye.add_data(wvf, "mV")

    # eye.plot()
    # From inspection of the eye diagram above, the lowest data levels at 0.4 UI (smallest part) are: 0.70740 and 0.2927
    eye_opening = 0.70740 - 0.2927
    # No noise on this, so purely bandwidth related eye closure
    tdec_expected = 10*np.log10(1/eye_opening)

    # From the eye diagram with *no noise*, grab the data levels so that
    # we can create an analytical PDF
    msmts = eye.get_measurements()

    assert np.isclose(msmts["tdec_8180"], tdec_expected, atol=0.01)