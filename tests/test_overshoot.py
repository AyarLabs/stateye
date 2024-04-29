def test_overshoot():
    import numpy as np
    from stateye import IdealEye
    from pprint import pprint
    import math

    amplitude = 1
    sps = 64
    nbits = 80
    datarate_gbps = 32
    period = 1 / (datarate_gbps * 1e9)
    dt_sec = period / sps

    PO = 10  # percentage overshoot
    # damping factor
    d = -np.log(PO/100) / np.sqrt(np.pi**2 + np.log(PO/100)**2)
    fn = 200e9  # natural frequency
    sigma = -d * fn
    omega_d = fn * np.sqrt(1 - d**2)
    p1, p2 = sigma + 1j * omega_d, sigma - 1j * omega_d
    K = p1 * p2

    # Peak time (from start of step function, *not* from 50% point)
    # t_peak = np.pi / omega_d
    # print(f"t_peak = {t_peak}")

    freq = np.fft.rfftfreq(nbits*sps, d=dt_sec)
    s = 1j * 2 * np.pi * freq
    tf = K / ((s - p1) * (s - p2))

    sequence = np.tile([0]*8 + [1]*8, nbits//16)
    wvf = np.repeat(amplitude * sequence, sps).astype(float)
    wvf_f = np.fft.irfft(tf * np.fft.rfft(wvf))

    # Initialize eye
    eye = IdealEye(
        datarate_gbps=datarate_gbps,
        dt_sec=dt_sec,
        scale_y=2.0,
        num_bits_to_filter_on_before=0,
        num_bits_to_filter_on_after=0,
        sampling_offset_mode="half_ui"
    )
    eye.add_data(wvf_f, "mV")
    # eye.plot()

    m = eye.get_measurements()
    # should be accurate to <5% when using crossing point (xp) OMA:
    assert math.isclose(m["overshoot_percentage_xp"], PO, rel_tol=5e-2)
    assert math.isclose(m["undershoot_percentage_xp"], PO, rel_tol=5e-2)

    # should be more accurate (<0.1%) for longer patterns:
    assert math.isclose(m["overshoot_percentage_4140"], PO, rel_tol=1e-3)
    assert math.isclose(m["undershoot_percentage_4140"], PO, rel_tol=1e-3)
    assert math.isclose(m["overshoot_percentage_8180"], PO, rel_tol=1e-3)
    assert math.isclose(m["undershoot_percentage_8180"], PO, rel_tol=1e-3)

    measured_delay_ps = 15.558 - 2.872  # Measured manually from eye diagam
    for oma_type in ["xp", "4140", "8180"]:
        assert math.isclose(m[f"abs_overshoot_time_{oma_type}"], measured_delay_ps, abs_tol=dt_sec*1e12)
        assert math.isclose(m[f"abs_undershoot_time_{oma_type}"], measured_delay_ps, abs_tol=dt_sec*1e12)