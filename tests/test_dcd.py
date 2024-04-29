def test_dcd():
    from stateye import IdealEye
    import numpy as np
    from sample_signals.generate_signals import filter_waveform
    from scipy.signal import bessel
    import math

    amplitude = 1
    sps = 32
    nbits = 1024
    datarate_gbps = 32
    period = 1 / (datarate_gbps * 1e9)
    dt_sec = period / sps
    dcd_px: int = 4
    dcd_psec = -dcd_px * dt_sec * 1e12  # duty cycle distortion = rising edge - falling edge

    sequence = np.tile([0]*8 + [1]*8, nbits//16)
    duration = np.tile([sps-dcd_px]+[sps]*7+[sps+dcd_px]+[sps]*7, nbits//16)
    wvf = np.repeat(amplitude * sequence, duration).astype(float)
    wvf_f = filter_waveform(wvf, np.arange(len(wvf))*dt_sec, 100e9, 4, bessel)

    # Initialize eye
    eye = IdealEye(
        datarate_gbps=datarate_gbps,
        dt_sec=dt_sec,
        sampling_offset_mode="half_ui",
        num_bits_to_filter_on_before=0,
        num_bits_to_filter_on_after=0
    )

    eye.add_data(wvf_f, "mV")
    # eye.plot()
    
    m = eye.get_measurements()
    assert math.isclose(m['dcd_xp'], dcd_psec, rel_tol=1e-3)
    assert math.isclose(m['dcd_4140'], dcd_psec, rel_tol=1e-3)
    assert math.isclose(m['dcd_8180'], dcd_psec, rel_tol=1e-3)