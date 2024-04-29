def test_inner_eye_height_and_width():
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
        num_bits_to_filter_on_after=0,
        num_bits_to_filter_on_before=0
    )

    # Make a sample waveform
    oma = 1.0
    wvf = generate_half_sine_data(
        datarate_gbps=datarate_gbps,
        samples_per_symbol=samples_per_symbol,
        nbits=10000,
        amplitude=oma,
        rise_fall_time_sec=15e-12,
        add_noise=True,
    )
    eye.add_data(wvf, "mV")

    msmts = eye.get_measurements()

    # print(f'msmts["inner_eye_height"] = {msmts["inner_eye_height"]}')
    # print(f'msmts["inner_eye_width"] = {msmts["inner_eye_width"]}')
    # eye.plot()

    # hard coded numbers from manually measuring off the eye diagram
    assert math.isclose(msmts["inner_eye_height"], 0.640, abs_tol=eye.dy)
    assert math.isclose(msmts["inner_eye_width"], 25.635, abs_tol=eye.dx)


def test_vertical_eye_height():
    # Test that the eye height at a BER all the way down to 1E-15 can be extrapolated
    # to within < 1% of "truth" (using the known PDF)
    from stateye import IdealEye
    from sample_signals.generate_signals import generate_data_with_filtered_noise
    import numpy as np
    import math
    from scipy import interpolate
    from scipy.optimize import root
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
        ny=2048,
        num_bits_to_filter_on_before=2,
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

    # From the eye diagram with *no noise*, grab the data levels so that
    # we can create an analytical PDF
    msmts = eye.get_measurements()
    dlevs = sorted(msmts["d_lev"])

    # Now add vertical gaussian noise to the eye diagram
    def gaussian(x, a, x0, sigma):
        return a * np.exp(-((x - x0) ** 2) / (2 * sigma ** 2))
    y = np.arange(-0.5, 0.5, 0.01)
    sigma = 0.025
    eye.add_vertical_noise(noise_distribution=gaussian(y, 1, 0, sigma), y_values=y, y_values_unit="mV")
    msmts = eye.get_measurements()

    """ Now numerically compute the BER from the known PDF """
    xpts = np.linspace(-1, 2.1, 1000)
    one_level = np.sum([gaussian(xpts, 1, dlev, sigma) for dlev in dlevs[len(dlevs)//2:]], axis=0)
    one_level_cdf = np.cumsum(one_level)
    one_level_cdf /= np.sum(one_level)
    f = interpolate.interp1d(xpts, one_level_cdf)

    target_bers = [1e-3, 1e-6, 1e-9, 1e-12, 1e-15]
    for ber in target_bers:
        x = eye.threshold + msmts[f"upper_eye_height_{ber}"]
        sol = root(lambda _x: np.log10(f(_x)) - np.log10(ber), x0=x)
        x_opt = sol.x[0]
        # print(f"x error (ber={ber}) = {x - x_opt}")
        assert math.isclose(x, x_opt, abs_tol=1e-2)

    #     plt.semilogy(x, ber, 'go')
    #     plt.semilogy([x, x_opt], [ber, f(x_opt)], 'r-')
    # plt.semilogy(xpts, one_level_cdf)
    # plt.ylim([1e-30, 1])
    # plt.show()

    xpts = np.linspace(-1.0, 1.0, 1000)

    # At each lower data level place a gaussian
    zero_level = np.sum([gaussian(xpts, 1, dlev, sigma) for dlev in dlevs[:len(dlevs)//2]], axis=0)
    zero_level_cdf = np.flip(np.cumsum(np.flip(zero_level)))
    zero_level_cdf /= np.sum(zero_level)
    f = interpolate.interp1d(xpts, zero_level_cdf)

    target_bers = [1e-3, 1e-6, 1e-9, 1e-12, 1e-15]
    for ber in target_bers:
        x = eye.threshold - msmts[f"lower_eye_height_{ber}"]
        sol = root(lambda _x: np.log10(f(_x)) - np.log10(ber), x0=x)
        x_opt = sol.x[0]
        # print(f"x error (ber={ber}) = {x - x_opt}")
        assert math.isclose(x, x_opt, abs_tol=1e-2)

    #     plt.semilogy(x, ber, 'go')
    #     plt.semilogy([x, x_opt], [ber, f(x_opt)], 'r-')
    # plt.semilogy(xpts, zero_level_cdf)
    # plt.ylim([1e-30, 1])
    # plt.show()


# def test_horizontal_eye_width():
#     """Analysis of the horizontal slice of the eye diagram.
#     This test checks the following analysis methods:
#         - BER all the way down to 1E-15 can be extrapolated to within < 1% of
#             "truth" (using the known PDF)
#         - The random jitter (RJ) can be extracted
#         - The total deterministic jitter profile can be extracted

#     """
#     from stateye import IdealEye
#     from sample_signals.generate_signals import generate_data_with_known_jitter
#     import numpy as np
#     import math
#     from scipy import interpolate
#     from scipy.optimize import root
#     import matplotlib.pyplot as plt

#     np.random.seed(1123124)

#     datarate_gbps = 32
#     samples_per_symbol = 32
#     symbol_time_sec = 1 / (datarate_gbps * 1e9)
#     dt_sec = symbol_time_sec / samples_per_symbol

#     # Initialize eye
#     eye = IdealEye(
#         datarate_gbps=datarate_gbps,
#         dt_sec=dt_sec,
#     )

#     random_jitter_std_psec = 0.625
#     deterministic_jitter_pk_pk = 4.6875

#     # Make a sample waveform
#     wvf = generate_data_with_known_jitter(
#         samples_per_symbol=samples_per_symbol,
#         nbits=100000,
#         random_jitter_std_ui=random_jitter_std_psec / (symbol_time_sec * 1e12),
#         deterministic_jitter_ui=deterministic_jitter_pk_pk / (symbol_time_sec * 1e12),
#     )
#     eye.add_data(wvf, "mV")

#     msmts = eye.get_measurements()

#     # print(f"msmts['random_jitter_std'] = {msmts['random_jitter_std']}")
#     # print(f"msmts['deterministic_jitter_pk_pk'] = {msmts['deterministic_jitter_pk_pk']}")
#     assert math.isclose(
#         random_jitter_std_psec, msmts["random_jitter_std"], rel_tol=1e-2
#     )
#     assert math.isclose(
#         deterministic_jitter_pk_pk, msmts["deterministic_jitter_pk_pk"], rel_tol=1e-2
#     )

#     """ Now numerically compute the horizontal BER from the known PDF """

#     def gaussian(x, a, x0, sigma):
#         return a * np.exp(-((x - x0) ** 2) / (2 * sigma ** 2))

#     xpts = np.linspace(-10, 10, 1000)
#     jitter = gaussian(
#         xpts, 0.5, -deterministic_jitter_pk_pk / 2, random_jitter_std_psec
#     )
#     jitter += gaussian(
#         xpts, 0.5, deterministic_jitter_pk_pk / 2, random_jitter_std_psec
#     )
#     lower_cdf = np.cumsum(jitter)
#     lower_cdf /= np.sum(jitter)
#     upper_cdf = np.cumsum(np.flip(jitter))
#     upper_cdf /= np.sum(jitter)

#     f_lower = interpolate.interp1d(xpts, lower_cdf)
#     f_upper = interpolate.interp1d(np.flip(xpts), upper_cdf)

#     period = 31.25

#     target_bers = [1e-3, 1e-6, 1e-9, 1e-12, 1e-15]
#     for ber in target_bers:
#         x_est = (period - msmts[f"eye_width_{ber}"]) / 2

#         sol = root(lambda _x: f_lower(_x) - ber, x0=-x_est)
#         x_opt_lower = sol.x[0]
#         # print(f"x_opt_lower = {x_opt_lower}")

#         sol = root(lambda _x: f_upper(_x) - ber, x0=x_est)
#         x_opt_upper = sol.x[0]
#         # print(f"x_opt_upper = {x_opt_upper}")

#         true_eye_width = period - (x_opt_upper - x_opt_lower)
#         assert math.isclose(true_eye_width, msmts[f"eye_width_{ber}"], rel_tol=1e-2)

#     #     plt.semilogy(x_est, ber, 'go')
#     #     plt.semilogy(-x_est, ber, 'go')
#     #     plt.semilogy([x_est, x_opt_upper], [ber, f_upper(x_opt_upper)], 'r-')
#     #     plt.semilogy([-x_est, x_opt_lower], [ber, f_lower(x_opt_lower)], 'r-')
#     # plt.semilogy(np.flip(xpts), upper_cdf)
#     # plt.semilogy(xpts, lower_cdf)
#     # plt.ylim([1e-30, 1])
#     # plt.show()


def test_slicer_sensitivity():
    # Test that the bit error rate is computed correctly when the user supplies
    # a value for the slicer sensitivity.
    from stateye import IdealEye
    from sample_signals.generate_signals import generate_data_with_filtered_noise
    import numpy as np

    np.random.seed(1123124)

    datarate_gbps = 32
    samples_per_symbol = 64
    symbol_time = 1 / (datarate_gbps * 1e9)
    dt_sec = symbol_time / samples_per_symbol
    sensitivity = 100e-3  # mV

    # Make a sample waveform
    np.random.seed(123)
    wvf = generate_data_with_filtered_noise(
        samples_per_symbol=samples_per_symbol,
        nbits=10000,
        amplitude=1 - sensitivity,
        std=0.0,
        dt_sec=dt_sec,
        bw_3db_Hz=200e9,
        npoles=1,
    )

    # Initialize eye
    eye = IdealEye(
        datarate_gbps=datarate_gbps,
        dt_sec=dt_sec,
        nx=256//2,
        ny=2048,
        sampling_offset_mode="half_ui",
        num_bits_to_filter_on_after=0,
        num_bits_to_filter_on_before=2
    )
    eye.add_data(wvf, "mV")

    def gaussian(x, a, x0, sigma):
        return a * np.exp(-((x - x0) ** 2) / (2 * sigma ** 2))
    y = np.arange(-0.5, 0.5, 0.01)
    sigma = 0.025
    eye.add_vertical_noise(noise_distribution=gaussian(y, 1, 0, sigma), y_values=y, y_values_unit="mV")

    msmts = eye.get_measurements()
    eye_height_shifted = msmts['eye_height_0.001']
    print(f"eye resolution dy={eye.dy}")
    print(f"1e-12 eye height (shifted) = {eye_height_shifted}")

    # Make a sample waveform
    np.random.seed(123)
    wvf = generate_data_with_filtered_noise(
        samples_per_symbol=samples_per_symbol,
        nbits=10000,
        amplitude=1,
        std=0.0,
        dt_sec=dt_sec,
        bw_3db_Hz=200e9,
        npoles=1,
    )

    # Initialize eye
    eye = IdealEye(
        datarate_gbps=datarate_gbps,
        dt_sec=dt_sec,
        nx=256//2,
        ny=2048,
        sampling_offset_mode="half_ui",
        num_bits_to_filter_on_after=0,
        num_bits_to_filter_on_before=2,
    )
    eye.add_data(wvf, "mV")

    def gaussian(x, a, x0, sigma):
        return a * np.exp(-((x - x0) ** 2) / (2 * sigma ** 2))
    y = np.arange(-0.5, 0.5, 0.01)
    eye.add_vertical_noise(noise_distribution=gaussian(y, 1, 0, sigma), y_values=y, y_values_unit="mV")

    msmts = eye.get_measurements()
    eye_height_original = msmts['eye_height_0.001']
    print(f"1e-12 eye height (no sensitivity added yet) = {eye_height_original}")

    eye.set_slicer_sensitivity(sensitivity, "mV")

    msmts = eye.get_measurements()
    eye_height_with_sensitivity = msmts['eye_height_0.001']
    print(f"1e-12 eye height (sensitivity added) = {eye_height_with_sensitivity}")

    # The eye height from adding the sensitivity should be equivalent to the height for
    # the same dataset but with 1 and 0 levels moved closer together
    assert np.abs(eye_height_with_sensitivity - eye_height_shifted) < eye.dy

    # Adding a slicer sensitivity should decrease the eye height at a given BER
    assert eye_height_with_sensitivity < eye_height_original
