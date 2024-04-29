"""
Dictionary of all measurement names and directions.

key = measurement name
value = (datatype, measurement-direction)

The measurement-direction is used to infer the units.  For example, if power
waveforms are stored by the Eye object, and a measurement has direction "y",
then the returned units will be in units of Watts.  If the measurement direction
is "t" then the returned quantity is in units of time (e.g. psec).  If None,
then the returned value is dimensionless.
"""
MEASUREMENTS = {
    "zero_level_xp": (float, "y"),
    "zero_level_4140": (float, "y"),
    "zero_level_8180": (float, "y"),
    "one_level_xp": (float, "y"),
    "one_level_4140": (float, "y"),
    "one_level_8180": (float, "y"),
    "average": (float, "y"),
    "rise_time_10-90_xp": (float, "t"),
    "fall_time_90-10_xp": (float, "t"),
    "rise_time_20-80_xp": (float, "t"),
    "fall_time_80-20_xp": (float, "t"),
    "rise_time_10-90_4140": (float, "t"),
    "fall_time_90-10_4140": (float, "t"),
    "rise_time_20-80_4140": (float, "t"),
    "fall_time_80-20_4140": (float, "t"),
    "rise_time_10-90_8180": (float, "t"),
    "fall_time_90-10_8180": (float, "t"),
    "rise_time_20-80_8180": (float, "t"),
    "fall_time_80-20_8180": (float, "t"),
    "abs_rise_time_10_xp": (float, "t"),
    "abs_rise_time_20_xp": (float, "t"),
    "abs_rise_time_50_xp": (float, "t"),
    "abs_rise_time_80_xp": (float, "t"),
    "abs_rise_time_90_xp": (float, "t"),
    "abs_fall_time_10_xp": (float, "t"),
    "abs_fall_time_20_xp": (float, "t"),
    "abs_fall_time_50_xp": (float, "t"),
    "abs_fall_time_80_xp": (float, "t"),
    "abs_fall_time_90_xp": (float, "t"),
    "abs_rise_time_10_4140": (float, "t"),
    "abs_rise_time_20_4140": (float, "t"),
    "abs_rise_time_50_4140": (float, "t"),
    "abs_rise_time_80_4140": (float, "t"),
    "abs_rise_time_90_4140": (float, "t"),
    "abs_fall_time_10_4140": (float, "t"),
    "abs_fall_time_20_4140": (float, "t"),
    "abs_fall_time_50_4140": (float, "t"),
    "abs_fall_time_80_4140": (float, "t"),
    "abs_fall_time_90_4140": (float, "t"),
    "abs_rise_time_10_8180": (float, "t"),
    "abs_rise_time_20_8180": (float, "t"),
    "abs_rise_time_50_8180": (float, "t"),
    "abs_rise_time_80_8180": (float, "t"),
    "abs_rise_time_90_8180": (float, "t"),
    "abs_fall_time_10_8180": (float, "t"),
    "abs_fall_time_20_8180": (float, "t"),
    "abs_fall_time_50_8180": (float, "t"),
    "abs_fall_time_80_8180": (float, "t"),
    "abs_fall_time_90_8180": (float, "t"),
    "threshold": (float, "y"),
    "extinction_ratio_xp": (float, None),
    "extinction_ratio_4140": (float, None),
    "extinction_ratio_8180": (float, None),
    "oma_xp": (float, "y"),
    "oma_8180": (float, "y"),
    "oma_4140": (float, "y"),
    "dcd_xp": (float, "t"),
    "dcd_4140": (float, "t"),
    "dcd_8180": (float, "t"),
    "vecp_xp": (float, None),
    "vecp_4140": (float, None),
    "vecp_8180": (float, None),
    "tdec_xp": (float, None),
    "tdec_4140": (float, None),
    "tdec_8180": (float, None),
    "inner_eye_height": (float, "y"),
    "inner_eye_width": (float, "t"),
    "overshoot_percentage_xp": (float, None),
    "overshoot_percentage_4140": (float, None),
    "overshoot_percentage_8180": (float, None),
    "undershoot_percentage_xp": (float, None),
    "undershoot_percentage_4140": (float, None),
    "undershoot_percentage_8180": (float, None),
    "abs_overshoot_time_xp": (float, "t"),
    "abs_overshoot_time_4140": (float, "t"),
    "abs_overshoot_time_8180": (float, "t"),
    "abs_undershoot_time_xp": (float, "t"),
    "abs_undershoot_time_4140": (float, "t"),
    "abs_undershoot_time_8180": (float, "t"),
    "vertical_ber": (float, None),
    "vertical_ber_optimized": (float, None),
    "eye_height_0.025": (float, "y"),
    "eye_height_0.001": (float, "y"),
    "eye_height_1e-06": (float, "y"),
    "eye_height_1e-09": (float, "y"),
    "eye_height_1e-12": (float, "y"),
    "eye_height_1e-15": (float, "y"),
    "upper_eye_height_0.025": (float, "y"),
    "upper_eye_height_0.001": (float, "y"),
    "upper_eye_height_1e-06": (float, "y"),
    "upper_eye_height_1e-09": (float, "y"),
    "upper_eye_height_1e-12": (float, "y"),
    "upper_eye_height_1e-15": (float, "y"),
    "lower_eye_height_0.025": (float, "y"),
    "lower_eye_height_0.001": (float, "y"),
    "lower_eye_height_1e-06": (float, "y"),
    "lower_eye_height_1e-09": (float, "y"),
    "lower_eye_height_1e-12": (float, "y"),
    "lower_eye_height_1e-15": (float, "y"),
    "eye_width_0.025": (float, "t"),
    "eye_width_0.001": (float, "t"),
    "eye_width_1e-06": (float, "t"),
    "eye_width_1e-09": (float, "t"),
    "eye_width_1e-12": (float, "t"),
    "eye_width_1e-15": (float, "t"),
    "upper_eye_width_0.025": (float, "t"),
    "upper_eye_width_0.001": (float, "t"),
    "upper_eye_width_1e-06": (float, "t"),
    "upper_eye_width_1e-09": (float, "t"),
    "upper_eye_width_1e-12": (float, "t"),
    "upper_eye_width_1e-15": (float, "t"),
    "lower_eye_width_0.025": (float, "t"),
    "lower_eye_width_0.001": (float, "t"),
    "lower_eye_width_1e-06": (float, "t"),
    "lower_eye_width_1e-09": (float, "t"),
    "lower_eye_width_1e-12": (float, "t"),
    "lower_eye_width_1e-15": (float, "t"),
    "d_lev": (list, "y"),
}

# Statistically averaged quantities below
MEASUREMENT_COUNTS = {
    "threshold": (int, "y"),
    "zero_level_xp": (int, "y"),
    "zero_level_4140": (int, "y"),
    "zero_level_8180": (int, "y"),
    "one_level_xp": (int, "y"),
    "one_level_4140": (int, "y"),
    "one_level_8180": (int, "y"),
    "extinction_ratio_xp": (int, None),
    "extinction_ratio_4140": (int, None),
    "extinction_ratio_8180": (int, None),
    "oma_xp": (int, "y"),
    "oma_8180": (int, "y"),
    "oma_4140": (int, "y"),
    "average": (int, "y"),
    "rise_time_10-90_xp": (int, "t"),
    "fall_time_90-10_xp": (int, "t"),
    "rise_time_20-80_xp": (int, "t"),
    "fall_time_80-20_xp": (int, "t"),
    "rise_time_10-90_4140": (int, "t"),
    "fall_time_90-10_4140": (int, "t"),
    "rise_time_20-80_4140": (int, "t"),
    "fall_time_80-20_4140": (int, "t"),
    "rise_time_10-90_8180": (int, "t"),
    "fall_time_90-10_8180": (int, "t"),
    "rise_time_20-80_8180": (int, "t"),
    "fall_time_80-20_8180": (int, "t"),
    "dcd_xp": (int, "t"),
    "dcd_4140": (int, "t"),
    "dcd_8180": (int, "t"),
    "overshoot_percentage_xp": (int, "y"),
    "overshoot_percentage_4140": (int, "y"),
    "overshoot_percentage_8180": (int, "y"),
    "undershoot_percentage_xp": (int, "y"),
    "undershoot_percentage_4140": (int, "y"),
    "undershoot_percentage_8180": (int, "y"),
    "abs_overshoot_time_xp": (int, "t"),
    "abs_overshoot_time_4140": (int, "t"),
    "abs_overshoot_time_8180": (int, "t"),
    "abs_undershoot_time_xp": (int, "t"),
    "abs_undershoot_time_4140": (int, "t"),
    "abs_undershoot_time_8180": (int, "t"),
    "d_lev": (list, "y"),
    "abs_rise_time_10_xp": (int, "t"),
    "abs_rise_time_20_xp": (int, "t"),
    "abs_rise_time_50_xp": (int, "t"),
    "abs_rise_time_80_xp": (int, "t"),
    "abs_rise_time_90_xp": (int, "t"),
    "abs_fall_time_10_xp": (int, "t"),
    "abs_fall_time_20_xp": (int, "t"),
    "abs_fall_time_50_xp": (int, "t"),
    "abs_fall_time_80_xp": (int, "t"),
    "abs_fall_time_90_xp": (int, "t"),
    "abs_rise_time_10_4140": (int, "t"),
    "abs_rise_time_20_4140": (int, "t"),
    "abs_rise_time_50_4140": (int, "t"),
    "abs_rise_time_80_4140": (int, "t"),
    "abs_rise_time_90_4140": (int, "t"),
    "abs_fall_time_10_4140": (int, "t"),
    "abs_fall_time_20_4140": (int, "t"),
    "abs_fall_time_50_4140": (int, "t"),
    "abs_fall_time_80_4140": (int, "t"),
    "abs_fall_time_90_4140": (int, "t"),
    "abs_rise_time_10_8180": (int, "t"),
    "abs_rise_time_20_8180": (int, "t"),
    "abs_rise_time_50_8180": (int, "t"),
    "abs_rise_time_80_8180": (int, "t"),
    "abs_rise_time_90_8180": (int, "t"),
    "abs_fall_time_10_8180": (int, "t"),
    "abs_fall_time_20_8180": (int, "t"),
    "abs_fall_time_50_8180": (int, "t"),
    "abs_fall_time_80_8180": (int, "t"),
    "abs_fall_time_90_8180": (int, "t"),
}
