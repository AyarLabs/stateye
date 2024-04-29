import numpy as np
from abc import abstractmethod
import h5py
from colorama import Fore, Style
from .xiaolin_wu import quick_draw
from .measurements.waveform_analysis import nrz_waveform_analysis
from .measurements.histogram_analysis import nrz_histogram_analysis
from .measurements.bathtub import generate_vertical_bathtub, to_q_scale
import matplotlib.pyplot as plt
import matplotlib
from typing import Optional, List, Callable, Dict
from scipy.interpolate import interp1d
from scipy.ndimage import convolve1d
from .measurement_names import MEASUREMENTS, MEASUREMENT_COUNTS
from .metadata_names import METADATA
from .units import ureg
import copy
from functools import reduce

"""
Ayar Labs
Derek M. Kita
"""

DEFAULT_TIME_UNITS = "psec"
DEFAULT_VOLTAGE_UNITS = "mV"
DEFAULT_POWER_UNITS = "mW"
DEFAULT_CURRENT_UNITS = "mA"
ALLOWED_SAMPLING_OFFSET_MODES = ["adaptive", "custom", "half_ui"]


class Eye:
    """
    Eye class containing information about a waveform, stored in a Nx2D arrays
    (histograms), where N is the number of unique data patterns the waveform
    is filtered on.  For example, if the filter width is 3 and the format is NRZ,
    there will be 2^3=8 (2^N) eye histograms generated.

    Args:
        - ``datarate_gbps`` (`float`): Baudrate of the signal (gigabaud).
        - ``dt_sec`` (`float`): Time step between subsequent samples.
        - ``scale_y`` (`float`): Scaling coefficient for the size of the
            histogram y-direction, relative to the maximum range seen from the
            first data added.  For example, a value of 1 scales the histogram
            range such that it spans only from ymin to ymax for that first
            waveform added.
        - ``nx`` (`int`): Number of time points per UI in the eye diagram.
        - ``ny`` (`int`): Number of amplitude points in the eye diagram.
        - ``format`` (`str`): Symbol format.  Either 'NRZ' or 'PAM4'.
        - ``hdf5_path`` (`str`): Name of the hdf5 file that data is dumped to.
        - ``sampling_offset_mode`` (`str`):
        - ``dump_to_hdf5`` (`bool`): Flag to create an hdf5 file for dumping
            eye data.  As an alternative, you can make direct calls to Eye.save_copy(filepath)
        - ``num_bits_to_filter_on_before`` (`int`): Number of bits before data
            to filter on.  *CAUTION* use the smallest number of bits possible,
            as this will rapidly require large amounts of computer memory.
        - ``num_bits_to_filter_on_after`` (`int`): Number of bits after data
            to filter on.  *CAUTION* use the smallest number of bits possible,
            as this will rapidly require large amounts of computer memory.

    """

    def __init__(
        self,
        datarate_gbps: float,
        dt_sec: float,
        scale_y: float,
        nx: int,
        ny: int,
        format: str,
        hdf5_path: str,
        sampling_offset_mode: str,
        dump_to_hdf5: bool,
        num_bits_to_filter_on_before: int = 3,
        num_bits_to_filter_on_after: int = 1,
    ) -> None:
        assert scale_y > 1
        self.initialized = False
        self.nx = nx
        self.ny = ny
        self.scale_y = scale_y
        if format not in ["NRZ", "PAM4"]:
            raise Exception(f"format is an invalid value ({format})")
        self.format = format
        self.x_units = DEFAULT_TIME_UNITS

        _period = 1.0 / ureg.Quantity(datarate_gbps, "Gbps")
        self.period = _period.m_as(self.x_units)

        _dt = ureg.Quantity(dt_sec, "sec")
        self.dt = _dt.m_as(self.x_units)

        self.samples_per_symbol = int(round(self.period / self.dt))

        # Allocate memory for the eye histogram
        if not isinstance(num_bits_to_filter_on_after, int):
            raise ValueError(f"num_bits_to_filter_on_after must be an int.  Got: {num_bits_to_filter_on_after}")
        if not isinstance(num_bits_to_filter_on_before, int):
            raise ValueError(f"num_bits_to_filter_on_before must be an int.  Got: {num_bits_to_filter_on_before}")
        self.filter_bits_before = num_bits_to_filter_on_before
        self.filter_bits_after = num_bits_to_filter_on_after
        self.filter_width = num_bits_to_filter_on_before + num_bits_to_filter_on_after + 1

        if format == "NRZ":
            self.nz = 2**self.filter_width
            levels = [[0], [1]]
        elif format == "PAM4":
            self.nz = 4**self.filter_width
            levels = [[0], [1], [2], [3]]

        self.hist_data = np.zeros(
            shape=(self.nx, self.ny, self.nz),
            dtype=np.double,
        )
        self.bathtub = np.zeros(shape=(self.nx, self.ny), dtype=np.double)
        self.raw_bathtub = np.zeros(shape=(self.nx, self.ny), dtype=np.double)

        # Generate the corresponding filter patterns
        def permute_two_lists(l1, l2):
            return [_l1 + _l2 for _l1 in l1 for _l2 in l2]

        self.data_patterns = reduce(permute_two_lists, [levels for _ in range(self.filter_width)])
        # Map the histogram index (3rd dim) to the logical bit value
        self.data_values = [dp[self.filter_bits_before] for dp in self.data_patterns]

        # Initialize x-axis properties
        self.dx = self.period / nx
        self.x_axis = np.arange(0, self.period, self.dx)

        self.hdf5_path = hdf5_path
        self.dump_to_hdf5 = dump_to_hdf5

        self.num_bits = 0
        self.__msmts = {k: np.nan for k, v in MEASUREMENTS.items() if v[0] is not list}
        self.__msmt_counts = {k: 0 for k, _ in MEASUREMENTS.items()}
        self.__compute_histogram_analysis = False

        self.slicer_sensitivity = 0

        self.d_lev_fbefore = [dp[: self.filter_bits_before] for dp in self.data_patterns]
        self.d_lev_values = [dp[self.filter_bits_before] for dp in self.data_patterns]
        self.d_lev_fafter = [dp[self.filter_bits_before + 1 :] for dp in self.data_patterns]

        self.pattern_indices = np.array([])
        self.pattern_counts = np.array([])

        if sampling_offset_mode not in ALLOWED_SAMPLING_OFFSET_MODES:
            raise Exception(f"Warning! sampling_offset_mode ({sampling_offset_mode}) not in allowed values: {ALLOWED_SAMPLING_OFFSET_MODES}")
        self.sampling_offset_mode = sampling_offset_mode

        self.tdec_s_noise = 0.0
        self.tdec_m1 = 0.0
        self.tdec_m2 = 0.0
        self.tdec_ber = 1e-12

    @abstractmethod
    def add_data(self):
        pass

    def init_check(add_data_func: Callable) -> Callable:
        # Check eye histogram has been initialized
        def function_wrapper(self, wvf, wvf_units, *args, **kwargs):
            if not self.initialized:
                self.__initialize_eye(wvf, wvf_units)
            add_data_func(self, wvf, wvf_units, *args, **kwargs)

        return function_wrapper

    def convert_to_default_units(self, wvf: np.ndarray, wvf_units: str) -> np.ndarray:
        wvf = ureg.Quantity(wvf, wvf_units)
        return wvf.m_as(self.y_units)

    def __initialize_eye(self, wvf: np.ndarray, wvf_units: str) -> None:
        _wvf = ureg.Quantity(wvf, wvf_units)
        if _wvf.is_compatible_with("V"):
            _wvf = _wvf.m_as(DEFAULT_VOLTAGE_UNITS)
            self.y_units = DEFAULT_VOLTAGE_UNITS
        elif _wvf.is_compatible_with("A"):
            _wvf = _wvf.m_as(DEFAULT_CURRENT_UNITS)
            self.y_units = DEFAULT_CURRENT_UNITS
        elif _wvf.is_compatible_with("W"):
            _wvf = _wvf.m_as(DEFAULT_POWER_UNITS)
            self.y_units = DEFAULT_POWER_UNITS
        else:
            raise Exception(f"'wvf_units' not supported: {wvf_units}")

        _range = np.max(_wvf) - np.min(_wvf)
        if _range == 0:
            _range = 1e-6
        self.ymax = np.max(_wvf) + _range * (0.5 * (self.scale_y - 1))
        self.ymin = np.min(_wvf) - _range * (0.5 * (self.scale_y - 1))

        self.dy = (self.ymax - self.ymin) / self.ny
        if self.dy != 0:
            self.y_axis = np.arange(self.ymin, self.ymax, self.dy)
        else:
            self.y_axis = np.arange(self.ymin - 1e-6, self.ymax + 1e-6, 1e-7)

        # Initialize some of the measurement parameters
        self.threshold_initialized = False
        self.threshold = np.mean(_wvf)  # will be updated later from the first waveform
        self.ymax_plot = self.threshold
        self.ymin_plot = self.threshold

        # Create file to dump eye to
        if self.dump_to_hdf5:
            self.__create_hdf5_dataset()

        self.initialized = True

    def __to_x_grid(self, x: np.ndarray) -> np.ndarray:
        return x * (self.nx / self.period)

    def __to_y_grid(self, y: np.ndarray) -> np.ndarray:
        return (y - self.ymin) * (self.ny / (self.ymax - self.ymin))

    def process_data(
        self,
        wvf: np.ndarray,
        time: np.ndarray,
        sampling_times: np.ndarray,
    ) -> None:
        # recompute histogram metrics, since data is getting added
        self.__compute_histogram_analysis = True

        # keep track of the bits processed
        n_cur_bits = len(sampling_times)
        self.num_bits += n_cur_bits

        """
        Measurements on waveform
        """
        # Convert to index units, time_idx = [0,1,2,3,...]
        # These do *NOT* have to be integers
        sampling_indices = (sampling_times - time[0]) / self.dt

        msmts, msmt_counts = nrz_waveform_analysis(
            time,
            wvf,
            sampling_indices,
            self.threshold,
            self.period
        )
        if not self.threshold_initialized:
            if not np.isnan(msmts["threshold"]):
                self.threshold = msmts["threshold"]  # update sampling threshold
            self.threshold_initialized = True

        # Update average waveform measurement values
        for k, m in msmts.items():
            if not isinstance(m, list):
                if msmt_counts[k] == 0:
                    continue
                prev_counts = self.__msmt_counts.get(k, 0)
                prev_value = self.__msmts.get(k, 0)
                if np.isnan(prev_value):
                    prev_value, prev_counts = 0, 0
                nv = prev_value * prev_counts + m * msmt_counts[k]
                if msmt_counts[k] != 0:
                    nv /= prev_counts + msmt_counts[k]
                self.__msmts.update({k: nv})
                self.__msmt_counts.update({k: prev_counts + msmt_counts[k]})
            else:
                # update average of d_lev measurements (which is a list)
                if k not in self.__msmts:  # initialize empty lists
                    self.__msmts[k] = [np.nan] * len(self.d_lev_values)
                    self.__msmt_counts[k] = [0] * len(self.d_lev_values)
                prev_counts = self.__msmt_counts[k]
                prev_value = self.__msmts[k]
                nv_list, newcount_list = [], []
                for i, subm in enumerate(m):
                    if msmt_counts[k][i] == 0:
                        continue
                    if np.isnan(prev_value[i]):
                        prev_value[i], prev_counts[i] = 0, 0
                    nv = prev_value[i] * prev_counts[i] + subm * msmt_counts[k][i]
                    if msmt_counts[k][i] != 0:
                        nv /= prev_counts[i] + msmt_counts[k][i]
                    nv_list.append(nv)
                    newcount_list.append(prev_counts[i] + msmt_counts[k][i])
                self.__msmts.update({k: nv_list})
                self.__msmt_counts.update({k: newcount_list})

        # Update largest/smallest observed y values (for plotting)
        self.ymin_plot = min(self.ymin_plot, np.min(wvf))
        self.ymax_plot = max(self.ymax_plot, np.max(wvf))

        """
        Update the histogram array with the provided waveform.
        """

        # Convert waveform to grid units
        wvf_grid = self.__to_y_grid(wvf)
        time_grid = self.__to_x_grid(time)
        sampling_times_grid = self.__to_x_grid(sampling_times)

        # In order for this to work, the sampling_times_grid must be strictly
        # monotonically increasing in time
        if not np.all(sampling_times_grid[1:] >= sampling_times_grid[:-1]):
            raise Exception(f"ERROR: Sampling times grid is not monotonically increasing!")

        # Mid points
        m = (sampling_times_grid[1:] + sampling_times_grid[:-1]) / 2

        # Fracture points (rounded), and put them in the array
        f = np.ones(len(sampling_times_grid), dtype=float) * len(time_grid)
        f[:-1] = np.round(m).astype(int)

        # Find the pattern indices for each sampling instant.
        sampling_values = np.interp(sampling_times, time, wvf)
        if self.format == "NRZ":

            def determine_bit(sv: float) -> int:
                if sv > self.threshold:
                    return 1
                else:
                    return 0

        else:  # "PAM4"
            raise NotImplementedError("TODO")
        data_values = [determine_bit(sv) for sv in sampling_values]
        pattern_indices = -1 * np.ones(len(sampling_times), dtype=int)
        for i in range(len(pattern_indices) - self.filter_width):
            data_idx = i + self.filter_bits_before
            pattern_indices[data_idx] = self.data_patterns.index(
                data_values[i : i + self.filter_width]
            )

        self.pattern_indices, self.pattern_counts = np.unique(pattern_indices, return_counts=True)
        # remove the -1 values (samples not added to histogram)
        _idx = np.where(self.pattern_indices == -1)
        self.pattern_indices = np.delete(self.pattern_indices, _idx[0][0])
        self.pattern_counts = np.delete(self.pattern_counts, _idx[0][0])

        assert self.hist_data.flags["C_CONTIGUOUS"]
        # Draw points on the histogram grid
        quick_draw(
            time_grid,
            wvf_grid,
            sampling_times_grid,
            f,
            pattern_indices,
            self.hist_data,
        )

    def add_vertical_noise(
        self,
        noise_distribution: np.ndarray,
        y_values: np.ndarray,
        y_values_unit: str,
    ) -> None:
        """Convolves a user-specified kernel with the eye diagram in the vertical
        direction.
        """
        yv = ureg.Quantity(y_values, y_values_unit).m_as(self.y_units)
        # Interpolate the CDF onto the same y-grid
        cdf = np.cumsum(noise_distribution / np.sum(noise_distribution))
        cdf_inv_func = interp1d(
            cdf, yv, bounds_error=False, fill_value=(min(yv), max(yv))
        )
        pts = cdf_inv_func(np.random.rand(self.num_bits))
        # Then use the inverse to sample points with this noise
        minyv = max(min(yv), self.ymin)
        maxyv = min(max(yv), self.ymax)
        noise_kernel, _ = np.histogram(
            pts,
            bins=np.arange(minyv - 0.5 * self.dy, maxyv + 1.5 * self.dy, self.dy),
        )
        noise_kernel = noise_kernel.astype(float)
        noise_kernel[np.nonzero(noise_kernel)] /= np.sum(noise_kernel)

        # Convolve the sampled vertical noise kernel with the eye histogram
        self.hist_data = convolve1d(self.hist_data, noise_kernel, axis=1, mode="wrap")

        self.__compute_histogram_analysis = True

    def scale_eye(self, gain: float) -> None:
        self.y_axis *= gain
        self.ymin *= gain
        self.ymax *= gain
        self.dy *= gain
        self.ymin_plot *= gain
        self.ymax_plot *= gain
        self.threshold *= gain

        self.__compute_histogram_analysis = True

    def add_jitter_distribution(
        self, jitter_times_psec: np.ndarray, jitter_distribution_values: np.ndarray
    ) -> None:
        jt = ureg.Quantity(jitter_times_psec, "psec").m_as(self.x_units)

        cdf = np.cumsum(jitter_distribution_values / np.sum(jitter_distribution_values))
        cdf_inv_func = interp1d(
            cdf, jt, bounds_error=False, fill_value=(min(jt), max(jt))
        )
        pts = cdf_inv_func(np.random.rand(self.num_bits))
        # Then use the inverse to sample points with this noise
        jitter_kernel, _ = np.histogram(
            pts,
            bins=np.arange(min(jt) - 1.0 * self.dx, max(jt) + 1.0 * self.dx, self.dx),
        )
        jitter_kernel = jitter_kernel.astype(float)
        jitter_kernel[np.nonzero(jitter_kernel)] /= np.sum(jitter_kernel)

        # Convolve the jitter kernel with the eye histogram
        self.hist_data = convolve1d(self.hist_data, jitter_kernel, axis=0, mode="wrap")

        self.__compute_histogram_analysis = True

    def set_slicer_sensitivity(
        self, sensitivity: float, sensitivity_units: str
    ) -> None:
        self.__compute_histogram_analysis = True
        ss = ureg.Quantity(sensitivity, sensitivity_units)
        self.slicer_sensitivity = ss.m_as(self.y_units)

    def add_flat_bounded_jitter(
        self, jitter_width: float, jitter_width_units: str
    ) -> None:
        jw = ureg.Quantity(jitter_width, jitter_width_units).m_as(self.x_units)
        jt = np.arange(-jw, jw, self.dx)
        jtu = ureg.Quantity(jt, self.x_units)
        j_d = np.zeros(len(jt))
        j_d[(jt >= -jw / 2) & (jt < jw / 2)] = 1
        self.add_jitter_distribution(
            jitter_times_psec=jtu.m_as("psec"),
            jitter_distribution_values=j_d,
        )

    def add_random_jitter(self, jitter_std: float, jitter_std_units: str) -> None:
        j_std = ureg.Quantity(jitter_std, jitter_std_units).m_as(self.x_units)
        jt = np.arange(-self.period / 4, self.period / 4, self.dx)
        jtu = ureg.Quantity(jt, self.x_units)
        j_d = np.exp(-(jt**2) / (2 * j_std**2)) / np.sqrt(2 * np.pi * j_std**2)
        self.add_jitter_distribution(
            jitter_times_psec=jtu.m_as("psec"),
            jitter_distribution_values=j_d,
        )

    def get_measurements(
        self,
        force_compute_histogram_statistics: bool = False,
    ) -> Dict[str, float]:
        if self.__compute_histogram_analysis or force_compute_histogram_statistics:
            generate_vertical_bathtub(
                hist=self.hist_data,
                bathtub=self.bathtub,
                raw_bathtub=self.raw_bathtub,
                pattern_indices=self.pattern_indices,
                pattern_counts=self.pattern_counts,
                data_values=self.data_values,
                sensitivity=self.slicer_sensitivity,
                y_scale=self.y_axis,
            )

            nrz_histogram_analysis(
                self.__msmts,
                self.__msmt_counts,
                self.hist_data,
                self.bathtub,
                self.ymin,
                self.ymax,
                self.dx,
                self.dy,
                self.threshold,
                self.pattern_counts,
                self.tdec_s_noise,
                self.tdec_m1,
                self.tdec_m2,
                self.tdec_ber,
            )
            self.__compute_histogram_analysis = False
        return self.__msmts

    def get_measurement_counts(self) -> Dict[str, int]:
        return self.__msmt_counts

    def get_current_measurements(self) -> Dict[str, float]:
        return self.__msmts

    def get_current_measurement_units(self) -> Dict[str, Optional[str]]:
        units = self.get_measurement_units()
        return {k: units[k] for k in self.__msmts.keys()}

    @property
    def msmts(self):
        return self.get_measurements()

    def get_measurement_units(self) -> Dict[str, Optional[str]]:
        units = {}
        for mname, (_, dir) in MEASUREMENTS.items():
            if dir == "t":
                units.update({mname: str(self.x_units)})
            elif dir == "y":
                units.update({mname: str(self.y_units)})
            elif dir == "":  # dimensionless measurements
                units.update({mname: ""})
            elif dir is None:
                units.update({mname: None})
            else:
                raise Exception(f"Unrecognized direction: {dir} ({mname})")
        return units
    
    def set_tdec_s_noise(self, s: float) -> None:
        self.tdec_s_noise = float(s)
        self.__compute_histogram_analysis = True

    def set_tdec_m1(self, m1: float) -> None:
        self.tdec_m1 = float(m1)
        self.__compute_histogram_analysis = True

    def set_tdec_m2(self, m2: float) -> None:
        self.tdec_m2 = float(m2)
        self.__compute_histogram_analysis = True

    def set_tdec_ber(self, ber: float) -> None:
        self.tdec_ber = float(ber)
        self.__compute_histogram_analysis = True

    def optimize_sampling_time(self, ber: float = 1e-12, max_offset_ui: float = 0.3) -> None:
        """
        Optimize the sampling time to minimize the BER
        """
        heights = np.array([np.sum(column < ber) for column in self.bathtub])
        beyond_max_offset = (np.abs(np.arange(0, 1, 1 / len(heights)) - 0.5) > max_offset_ui)
        heights[beyond_max_offset] = 0
        sidx = round(self.bathtub.shape[0] / 2)
        self.hist_data = np.roll(
            self.hist_data, round(sidx - np.argmax(heights)), axis=0
        )
        self.bathtub = np.roll(
            self.bathtub, round(sidx - np.argmax(heights)), axis=0
        )
        self.raw_bathtub = np.roll(
            self.raw_bathtub, round(sidx - np.argmax(heights)), axis=0
        )
        self.get_measurements(force_compute_histogram_statistics=True)

    def get_dlev(self, fb: List[int], v: float, fa: List[int]) -> float:
        """
        Get the d_lev value for a given pattern, relative to the threshold.
        """
        dl, counts = [], []
        m, c = self.get_measurements(), self.get_measurement_counts()
        for idx in range(len(self.d_lev_values)):
            fbmatch = fb == self.d_lev_fbefore[idx][len(self.d_lev_fbefore[idx])-len(fb):]
            vmatch = v == self.d_lev_values[idx]
            famatch = fa == self.d_lev_fafter[idx][:len(fa)]
            if fbmatch and vmatch and famatch:
                dl.append(m["d_lev"][idx] - m["threshold"])
                counts.append(c["d_lev"][idx])
        if len(dl) == 0:
            raise Exception(f"Could not find d_lev! fb={fb}, v={v}, fa={fa}.")
        return np.dot(dl, counts) / np.sum(counts)  # return the weighted avg

    def plot(
        self,
        show: bool = True,
        show_contours: bool = False,
        show_contour_label: bool = False,
        ber_thresholds: List[float] = [],
        bw: bool = False,
        pattern: Optional[List[int]] = None,
    ) -> plt.Figure:
        """
        Eye plotting function
        """
        if not self.initialized:
            raise Exception(
                "Cannot plot an eye with no data.  Please add data first with the add_data() method."
            )

        # Run hist analysis (if it hasn't run already), to get the ber countours
        _ = self.get_measurements()

        hist = copy.deepcopy(self.hist_data)

        if pattern is None:
            hist = np.sum(hist, axis=2)
        else:
            if pattern not in self.data_patterns:
                raise ValueError()
            hist = hist[:, :, self.data_patterns.index(pattern)]

        hist[hist < 1e-1] = 1e-2

        if not bw:
            _cmap = copy.copy(
                matplotlib.cm.get_cmap("inferno")
            )  # was "coolwarm" before
            _cmap.set_under("k")
            norm = matplotlib.colors.LogNorm(vmin=0.1)
        else:
            _cmap = copy.copy(matplotlib.cm.get_cmap("Greys"))
            _cmap.set_under("w")
            norm = matplotlib.colors.LogNorm(vmax=1.0)

        fig = plt.figure(figsize=(10, 5))
        plt.grid(color="gray")
        plt.imshow(
            np.roll(
                np.concatenate(
                    (hist.T[::-1], hist.T[::-1]),
                    axis=1,
                ),
                shift=self.nx // 2,
            ),
            aspect="auto",
            cmap=_cmap,
            norm=norm,
            extent=[-self.period, self.period, self.ymin, self.ymax],
        )
        plt.colorbar()
        if show_contours:
            levels = sorted(ber_thresholds)
            cs = plt.contour(
                np.roll(
                    np.concatenate(
                        (self.bathtub.T, self.bathtub.T),
                        axis=1,
                    ),
                    shift=self.nx // 2,
                ),
                levels,
                colors="w",
                origin="lower",
                extent=[-self.period, self.period, self.ymin, self.ymax],
            )
            if show_contour_label:
                plt.clabel(cs, inline=1, fontsize=8, fmt={lv: str(lv) for lv in levels})
        _range = self.ymax_plot - self.ymin_plot
        plt.ylim([self.ymin_plot - 0.025 * _range, self.ymax_plot + 0.025 * _range])
        plt.xlabel(f"time [{str(self.x_units)}]")
        plt.ylabel(f"signal [{str(self.y_units)}]")
        plt.tight_layout()
        if show:
            plt.show()

        return fig

    def plot_bathtub(
        self,
        show: bool = True,
        raw_bathtub: bool = False,
    ) -> plt.Figure:
        """
        Bathtub plotting function
        """
        if not self.initialized:
            raise Exception(
                "Cannot plot an eye with no data.  Please add data first with the add_data() method."
            )
        b = self.raw_bathtub if raw_bathtub else self.bathtub

        # Run hist analysis (if it hasn't run already)
        _ = self.get_measurements()

        _cmap = copy.copy(matplotlib.cm.get_cmap("inferno"))  # was "coolwarm" before
        _cmap.set_under("k")
        norm = matplotlib.colors.LogNorm(vmin=1e-15)

        fig = plt.figure(figsize=(10, 5))
        plt.grid(color="gray")
        plt.imshow(
            np.roll(
                np.concatenate(
                    (b.T[::-1], b.T[::-1]),
                    axis=1,
                ),
                shift=self.nx // 2,
            ),
            aspect="auto",
            cmap=_cmap,
            norm=norm,
            extent=[-self.period, self.period, self.ymin, self.ymax],
        )
        plt.colorbar()
        levels = [1e-15, 1e-12, 1e-9, 1e-6, 1e-3]
        cs = plt.contour(
            np.roll(
                np.concatenate(
                    (self.bathtub.T, self.bathtub.T),
                    axis=1,
                ),
                shift=self.nx // 2,
            ),
            levels,
            colors="k",
            origin="lower",
            extent=[-self.period, self.period, self.ymin, self.ymax],
        )
        plt.clabel(cs, inline=1, fontsize=8, fmt={lv: str(lv) for lv in levels})
        _range = self.ymax_plot - self.ymin_plot
        plt.ylim([self.ymin_plot - 0.025 * _range, self.ymax_plot + 0.025 * _range])
        plt.xlabel(f"time [{str(self.x_units)}]")
        plt.ylabel(f"signal [{str(self.y_units)}]")
        plt.tight_layout()
        if show:
            plt.show()

        return fig

    def plot_bathtub_cross_section(
        self,
        show: bool = True,
        direction: str = "horizontal",
        y_axis: str = "ber",
    ) -> plt.Figure:
        assert direction in ["horizontal", "vertical"]
        assert y_axis in ["q-scale", "ber"]
        if not self.initialized:
            raise Exception(
                "Cannot plot an eye with no data.  Please add data first with the add_data() method."
            )
        
        # Run hist analysis (if it hasn't run already)
        _ = self.get_measurements()

        f = (self.threshold - self.ymin) / (self.ymax - self.ymin)
        tidx = round(f * self.hist_data.shape[1])  # threshold index (y)
        sidx = round(self.bathtub.shape[0] / 2)  # sampling time index (x)

        fig = plt.figure(figsize=(10, 5))
        plt.grid(color="gray")
        if y_axis == "ber":
            if direction == "horizontal":
                plt.title("Horizontal cross section")
                mv = np.min([np.min(self.raw_bathtub[:, tidx]), np.min(self.bathtub[:, tidx])])
                plt.semilogy(
                    self.x_axis,
                    self.raw_bathtub[:, tidx],
                    'r.',
                    label="raw bathtub",
                )
                plt.semilogy(
                    self.x_axis,
                    self.bathtub[:, tidx],
                    label="bathtub fit",
                )
                plt.xlabel(f"time [{str(self.x_units)}]")
            else:
                plt.title("Vertical cross section")
                mv = np.min([np.min(self.raw_bathtub[sidx, :]), np.min(self.bathtub[sidx, :])])
                plt.semilogy(
                    self.y_axis,
                    self.raw_bathtub[sidx, :],
                    'r.',
                    label="raw bathtub",
                )
                plt.semilogy(
                    self.y_axis,
                    self.bathtub[sidx, :],
                    label="bathtub fit",
                )
                plt.xlabel(f"signal [{str(self.y_units)}]")
            plt.ylim([np.max([mv, 1e-15]), 1.0])
            plt.ylabel(f"bit error rate (BER)")
        else:  # q-scale
            if direction == "horizontal":
                plt.title("Horizontal cross section")
                lower_q = to_q_scale(self.bathtub[:sidx, tidx], np.max(self.bathtub[:sidx, tidx]))
                upper_q = to_q_scale(self.bathtub[sidx:, tidx], np.max(self.bathtub[sidx:, tidx]))
                lower_q_raw = to_q_scale(self.raw_bathtub[:sidx, tidx], np.max(self.raw_bathtub[:sidx, tidx]))
                upper_q_raw = to_q_scale(self.raw_bathtub[sidx:, tidx], np.max(self.raw_bathtub[sidx:, tidx]))
                mv = np.max([np.max(lower_q), np.max(upper_q), np.max(lower_q_raw), np.max(upper_q_raw)])
                plt.plot(
                    self.x_axis,
                    np.append(lower_q_raw, upper_q_raw),
                    'r.',
                    label="raw bathtub",
                )
                plt.plot(
                    self.x_axis,
                    np.append(lower_q, upper_q),
                    label="bathtub fit",
                )
                plt.xlabel(f"time [{str(self.x_units)}]")
            else:
                plt.title("Vertical cross section")
                lower_q = to_q_scale(self.bathtub[sidx, :tidx], np.max(self.bathtub[sidx, :tidx]))
                upper_q = to_q_scale(self.bathtub[sidx, tidx:], np.max(self.bathtub[sidx, tidx:]))
                lower_q_raw = to_q_scale(self.raw_bathtub[sidx, :tidx], np.max(self.raw_bathtub[sidx, :tidx]))
                upper_q_raw = to_q_scale(self.raw_bathtub[sidx, tidx:], np.max(self.raw_bathtub[sidx, tidx:]))
                mv = np.max([np.max(lower_q), np.max(upper_q), np.max(lower_q_raw), np.max(upper_q_raw)])
                plt.plot(
                    self.y_axis,
                    np.append(lower_q_raw, upper_q_raw),
                    'r.',
                    label="raw bathtub",
                )
                plt.plot(
                    self.y_axis,
                    np.append(lower_q, upper_q),
                    label="bathtub fit",
                )
                plt.xlabel(f"signal [{str(self.y_units)}]")
            plt.ylim([0, np.min([mv, 7.9])])
            plt.gca().invert_yaxis()
            plt.ylabel(f"Q-scale")
        plt.legend()
        plt.tight_layout()
        if show:
            plt.show()

        return fig


    def set_hdf5_path(self, path: str) -> None:
        if self.initialized:
            raise ValueError(
                f"Cannot update path to {path}, eye has already been initialized."
            )
        self.hdf5_path = path

    def __create_hdf5_dataset(self) -> None:
        """
        Initialize the eye diagram dataset
        """
        f = h5py.File(self.hdf5_path, "w")

        f.create_dataset(
            "eye",
            (self.nx, self.ny, self.nz),
            dtype=np.double,
            compression="gzip",
        )
        for array_name in ["bathtub", "raw_bathtub"]:
            f.create_dataset(
                array_name,
                (self.nx, self.ny),
                dtype=np.double,
                compression="gzip",
            )
        meta = f.create_group("metadata")
        for mv in METADATA:
            meta.create_dataset(mv, data=getattr(self, mv))
        f.create_group("measurements")
        f.create_group("measurement_counts")
        f.close()

    def dump_hdf5_dataset(self) -> None:
        """
        Save contents of eye to disk
        """
        f = h5py.File(self.hdf5_path, "r+")
        f["eye"][:] = self.hist_data
        f["bathtub"][:] = self.bathtub
        f["raw_bathtub"][:] = self.raw_bathtub
        for mv in METADATA:
            f["metadata"][mv][()] = getattr(self, mv)
        for mname, (_type, _) in MEASUREMENTS.items():
            if mname not in self.__msmts.keys():
                continue
            if mname not in f["measurements"]:
                f["measurements"].create_dataset(mname, data=self.__msmts[mname])
            else:
                if (_type == np.ndarray) or (_type == list):
                    f["measurements"][mname][:] = self.__msmts[mname]
                elif _type == float:
                    f["measurements"][mname][()] = self.__msmts[mname]
                else:
                    raise ValueError(f"{_type} types not implemented yet.")
        for mname, (_type, dir) in MEASUREMENT_COUNTS.items():
            if mname not in self.__msmt_counts.keys():
                continue
            if mname not in f["measurement_counts"]:
                f["measurement_counts"].create_dataset(
                    mname, data=self.__msmt_counts[mname]
                )
            else:
                if _type == list:
                    f["measurement_counts"][mname][:] = self.__msmt_counts[mname]
                elif _type == int:
                    f["measurement_counts"][mname][()] = self.__msmt_counts[mname]
                else:
                    raise ValueError(f"{_type} types not implemented yet.")
        f.close()

    def load_hdf5_dataset(self, filename: str = "none") -> None:
        if filename == "none":
            filename = self.hdf5_path
        f = h5py.File(filename, "r")
        self.hist_data = np.array(f["eye"][:])
        self.bathtub = np.array(f["bathtub"][:])
        self.raw_bathtub = np.array(f["raw_bathtub"][:])
        for mv in METADATA:
            if isinstance(f["metadata"][mv][()], bytes):
                setattr(self, mv, f["metadata"][mv][()].decode("utf-8"))
            else:
                setattr(self, mv, f["metadata"][mv][()])
        for mname, (_type, _) in MEASUREMENTS.items():
            try:
                if (_type == np.ndarray) or (_type == list):
                    self.__msmts.update({mname: f["measurements"][mname][:]})
                elif _type == float:
                    self.__msmts.update({mname: f["measurements"][mname][()]})
                else:
                    raise ValueError(f"{_type} types not implemented yet.")
            except KeyError:
                print(
                    Fore.RED
                    + Style.BRIGHT
                    + f"Warning! Tried to load in parameter {mname}, but it didn't"
                    + f"exist in the hdf5 file ({filename}). Skipping for now."
                    + Style.RESET_ALL
                )
        for mname, (_type, _) in MEASUREMENT_COUNTS.items():
            try:
                if _type == int:
                    self.__msmt_counts.update(
                        {mname: f["measurement_counts"][mname][()]}
                    )
                elif _type == list:
                    self.__msmt_counts.update(
                        {mname: f["measurement_counts"][mname][:]}
                    )
                else:
                    raise ValueError(f"{_type} types not implemented yet.")
            except KeyError:
                print(
                    Fore.RED
                    + Style.BRIGHT
                    + f"Warning! Tried to load in measurement_counts {mname}, but it didn't"
                    + f"exist in the hdf5 file ({filename}). Skipping for now."
                    + Style.RESET_ALL
                )
        self.initialized = True

    def save_copy(self, filepath: str) -> None:
        original_path = self.hdf5_path
        self.hdf5_path = filepath
        self.__create_hdf5_dataset()
        self.dump_hdf5_dataset()
        self.hdf5_path = original_path  # reset
