import numpy as np
from scipy.stats import kstest
from colorama import Fore, Style
from .eye import Eye
from .units import ureg


class CustomEye(Eye):
    def __init__(
        self,
        datarate_gbps: float,
        dt_sec: float,
        scale_y: float = 2,
        nx: int = 256,
        ny: int = 1024,
        format: str = "NRZ",
        hdf5_path: str = "eye.h5",
        dump_to_hdf5: bool = False,
        num_bits_to_filter_on_before: int = 3,
        num_bits_to_filter_on_after: int = 1,
    ):
        Eye.__init__(
            self,
            datarate_gbps,
            dt_sec,
            scale_y,
            nx,
            ny,
            format,
            hdf5_path,
            sampling_offset_mode="custom",
            dump_to_hdf5=dump_to_hdf5,
            num_bits_to_filter_on_before=num_bits_to_filter_on_before,
            num_bits_to_filter_on_after=num_bits_to_filter_on_after,
        )
        self.__time = None

    @Eye.init_check
    def add_data(
        self,
        wvf: np.ndarray,
        wvf_units: str,
        sampling_times: np.ndarray,
        sampling_times_units: str,
    ) -> None:
        wvf_array = self.convert_to_default_units(wvf, wvf_units)
        if self.__time is None:
            self.__time = np.arange(len(wvf_array)) * self.dt
        else:
            self.__time += len(wvf_array) * self.dt

        st = ureg.Quantity(sampling_times, sampling_times_units)
        st = st.m_as(self.x_units)

        if not self.are_times_normally_distributed(st, self.period):
            print(
                Fore.RED
                + Style.BRIGHT
                + "Warning! Sampling times don't seem to be normally distributed. "
                + "Please use caution when interpreting BER values computed from this."
                + Style.RESET_ALL
            )

        if (np.max(st) > np.max(self.__time)) or (np.min(st) < np.min(self.__time)):
            raise ValueError(
                "Warning! Cannot have sampling times out of time range"
                f" {np.min(self.__time)} - {np.max(self.__time)}.\n"
                f"Input sampling times min/max was {np.min(st)}/{np.max(st)}"
            )

        self.process_data(
            wvf=wvf_array,
            time=self.__time,
            sampling_times=st,
        )

    def are_times_normally_distributed(self, st: np.ndarray, period: float) -> bool:
        # take circular average
        phase = st * 2 * np.pi / period
        xlist, ylist = np.cos(phase), np.sin(phase)
        phase_list = np.arctan2(ylist, xlist)
        phase_list += np.pi - np.mean(phase_list)  # center
        st_wrapped = phase_list * period / (2 * np.pi)
        _, p_value = kstest(st_wrapped, "norm")
        return p_value > 0.05
