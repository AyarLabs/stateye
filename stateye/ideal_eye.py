import numpy as np
from .eye import Eye
from .ideal_cdr import lock_ideal_cdr


class IdealEye(Eye):
    def __init__(
        self,
        datarate_gbps: float,
        dt_sec: float,
        scale_y: float = 2,
        nx: int = 256,
        ny: int = 1024,
        format: str = "NRZ",
        hdf5_path: str = "eye.h5",
        sampling_offset_mode: str = "half_ui",
        dump_to_hdf5: bool = False,
        num_bits_to_filter_on_before: int = 3,
        num_bits_to_filter_on_after: int = 1,
    ):
        assert sampling_offset_mode in ["adaptive", "half_ui"]
        Eye.__init__(
            self,
            datarate_gbps,
            dt_sec,
            scale_y,
            nx,
            ny,
            format,
            hdf5_path,
            sampling_offset_mode,
            dump_to_hdf5,
            num_bits_to_filter_on_before,
            num_bits_to_filter_on_after,
        )

    @Eye.init_check
    def add_data(
        self,
        wvf: np.ndarray,
        wvf_units: str,
        sampling_offset: float = None,
    ) -> None:
        wvf_array = self.convert_to_default_units(wvf, wvf_units)
        times = np.arange(len(wvf_array)) * self.dt
        thr = np.mean(wvf_array)
        edge_offset, _sampling_offset = lock_ideal_cdr(
            time=times,
            wvf=wvf_array,
            threshold=thr,
            one_threshold=thr,
            zero_threshold=thr,
            samples_per_symbol=self.samples_per_symbol,
        )
        if self.sampling_offset_mode == "half_ui":
            self.sampling_offset = self.samples_per_symbol / 2
        elif self.sampling_offset_mode == "adaptive":
            if sampling_offset is not None:
                self.sampling_offset = sampling_offset
            else:
                if not hasattr(self, "sampling_offset"):
                    self.sampling_offset = _sampling_offset

        first_sample = (edge_offset + self.sampling_offset) % self.samples_per_symbol
        sampling_instants = np.arange(
            first_sample, len(wvf_array), self.samples_per_symbol
        )

        self.process_data(
            wvf=wvf_array,
            time=times,
            sampling_times=sampling_instants * self.dt,
        )
