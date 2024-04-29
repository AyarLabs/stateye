from colorama import Fore, Back, Style
import time as _time


def timer(func, print_status=False):
    def function_wrapper(*args, **kwargs):
        if print_status:
            print(
                "Running function: "
                + Fore.MAGENTA
                + Style.BRIGHT
                + str(func.__name__)
                + Style.RESET_ALL
            )
            start = _time.perf_counter()
        rvals = func(*args, **kwargs)
        if print_status:
            print(
                "Elapsed time: "
                + Fore.CYAN
                + Style.BRIGHT
                + str(_time.perf_counter() - start)
                + " sec"
                + Style.RESET_ALL
            )
        return rvals

    return function_wrapper
