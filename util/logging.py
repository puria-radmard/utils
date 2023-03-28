import os, time
from collections import deque

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


def time_to_string(timestamp):
    days = int(timestamp // 86400)
    hours = int(timestamp // 3600 % 24)
    minutes = int(timestamp // 60 % 60)
    seconds = int(timestamp % 60)
    return f"{days}d:{hours}h:{minutes}m:{seconds}s"


def configure_logging_paths(logging_directory):
    """
    Given a base log path, create a logging folder for this job (logging_directory with an index).
    Inside that create a txt file for string logging.
    """
    i = 0
    while i < 1000:
        try:
            curr_path = f"{logging_directory}_{i}"
            os.mkdir(curr_path)
            print_path = os.path.join(curr_path, "epoch_log.csv")
            warning_file = os.path.join(curr_path, "warnings.txt")
            return print_path, curr_path, warning_file
        except FileExistsError:
            i += 1
    raise Exception("Too many subfolders being created!")


def configure_state_path(logging_directory, filename):
    i = 1
    while i <= 1000:
        curr_path = os.path.join(logging_directory, f"{filename}{i}")
        if not os.path.exists(curr_path):
            return curr_path
        i += 1
    raise Exception("Too many subfolders being created!")


class LoopTimer:

    def __init__(self, total_loops, mem=10):
        self.times = deque(maxlen = mem)
        self.start_time = time.time()
        self.total_loops = total_loops
        self.num_loops = 0

    def loop_start(self):
        self.num_loops += 1
        self.loop_start_time = time.time()

    def loop_end(self):
        loop_duration = time.time() - self.loop_start_time
        self.times.append(loop_duration)
        average_loop_time = sum(self.times) / len(self.times)

        elapsed = time.time() - self.start_time
        elapsed_string = time_to_string(elapsed)
        remaining_string = time_to_string(average_loop_time * (self.total_loops - self.num_loops))
        return elapsed_string, remaining_string


