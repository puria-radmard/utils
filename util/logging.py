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


def print_to_log(losses, batch_num, total_batches, elapsed_string, remaining_string, print_path):
    print_row = [f"{batch_num + 1} / {total_batches}"]
    print_row.extend(losses)
    print_row.append(f"{elapsed_string}    ({remaining_string})")
    with open(print_path, 'a') as f:
        print(*print_row, sep='\t', file = f)



def time_to_string(timestamp):
    days = int(timestamp // 86400)
    hours = int(timestamp // 3600 % 24)
    minutes = int(timestamp // 60 % 60)
    seconds = int(timestamp % 60)
    return f"{days}d:{hours}h:{minutes}m:{seconds}s"


def fixed_aspect_ratio(ax, ratio):
    '''
    Set a fixed aspect ratio on matplotlib loglog plots 
    regardless of axis units
    '''
    xvals,yvals = ax.axes.get_xlim(),ax.axes.get_ylim()

    xrange = xvals[1]-xvals[0]
    yrange = yvals[1]-yvals[0]
    ax.set_aspect(ratio*(xrange/yrange), adjustable='box')


def configure_logging_paths(logging_directory, log_suffixes = [""], index_new = True):
    """
    Given a base log path, create a logging folder for this job (logging_directory with an index).
    Inside that create a txt file for string logging.
    """
    if index_new:
        i = 0
        while i < 1000:
            try:
                curr_path = f"{logging_directory}_{i}"
                os.mkdir(curr_path)
                print_paths = []
                for ls in log_suffixes:
                    print_path = os.path.join(curr_path, f"epoch_log_{ls}.csv")
                    print_paths.append(print_path)
                warning_file = os.path.join(curr_path, "warnings.txt")
                return print_paths, curr_path, warning_file
            except FileExistsError:
                i += 1
        raise Exception("Too many subfolders being created!")
    else:
        os.mkdir(logging_directory)
        print_paths = []
        for ls in log_suffixes:
            print_path = os.path.join(logging_directory, f"epoch_log_{ls}.csv")
            print_paths.append(print_path)
        warning_file = os.path.join(logging_directory, "warnings.txt")
        return print_path, logging_directory, warning_file


def configure_state_path(logging_directory, filename, filetype=""):
    i = 1
    filetype = filetype.lstrip(".")
    while i <= 1000:
        curr_path = os.path.join(logging_directory, f"{filename}{i}.{filetype}")
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


