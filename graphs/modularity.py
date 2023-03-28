import os, argparse, glob, json, pickle, torch
import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm
from util.measures import modularity
from mrc_sernn_project.util.viz import CPU_Unpickler

CAD_OPT = os.environ.get("CUDA_AVAILABLE_DEVICES")
device = f"cuda:{CAD_OPT}" if torch.cuda.is_available() and CAD_OPT else "cpu"


parser = argparse.ArgumentParser(
    description="Go over each pkl file in log_directory and"
)

parser.add_argument(
    "--log_directory",
    "-ld",
    type=str,
    help="Directory of logs to pick up and analyse. Formal will be like logs/a_first_attempt_multitask/a_sweeping_lr_for_rnn-0",
)


if __name__ == "__main__":

    args = parser.parse_args()

    base_log_directory = args.log_directory

    for log_directory in tqdm(glob.glob(os.path.join(base_log_directory, "*_run_*"))):

        # Schema: {epoch_num: [{c: ..., q: ...}] ...}
        result_dict = {}

        log_pickles = glob.glob(os.path.join(log_directory, "*.pkl"))

        for log_pickle_path in log_pickles:

            with open(log_pickle_path, "rb") as f:
                try:
                    epoch_log_dict = pickle.load(f)
                except:
                    epoch_log_dict = CPU_Unpickler(f).load()

            epoch_num = int(epoch_log_dict["epoch_number"])
            rnn_weights = epoch_log_dict["seRNN_param"]
            epoch_measures = []

            try:
                modularity_dict = modularity(param=rnn_weights, for_json=True)
            except Exception as e:
                print("Could not calculate modularity for", log_directory, "due to", e)
                continue

            epoch_measures.append(modularity_dict)

            result_dict[epoch_num] = epoch_measures

        with open(os.path.join(log_directory, "modularity.json"), "w") as f:
            json.dump(result_dict, f)
