import os, argparse, glob, json, pickle
from tqdm import tqdm
from util.measures import small_worldness
from mrc_sernn_project.util.viz import CPU_Unpickler


parser = argparse.ArgumentParser(
    description="Go over each pkl file in log_directory and"
)

parser.add_argument(
    "--log_directory",
    "-ld",
    type=str,
    help="Directory of logs to pick up and analyse. Formal will be like logs/a_first_attempt_multitask/a_sweeping_lr_for_rnn-0",
)
parser.add_argument(
    "--binarisation_threshold_props",
    default=[0.1],
    nargs="+",
    type=float,
    help="List of proportion thresholds to binaryise seRNN hidden weight by",
)
parser.add_argument(
    "--num_random_graphs",
    default=1000,
    nargs="+",
    type=float,
    help="Number of random batched graphs to generate for each metric",
)


if __name__ == "__main__":

    args = parser.parse_args()

    base_log_directory = args.log_directory
    binarisation_threshold_props = args.binarisation_threshold_props
    num_random_graphs = args.num_random_graphs

    for log_directory in tqdm(glob.glob(os.path.join(base_log_directory, "*_run_*"))):

        # Schema: {epoch_num: [{c: ..., q: ...}] ...}
        result_dict = {}

        log_pickles = glob.glob(os.path.join(log_directory, "*.pkl"))

        for log_pickle_path in log_pickles:

            if (
                int(log_pickle_path.split("epoch_")[-1].split(".pkl")[0]) - 25
            ) % 100 != 0:
                continue

            with open(log_pickle_path, "rb") as f:
                try:
                    epoch_log_dict = pickle.load(f)
                except:
                    epoch_log_dict = CPU_Unpickler(f).load()

            epoch_num = int(epoch_log_dict["epoch_number"])
            rnn_weights = epoch_log_dict["seRNN_param"]
            epoch_measures = []

            for thres_prop in binarisation_threshold_props:

                small_worldness_dict = small_worldness(
                    param=rnn_weights,
                    thres_prop=thres_prop,
                    num_random_graphs=num_random_graphs,
                )

                epoch_measures.append(
                    dict(binarisation_threshold=thres_prop, **small_worldness_dict)
                )

            result_dict[epoch_num] = epoch_measures

        with open(os.path.join(log_directory, "small_worldness.json"), "w") as f:
            json.dump(result_dict, f)
