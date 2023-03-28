import os, argparse, glob, json, pickle, torch
import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm
from util.measures import weights_fundamentals
from scripts.a_first_attempt_multitask.master_training_script import configure_seRNN_args
from util.notebooks import args_dict_from_path
from mrc_sernn_project.util.viz import CPU_Unpickler

CAD_OPT = os.environ.get('CUDA_AVAILABLE_DEVICES')
device = f'cuda:{CAD_OPT}' if torch.cuda.is_available() and CAD_OPT else 'cpu'


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
    "--grid_model_directory",
    type=str,
    required=False,
    default="/homes/pr450/repos/seRNNTorch/logs/a_first_attempt_multitask/a2_sweeping_lr_and_gamma_for_sernn/PerceptualDecisionMaking_run_0"
)


if __name__ == "__main__":

    args = parser.parse_args()

    base_log_directory = args.log_directory

    for log_directory in tqdm(glob.glob(os.path.join(base_log_directory, "*_run_*"))):

        # Schema: {epoch_num: [{c: ..., q: ...}] ...}
        result_dict = {}

        log_pickles = glob.glob(os.path.join(log_directory, "*.pkl"))
        
        args_source_dir = log_directory if args.grid_model_directory is None else args.grid_model_directory
        log_args = args_dict_from_path(args_source_dir, as_config=True) 
        reg, _ = configure_seRNN_args(log_args)
        grid = reg.grid

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
                weights_information_dict = weights_fundamentals(param=rnn_weights, grid=grid)
            except Exception as e:
                print('Could not calculate weights fundamental metrics for', log_directory, 'due to', e)

            epoch_measures.append(weights_information_dict)

            result_dict[epoch_num] = epoch_measures


        with open(os.path.join(log_directory, "weight_fundamentals.json"), "w") as f:
            json.dump(result_dict, f)
