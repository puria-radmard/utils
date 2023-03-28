from glob import glob
from functools import reduce
import os, json
import pandas as pd


class ConfigPlaceholder:
    def __init__(self, d) -> None:
        self.__dict__.update(d)


def args_dict_from_path(rp, as_config=False):
    json_path = os.path.join(rp, "args.json")
    with open(json_path, "r") as f:
        args = json.load(f)
    return ConfigPlaceholder(args) if as_config else args


def csv_log_from_path(rp, index_column="Episode"):
    df_path = os.path.join(rp, "epoch_log.csv")
    df = pd.read_csv(df_path)
    return df.set_index(index_column)


def file_paths_from_configs(base_path, **matching_kwargs):
    run_paths = glob(os.path.join(base_path, "*run_*"))
    ret_paths = []
    for rp in run_paths:
        args = args_dict_from_path(rp)
        match = True
        for k, v in matching_kwargs.items():
            if args[k] != v:
                match = False
        if match:
            ret_paths.append(rp)
    return ret_paths


def seperate_paths_by_arg(paths, seperating_arg):
    ret = {}
    for path in paths:
        sep_arg = args_dict_from_path(path)[seperating_arg]
        if isinstance(sep_arg, list):
            sep_arg = tuple(sep_arg)
        ret[sep_arg] = ret.get(sep_arg, []) + [path]
    return ret


def mean_and_std_csvs_by_dfs(dfs, happy_columns):
    assert all([df.columns.tolist() == dfs[0].columns.tolist() for df in dfs])
    df_merged = reduce(lambda left, right: pd.concat([left, right], axis=1), dfs)
    df_reduced = df_merged[happy_columns]
    df_grouped = df_reduced.groupby(by=df_reduced.columns, axis=1)
    mean_df = df_grouped.mean()
    std_df = df_grouped.std()
    lower_df = mean_df - 2 * std_df
    upper_df = mean_df + 2 * std_df
    # lower_df = df_grouped.min()
    # upper_df = df_grouped.max()
    return {"mean": mean_df, "std": std_df, "lower": lower_df, "upper": upper_df}


def mean_and_std_csvs_by_paths(paths, happy_columns, joining_column="Episode"):
    dfs = [csv_log_from_path(path, joining_column) for path in paths]
    return mean_and_std_csvs_by_dfs(dfs, happy_columns)


def unmatching_lines(all_epoch_lists, all_value_lists):
    max_epoch = 0
    dfs = []
    for epoch_list, value_list in zip(all_epoch_lists, all_value_lists):
        max_epoch = max(max_epoch, max(epoch_list))
        dfs.append(pd.DataFrame(columns=["value"], data=value_list, index=epoch_list))
    dfs = [df.reindex(range(max_epoch)).interpolate(method="index") for df in dfs]
    return mean_and_std_csvs_by_dfs(dfs, happy_columns=["value"])


def get_all_metric_lines_from_path_list(metric, log_paths):

    all_metric_epochs = []
    all_metric_values_lists = []

    for log_path in log_paths:
        if metric in ["small_worldness", "modularity"]:
            metric_path = os.path.join(log_path, f"{metric}.json")
        elif metric in [
            "average_abs_weight",
            "average_abs_diag_weight",
            "weights_distances_correlation",
        ]:
            metric_path = os.path.join(log_path, f"weight_fundamentals.json")

        with open(metric_path, "r") as f:
            metric_results = json.load(f)

        metric_epochs = [int(k) for k in metric_results.keys()]

        if metric == "small_worldness":
            metric_values = []
            for val_list in metric_results.values():
                metric_values.append(
                    (val_list[0]["C"] / val_list[0]["Cr"])
                    / (val_list[0]["L"] / val_list[0]["Lr"])
                )
            # metric_values = [val_list[0]['small_worldness'] for val_list in metric_results.values()]

        elif metric == "modularity":
            metric_values = [val_list[0]["q"] for val_list in metric_results.values()]

        elif metric in [
            "average_abs_weight",
            "average_abs_diag_weight",
            "weights_distances_correlation",
        ]:
            metric_values = [
                val_list[0][metric] for val_list in metric_results.values()
            ]

        all_metric_epochs.append(metric_epochs)
        all_metric_values_lists.append(metric_values)

    return all_metric_epochs, all_metric_values_lists


def get_all_metric_plots_from_path_list(metric, log_paths, return_individual=False):
    all_metric_epochs, all_metric_values_lists = get_all_metric_lines_from_path_list(
        metric, log_paths
    )
    aggregation_dfs = unmatching_lines(all_metric_epochs, all_metric_values_lists)
    if return_individual:
        return aggregation_dfs, all_metric_epochs, all_metric_values_lists
    else:
        return aggregation_dfs
