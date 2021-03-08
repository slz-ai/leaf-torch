import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import visualization_utils

SHOW_WEIGHTED = True  # show weighted accuracy instead of unweighted accuracy
PLOT_CLIENTS = False
stat_file = 'metrics_stat.csv'  # change to None if desired
sys_file = 'metrics_sys.csv'  # change to None if desired


def plot_acc_vs_round(stat_metrics, sys_metrics):
    """Plots accuracy vs. round number."""
    if stat_metrics is not None:
        visualization_utils.plot_accuracy_vs_round_number(
            stat_metrics, True, plot_stds=False)
    if PLOT_CLIENTS and stat_metrics is not None:
        visualization_utils.plot_accuracy_vs_round_number_per_client(
            stat_metrics, sys_metrics, max_num_clients=20)


def plot_bytes_vs_round(stat_metrics, sys_metrics):
    """Plots the cumulative sum of the bytes written and read by the server in
    the past rolling_window rounds versus the round number.
    """
    if stat_metrics is not None:
        visualization_utils.plot_bytes_written_and_read(
            sys_metrics, rolling_window=100)


def plot_comp_vs_round(stat_metrics, sys_metrics):
    visualization_utils.plot_client_computations_vs_round_number(
        sys_metrics, aggregate_window=1, max_num_clients=20, range_rounds=(1, 4))


def calc_longest_flops(stat_metrics, sys_metrics):
    print('Longest FLOPs path: %s' %
          visualization_utils.get_longest_flops_path(sys_metrics))


if __name__ == "__main__":
    metrics = visualization_utils.load_data(stat_file, sys_file)
    plot_acc_vs_round(*metrics)
    # plot_bytes_vs_round(*metrics)
    # plot_comp_vs_round(*metrics)
    # calc_longest_flops(*metrics)
