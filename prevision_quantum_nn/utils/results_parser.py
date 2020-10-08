""" results parser module """
import pandas as pd

METRICS = ["auc:", "accuracy:", "reward:"]


def parse_results(results_file):
    """Parse a result file in order to extract parameters.

    Args:
        results_file (string):path to a log file of an
            application

    Returns:
        results: Pandas dataframe
            dataframe containing parameters to be plotted.
    """
    results = pd.DataFrame()

    # open results file
    in_file = open(results_file, "r")

    # parse results file
    for line in in_file:

        # supervised learning
        iteration = None
        train_loss = None
        val_loss = None
        metric = None
        supervised_learning_results_found = False

        # reinforcement learning
        episode = None
        reward = None
        reinforcement_learning_results_found = False

        # get splitted logging line
        log = line.split()

        # recover iteration number
        if "iter:" in log:
            iteration = int(log[log.index("iter:") + 1])

        # recover episode number
        if "episode:" in log:
            episode = int(log[log.index("episode:") + 1])
            reinforcement_learning_results_found = True

        # recover train loss
        if "train_loss:" in log:
            train_loss = float(log[log.index("train_loss:") + 1])
            supervised_learning_results_found = True

        # recover val loss
        if "val_loss:" in log:
            val_loss = float(log[log.index("val_loss:") + 1])

        # recover metric
        for metric_name in METRICS:
            if metric_name in log:
                metric = float(log[log.index(metric_name) + 1])

        # recover reward
        if "reward:" in log:
            reward = float(log[log.index("reward:") + 1])

        # if results in log, append to results
        if supervised_learning_results_found:
            current_results = {
                "iteration": iteration,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "metric": metric,
            }
            results = results.append(current_results, ignore_index=True)

        if reinforcement_learning_results_found:
            current_results = {
                "episode": episode,
                "reward": reward,
            }
            results = results.append(current_results, ignore_index=True)

    # close results file
    in_file.close()
    return results
