""" results plotter module """
import matplotlib.pylab as plt


def plot_losses(results, prefix="qnn"):
    """Plot the losses of an application.

    Args:
        results (Pandas dataframe): dataframe containing the losses
        prefix (string): part of the name given to the generated
            plot
    """
    if "train_loss" not in results.columns:
        print(Warning("No supervised learning results found in data"))
        return

    # create figure
    fig, axes = plt.subplots(1, 1)

    # plot train loss
    axes.plot(results["iteration"],
              results["train_loss"],
              label="train loss",
              # purple
              color=(120/255, 131/255, 212/255))

    # plot validation loss
    val_loss_results = results[results["val_loss"].notnull()]
    axes.plot(val_loss_results["iteration"],
              val_loss_results["val_loss"],
              label="val loss",
              # green
              color=(134/255, 212/255, 166/255))

    # set meta data
    axes.set_xlabel("iteration number")
    axes.set_ylabel("train_loss")
    axes.legend()

    # save figure
    fig.savefig(f"{prefix}_losses.png")


def plot_metric(results, prefix="qnn"):
    """Plot the relevant metric of an application.

    Args:
        results (Pandas dataframe): dataframe containing the metrics
        prefix (string): part of the name given to the generated
            plot
    """
    if "metric" not in results.columns:
        print(Warning("No supervised learning metric found in data"))
        return

    fig, axes = plt.subplots(1, 1)

    # plot metric
    metric_results = results[results["metric"].notnull()]
    axes.plot(metric_results["iteration"],
              metric_results["metric"],
              label="metric",
              color=(134/255, 212/255, 166/255))

    # plot metadata
    axes.set_xlabel("iteration number")
    axes.set_ylabel("metric")
    axes.set_ylim(0., 1.)
    axes.legend()

    # save figure
    fig.savefig(f"{prefix}_metric.png")


def plot_reward(results, prefix="qnn"):
    """Plot the reward of a RL application.

    Args:
        results (Pandas dataframe): dataframe containing the rewards
        prefix (string): part of the name given to the generated
            plot
    """

    if "episode" not in results.columns:
        print(Warning("No reinforcement learning results found in data"))
        return

    # create figure
    fig, axes = plt.subplots(1, 1)

    # plot metric
    axes.plot(results["episode"],
              results["reward"],
              label="reward",
              color=(134/255, 212/255, 166/255))

    # plot metadata
    axes.set_xlabel("episode number")
    axes.set_ylabel("reward")
    axes.legend()

    # save figure
    fig.savefig(f"{prefix}_reward.png")
