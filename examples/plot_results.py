import sys
import prevision_quantum_nn as qnn

if __name__ == "__main__":

    if len(sys.argv) < 2:
        raise ValueError("You have not provided any results file name.")

    results_file = sys.argv[1]

    # read results
    results = qnn.parse_results(results_file)

    # plot losses
    qnn.plot_losses(results, prefix=results_file)

    # plot metric
    qnn.plot_metric(results, prefix=results_file)

    # plot metric
    qnn.plot_reward(results, prefix=results_file)
