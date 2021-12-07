import pandas as pd
import matplotlib as mpl
import matplotlib.pylab as plt


# extract data from output files
def data_collector(file):
    iter_check = "val_loss"
    best_iter_check = "Best iter:"
    time_check = "elapsed time (s):"
    best_iter = 0
    elapsed_time = 0.0
    data = pd.DataFrame(columns=['loss',
                                 'val_loss',
                                 'nb_step',
                                 'auc'])
    with open(file) as in_file:
        for line in in_file:
            if iter_check in line:
                get_data = line.split(" ")
                current_iter = int(get_data[6])
                loss = float(get_data[8])
                val_loss = float(get_data[10])
                auc = float(get_data[12])
                data = data.append(pd.Series([loss,
                                              val_loss,
                                              current_iter,
                                              auc],
                                             index=data.columns),
                                   ignore_index=True)
            if best_iter_check in line:
                get_iter = line.split(" ")
                best_iter = int(get_iter[14])
            if time_check in line:
                get_time = line.split(" ")
                elapsed_time = float(get_time[8])

    return data, best_iter, elapsed_time


# plotting functions
def plot_data(data):
    # global params
    mpl.rcParams['font.family'] = 'DejaVu Sans'
    plt.rcParams['font.size'] = 18
    plt.rcParams['axes.linewidth'] = 2

    # variables
    fig, axs = plt.subplots(2)
    axs[0].set_xlabel('step number')
    axs[0].set_ylabel('loss')
    axs[0].plot(data['nb_step'], data['loss'], label='loss')
    axs[0].plot(data['nb_step'], data['val_loss'], label='val loss')
    axs[1].set_xlabel('step number')
    axs[1].set_ylabel('auc')
    axs[1].plot(data['nb_step'], data['auc'], label='auc')
    axs[0].legend()
    axs[1].legend()
    plt.show()


if __name__ == "__main__":

    # file should be the name of an existing .listing file
    file = "test.listing"

    data, best_iter, elapsed_time = data_collector(file)

    if best_iter > 0:
        print("Best iteration:", best_iter)
    else:
        print("Best iteration not found")

    if elapsed_time > 0:
        print("Elapsed time (s):", elapsed_time)
    else:
        print("Elapsed time not found")

    plot_data(data)
