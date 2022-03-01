import numpy as np
import matplotlib.pyplot as plt
import sys
import os


def get_auc_list(file_name):
    return get_info_list(file_name, "auc:")


def get_train_loss_list(file_name):
    return get_info_list(file_name, "train_loss:")


def get_val_loss_list(file_name):
    return get_info_list(file_name, "val_loss:")


def get_grad_list(file_name):
    return get_info_list(file_name, "grad:")


def get_info_list(file_name, info):
    file = open(file_name, "r")
    a = []
    for line in file.read().split('\n'):
        elems = line.split()
        if info in elems:
            ind = elems.index(info)
            val = elems[ind + 1]
            a.append(val)
    y_vec = []
    for y in a:
        try:
            y_vec.append(float(y))
        except:
            pass
    y_vec = np.array(y_vec)
    return y_vec


if __name__ == "__main__":

    prefix_list = ["id_blk",
                   "random_init_single",
                   "random_init_double",
                   #               ]
                   # zeros = [
                   "zero_init_single",
                   "zero_init_double",
                   ]
    linestyles = ["solid",
                  "dashdot",
                  "dashdot",
                  "dotted",
                  "dotted",
                  ]

    num_q = 5

    if len(sys.argv) > 1:
        try:
            nb_turns_list = [float(sys.argv[1])]
        except:
            nb_turns_list = eval(sys.argv[1])
            nb_turns_list = [float(x) for x in nb_turns_list]
    else:
        nb_turns_list = [0.7, 0.8, 0.9, 1.0, 1.2, 1.3]
        nb_turns_list = [0.9, 1.0, 1.1]

    nb_circuit = sys.argv[2] if len(sys.argv) > 2 else "6"

    if nb_circuit.isnumeric():
        circuit_prefix = "circuit" + nb_circuit
    else:
        circuit_prefix = nb_circuit  # "custom", "circuit5"

    type_metric = sys.argv[3] if len(sys.argv) > 3 else "auc"

    num_q = sys.argv[4] if len(sys.argv) > 4 else num_q
    str_num_q = f"_num_q_{num_q}"  # if num_q != 5 else ""

    dataset_name = sys.argv[5] if len(sys.argv) > 5 else ""
    str_dataset = f"_dataset_{dataset_name}" if dataset_name else ""

    ncols = len(nb_turns_list)
    nrows = 1
    figsize = (16, 9)

    if type_metric == "all":
        metrics = ["auc", "train_loss", "val_loss", "grad"]
    else:
        metrics = [type_metric]
    if type_metric == "all":
        nrows = len(metrics)
        figsize = (5 * ncols, 10)

    dpi = 300
    fig, axes = plt.subplots(ncols=ncols, nrows=nrows,
                             figsize=figsize)  # figsize=figsize, dpi=dpi)
    plt.rcParams.update({'font.size': 8})

    try:
        if len(axes.shape) == 1:
            axes.resize((len(axes), 1))
    except:
        None

    try:
        axes[0, 0]
    except:
        axes = {(0, 0): axes, 0: axes}

    for j, nb_turns in enumerate(nb_turns_list):

        """try:
            axes = axes.ravel()
        except:
            axes = [axes]"""

        str_nb_turns = f"_nb_turns_{nb_turns}" if nb_turns > 0 else ""

        for ind_prefix, prefix in enumerate(prefix_list):

            listing_filename = f"results/{circuit_prefix}" \
                               f"{str_dataset}" \
                               f"_{prefix}" \
                               f"{str_nb_turns}" \
                               f"{str_num_q}.listing"
            if not os.path.exists(listing_filename):
                print(listing_filename, "doesn't exist")
                continue

            for i, type_metric in enumerate(metrics):

                if type_metric == "auc":
                    val_list = get_auc_list(listing_filename)
                elif type_metric == "grad":
                    val_list = get_grad_list(listing_filename)
                elif type_metric == "val_loss":
                    val_list = get_val_loss_list(listing_filename)
                else:  # loss
                    val_list = get_train_loss_list(listing_filename)

                axes[i, j].plot(val_list, label=prefix,
                                linestyle=linestyles[ind_prefix])
                if nb_turns > 0:
                    axes[i, j].set_title(
                        f"nb_turns = {nb_turns}, {circuit_prefix}")
                else:
                    axes[i, j].set_title(f"{circuit_prefix}")
                axes[i, j].set_xlabel("nb of iterations")
                axes[i, j].set_ylabel(type_metric)
                if type_metric == "auc":
                    axes[i, j].set_ylim(bottom=0, top=1)
                    axes[i, j].legend(loc='lower right')
                else:  # loss
                    axes[i, j].legend(loc='upper right')
                if type_metric == "grad":
                    axes[i, j].set_yscale('log')

    plt.tight_layout()
    plt.show()

"""
ARCHIVE STRING


nb_turns_list = [0.7, 0.8, 0.9, 1.0, 1.2, 1.3]

ncols = 3
nrows = 2
figsize = (16, 9)
dpi = 300
fig, axes = plt.subplots(ncols=ncols, nrows=nrows,
                         figsize=figsize, dpi=dpi)
fig.suptitle(f"Comparison for n = {num_q} qubits")
plt.rcParams.update({'font.size': 8})
axes = axes.ravel()

for i, nb_turns in enumerate(nb_turns_list[:ncols*nrows]):
    for prefix in prefix_list:
        prefix = "3janv_" + prefix
        auc_list = get_auc_list(f"results/{prefix}"
                                f"_nb_turns_{nb_turns}"
                                f"{str_num_q}.listing")
        axes[i].plot(auc_list, label=prefix)

    axes[i].set_ylim(bottom=0, top=1)
    axes[i].set_title(f"nb_turns = {nb_turns}")
    axes[i].set_xlabel("nb of iterations")
    axes[i].set_ylabel("AUC")

    axes[i].legend(loc='lower right')  # , bbox_to_anchor=(1, 0.5))

plt.tight_layout()
plt.show()
"""
