import numpy as np
import matplotlib.pyplot as plt
import sys


def get_auc_list(file_name):
    file = open(file_name, "r")
    a = []
    for line in file.read().split('\n'):
        if line.find("auc") >= 0:
            auc = line[line.find("auc") + 5:]
            a.append(auc)
    y_vec = []
    for y in a:
        try:
            y_vec.append(float(y))
        except:
            pass
    y_vec = np.array(y_vec)
    return y_vec


if __name__ == "__main__":

    prefix_list = [f"id_blk",
                   f"zero_init_single",
                   f"zero_init_double",
                   f"random_init_single",
                   f"random_init_double",
                   ]

    num_q = 2

    str_num_q = f"_num_q_{num_q}" if num_q != 5 else ""

    if len(sys.argv) > 1:
        nb_turns = float(sys.argv[1])

        for prefix in prefix_list:
            auc_list = get_auc_list(f"results/{prefix}"
                                    f"_nb_turns_{nb_turns}"
                                    f"{str_num_q}.listing")
            plt.plot(auc_list, label=prefix)
            plt.ylim(bottom=0, top=1)
            plt.title(f"nb_turns = {nb_turns}")
            plt.xlabel("nb of iterations")
            plt.ylabel("AUC")
        plt.legend(loc='lower right')  # , bbox_to_anchor=(1, 0.5))
        plt.tight_layout()
        plt.show()
    else:
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

        for i, nb_turns in enumerate(nb_turns_list):
            for prefix in prefix_list:
                auc_list = get_auc_list(f"results/{prefix}"
                                        f"_nb_turns_{nb_turns}"
                                        f"{str_num_q}.listing")
                axes[i].plot(auc_list, label=prefix)

            axes[i].set_ylim(bottom=0, top=1)
            axes[i].set_title(f"nb_turns = {nb_turns}")
            axes[i].set_xlabel("nb of iterations")
            axes[i].set_ylabel("AUC")

        axes[0].legend(loc='lower right')  # , bbox_to_anchor=(1, 0.5))

        plt.tight_layout()
        plt.show()
