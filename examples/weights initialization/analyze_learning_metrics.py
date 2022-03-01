import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import pandas as pd
import json
from get_learning_metrics import save_learning_metrics

if __name__ == "__main__":
    print_missing_data = False
    print_recap = False
    group_by_init = False

    df = pd.read_csv('metrics.csv', on_bad_lines='skip')

    for info in sys.argv[1:]:
        if info[:7] == "spirals" or info == "breast_cancer":
            dataset = info
            df = df[df['dataset'] == dataset]

        if info == "missing":
            print_missing_data = True

        if info.isnumeric():
            num_q = float(info)
            df = df[df['num_q'] == num_q]

        if info == "recap":
            print_recap = True

        if info in ['group_by', 'group_by_init']:
            group_by_init = True


    if not print_recap:
        recap = df.groupby('init').mean()
        recap['count'] = df.groupby('init').count()['auc_iter']
        recap = recap.drop(['mean', 'best'])

        print(recap.columns)
        print(recap[['achieve_cvg', 'first_to_cvg',
                     'auc_iter', 'count',
                     # 'auc_first', 'train_loss_first', 'val_loss_first'
                     ]])
    else:
        recap = df
        #print(recap.columns)
        recap = recap[recap['init'] != 'mean']
        recap = recap[recap['init'] != 'best']

        if not group_by_init:
            print(recap[['dataset', 'init', 'circuit', 'num_q', 'num_layers']])
        else:
            cols = ['dataset', 'circuit', 'num_q', 'num_layers']
            recap_gb = recap.groupby(cols)['init'].unique()
            recap_gb = pd.DataFrame(recap_gb)
            # print("cols", recap_gb.columns)
            recap_gb['count'] = recap_gb['init'].apply(len)

            def get_missing(liste):
                return [x for x in ['id_blk',
                                    'random_init_single',
                                    'random_init_double',
                                    'zero_init_single',
                                    'zero_init_double']
                        if x not in liste]
            recap_gb['missing'] = recap_gb['init'].apply(get_missing)

            recap_gb = recap_gb[['count', 'missing']]
            print(recap_gb)

    if print_missing_data:
        seen_filenames = list(df['filename'] + ".listing")
        print("\n \n Missing data:")
        a = 1
        for filename in os.listdir('results'):
            if filename[-len(".listing"):] == ".listing" and \
                    "results/" + filename not in seen_filenames:
                print(filename)
                a += 1
        print("Number of missing data:", a)
