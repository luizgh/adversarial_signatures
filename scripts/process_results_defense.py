from collections import defaultdict
from typing import DefaultDict

import pandas as pd
import pickle
from pathlib import Path
import numpy as np
import argparse


def process_dataframe(list_of_results):
    results = pd.DataFrame(list_of_results,
                                   columns=['user', 'model', 'image_type',
                                            'attack_type', 'image_idx',
                                            'attack_img', 'rmse', 'score',
                                            'success'])
    results['success'].fillna(0, inplace=True)
    results['success'] = results['success'].astype(bool)

    results['model'] = results['model'].str.replace('model\_cnn\_linear',
                                                    'SigNet & Linear')
    results['model'] = results['model'].str.replace('model\_cnn\_rbf',
                                                    'SigNet & RBF')
    results['model'] = results['model'].str.replace('model\_lbp\_linear',
                                                    'CLBP & Linear')
    results['model'] = results['model'].str.replace('model\_lbp\_rbf',
                                                    'CLBP & RBF')


    return results


def load_results(fname):
    with open(fname, 'rb') as f:
        gen, forg = pickle.load(f)

    results_genuine = process_dataframe(gen)
    results_forgery = process_dataframe(forg)

    return results_genuine, results_forgery


def format_pct(x):
    if np.isnan(x):
        return '-'
    return  '%.2f' % (x * 100)


def format_normal(x):
    if np.isnan(x):
        return '-'
    else:
        return '%.2f' % x


parser = argparse.ArgumentParser()
parser.add_argument('results_folder')

args = parser.parse_args()

knowledge_scenarios = ['pk', 'lk1', 'lk2']
defense_models = ['', 'ensadv', 'madry']
datasets = ['mcyt', 'cedar', 'brazilian', 'gpds']

base_path = Path(args.results_folder)
# base_path = Path('~/runs/adv/').expanduser()

all_results_genuine = []
all_results_forgery = []
all_results_genuine_bydataset = defaultdict(list)
all_results_forgery_bydataset = defaultdict(list)

for k in knowledge_scenarios:
    for model in defense_models:
        for d in datasets:
            if model == '':
                model_ = model
                modelname = 'baseline'
            else:
                model_ = '{}_'.format(model)
                modelname = model
            filename = base_path / '{}_cnn_half_{}{}.pickle'.format(d, model_, k)

            results_genuine, results_forgery = load_results(filename)
            results_genuine['knowledge'] = k
            results_genuine['defense'] = modelname
            results_genuine['dataset'] = d

            results_forgery['knowledge'] = k
            results_forgery['defense'] = modelname
            results_forgery['dataset'] = d

            all_results_genuine.append(results_genuine)
            all_results_forgery.append(results_forgery)

            all_results_genuine_bydataset[d].append(results_genuine)
            all_results_forgery_bydataset[d].append(results_forgery)


def print_results(results_genuine, results_forgery):
    df = results_genuine[(results_genuine['attack_type'] == 'fgm') |
                         (results_genuine['attack_type'] == 'carlini')].drop(columns=['attack_img'])

    # Fixing the order:
    df.loc[df['knowledge'] == 'pk', 'knowledge'] = '_pk'
    df.loc[df['attack_type'] == 'fgm', 'attack_type'] = '_fgm'

    pd.set_option('display.float_format', format_pct)
    g = df.groupby(['defense', 'model', 'knowledge', 'attack_type'])
    subset = g[['success']].mean()
    p = subset.reset_index().pivot_table(index=['defense', 'model'],
                                     values='success',
                                     columns=['attack_type', 'knowledge'])
    print('Genuine, success')
    print(p.to_latex())


    pd.set_option('display.float_format', format_normal)
    only_success = df[df['success'] == True]
    g = only_success.groupby(['defense', 'model', 'knowledge', 'attack_type'])
    subset = g[['rmse']].mean()
    p = subset.reset_index().pivot_table(index=['defense', 'model'],
                                     values='rmse',
                                     columns=['attack_type', 'knowledge'])

    pd.set_option('display.float_format', format_normal)
    print('Genuine, RMSE')
    print(p.to_latex())

    df = results_forgery[(results_forgery['attack_type'] == 'fgm') |
                         (results_forgery['attack_type'] == 'carlini')].drop(columns=['attack_img'])

    # Fixing the order:
    df.loc[df['knowledge'] == 'pk', 'knowledge'] = '_pk'
    df.loc[df['attack_type'] == 'fgm', 'attack_type'] = '_fgm'

    pd.set_option('display.float_format', format_pct)
    g = df.groupby(['defense', 'model', 'knowledge', 'attack_type', 'image_type'])
    subset = g[['success']].mean()
    p = subset.reset_index().pivot_table(index=['defense', 'model', 'image_type'],
                                     values='success',
                                     columns=['attack_type', 'knowledge'])

    print('Forgery, success')
    print(p.to_latex())



    only_success = df[df['success'] == True]
    g = only_success.groupby(['defense', 'model', 'knowledge', 'attack_type', 'image_type'])
    subset = g[['rmse']].mean()
    p = subset.reset_index().pivot_table(index=['defense', 'model', 'image_type'],
                                     values='rmse',
                                     columns=['attack_type', 'knowledge'])

    pd.set_option('display.float_format', format_normal)
    print('Forgery, RMSE')
    print(p.to_latex())


results_genuine = pd.concat(all_results_genuine)
results_forgery = pd.concat(all_results_forgery)

print('Number of rows: {}'.format(len(results_genuine)))

print('Consolidated results')
print_results(results_genuine, results_forgery)

for d in datasets:
    results_genuine = pd.concat(all_results_genuine_bydataset[d])
    results_forgery = pd.concat(all_results_forgery_bydataset[d])
    print(d)
    print_results(results_genuine, results_forgery)