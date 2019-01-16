from collections import defaultdict

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
                                            'success', 'attack_img_otsu',
                                            'score_otsu', 'success_otsu'])
    results['success'].fillna(0, inplace=True)
    results['success'] = results['success'].astype(bool)

    results['success_otsu'].fillna(0, inplace=True)
    results['success_otsu'] = results['success_otsu'].astype(bool)

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

defense_models = ['', 'ensadv', 'madry']
datasets = ['mcyt', 'cedar', 'brazilian', 'gpds']

base_path = Path(args.results_folder)
# base_path = Path('~/runs/adv/otsu').expanduser()

all_results_genuine = []
all_results_forgery = []
all_results_genuine_bydataset = defaultdict(list)
all_results_forgery_bydataset = defaultdict(list)

for model in defense_models:
    for d in datasets:
        if model == '':
            model_ = model
            modelname = 'baseline'
        else:
            model_ = '{}_'.format(model)
            modelname = model
        filename = base_path / '{}_cnn_half_{}pk_otsu.pickle'.format(d, model_)

        results_genuine, results_forgery = load_results(filename)
        results_genuine['defense'] = modelname
        results_genuine['dataset'] = d

        results_forgery['defense'] = modelname
        results_forgery['dataset'] = d

        all_results_genuine.append(results_genuine)
        all_results_forgery.append(results_forgery)
        all_results_genuine_bydataset[d].append(results_genuine)
        all_results_forgery_bydataset[d].append(results_forgery)


for d in datasets:
    filename = base_path / '{}_lbp_attacks_pk_otsu.pickle'.format(d)
    results_genuine, results_forgery = load_results(filename)
    results_genuine['defense'] = 'CLBP'
    results_genuine['dataset'] = d

    results_forgery['defense'] = 'CLBP'
    results_forgery['dataset'] = d

    all_results_genuine.append(results_genuine)
    all_results_forgery.append(results_forgery)
    all_results_genuine_bydataset[d].append(results_genuine)
    all_results_forgery_bydataset[d].append(results_forgery)


def print_results(results_genuine, results_forgery):
    print('Attacks on genuine signatures')
    df = results_genuine.drop(columns=['attack_img'])

    # Fixing the order:
    pd.set_option('display.float_format', format_pct)
    g = df.groupby(['defense', 'model', 'attack_type'])
    subset = g[['success', 'success_otsu']].mean()

    p = subset.stack().unstack('attack_type').unstack()
    p = p.reindex(columns = ['fgm', 'carlini', 'anneal', 'decision'],
                  level='attack_type')

    print(p.to_latex())


    print('Attacks on forgeries')
    df = results_forgery.drop(columns=['attack_img'])

    pd.set_option('display.float_format', format_pct)
    g = df.groupby(['defense', 'model', 'image_type', 'attack_type'])
    subset = g[['success', 'success_otsu']].mean()

    p = subset.stack().unstack('attack_type').unstack()
    p = p.reindex(columns = ['fgm', 'carlini', 'anneal', 'decision'],
                  level='attack_type')

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