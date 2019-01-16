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


order = {'fgm': 0, 'carlini': 1, 'anneal': 2, 'decision': 3}
def print_latex(df):
    p = pd.DataFrame(df.to_records()).pivot(index='model', columns='attack_type')
    p.columns = p.columns.get_level_values(1)
    p = p[sorted(p.columns, key=lambda x: order[x])]
    print(p.to_latex())


def success_by_attack(df):
    return df.groupby(['model', 'attack_type'])[['success']].mean()


def l2_by_successfull_attack(df):
    successfull_attacks = df[df['success'] == True]
    return successfull_attacks.groupby(['model', 'attack_type'])[['mse']].mean()


def aggregate_results(df, values, others_to_group=None):
    if others_to_group is None:
        to_group = ['model', 'attack_type']
        indexes = ['model']
    else:
        to_group = ['model', 'attack_type'] + others_to_group
        indexes = ['model'] + others_to_group

    subset = df.groupby(to_group)[values].mean()
    p = subset.reset_index().pivot_table(index=indexes, values=values,
                                         columns=['attack_type'], )
    return p


def aggregate_results_multiple(df, values, others_to_group=None):
    if others_to_group is None:
        to_group = ['model', 'attack_type']
        indexes = ['model']
    else:
        to_group = ['model', 'attack_type'] + others_to_group
        indexes = ['model'] + others_to_group

    subset = df.groupby(to_group)[values].mean()

    return subset.stack().unstack('attack_type').unstack()


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
parser.add_argument('results_path', nargs='*')

args = parser.parse_args()

all_results_genuine = []
all_results_forgery = []
for f in args.results_path:
    results_genuine, results_forgery = load_results(f)
    all_results_genuine.append(results_genuine)
    all_results_forgery.append(results_forgery)

results_genuine = pd.concat(all_results_genuine)
results_forgery = pd.concat(all_results_forgery)

print('Number of rows: {}'.format(len(results_genuine)))


print('Attacks on Genuine - success')
pd.set_option('display.float_format', format_pct)
p = aggregate_results(results_genuine, 'success')
p = p.reindex(columns=['fgm', 'carlini', 'anneal', 'decision'])
print(p.to_latex())


print('Attacks on Genuine - l2 of successful attacks')
pd.set_option('display.float_format', format_normal)
df = results_genuine
successful_attacks = df[df['success'] == True]
p = aggregate_results(successful_attacks, 'rmse')
#p.columns = p.columns.get_level_values(1)
p = p.reindex(columns=['fgm', 'carlini', 'anneal', 'decision'])
print(p.to_latex())


print('Attacks on forgeries - success')
pd.set_option('display.float_format', format_pct)
p = aggregate_results(results_forgery, 'success', others_to_group=['image_type'])
p = p.reindex(columns=['fgm', 'carlini', 'anneal', 'decision'])
print(p.to_latex())


print('Attacks on forgeries - l2 of successful attacks')
pd.set_option('display.float_format', format_normal)

df = results_forgery
successful_attacks = df[df['success'] == True]
p = aggregate_results(successful_attacks, 'rmse', others_to_group=['image_type'])
p=p.reindex(columns=['fgm', 'carlini', 'anneal', 'decision'])
print(p.to_latex())
