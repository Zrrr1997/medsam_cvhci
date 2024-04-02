
import argparse
import os
import pandas as pd
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--csv_1', type=str, help='path to first csv file')
parser.add_argument('--csv_2', type=str, help='path to second csv file')

args = parser.parse_args()
csv_1 = args.csv_1
csv_2 = args.csv_2

if __name__ == '__main__':
    df_metrics_1 = pd.read_csv(os.path.join(csv_1, 'metrics.csv'))
    df_time_1 = pd.read_csv(os.path.join(csv_1, 'efficiency.csv'))



    df_metrics_2 = pd.read_csv(os.path.join(csv_2, 'metrics.csv'))
    df_time_2 = pd.read_csv(os.path.join(csv_2, 'efficiency.csv'))


    assert len(df_time_2) == len(df_time_1)
    assert len(df_metrics_1) == len(df_metrics_2)

    time_ranking = (df_time_1['time'] >= df_time_2['time']) * 1 

    time_ranking = time_ranking.to_numpy()

    dsc_ranking = (df_metrics_1['dsc'] <= df_metrics_2['dsc']) * 1
    dsc_ranking = dsc_ranking.to_numpy()

    nsd_ranking = (df_metrics_1['nsd'] <= df_metrics_2['nsd']) * 1   
    nsd_ranking = nsd_ranking.to_numpy()
    time, dsc, nsd = np.mean(time_ranking), np.mean(dsc_ranking), np.mean(nsd_ranking)
    print(time, dsc, nsd)
    ranking = (time + dsc + nsd) / 3
    print(ranking)
