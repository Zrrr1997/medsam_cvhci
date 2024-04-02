
import argparse
import os
import pandas as pd

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

    print(len(df_time_2))
    print(len(df_time_1))