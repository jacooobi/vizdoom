#!/usr/bin/env python

import matplotlib.pyplot as plt
import pandas as pd
import argparse


def get_series(file_path):
    df = pd.read_csv(file_path)
    df = df['total_reward']
    return df


def create_plot(series, dpi):
    fig, ax = plt.subplots(figsize=(6, 3), dpi=dpi)
    ax.plot(series)
    return fig, ax


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('log_path', type=str, help='Statistics file')
    parser.add_argument('--save-to-file', type=str,
                        help='Image output filename')
    parser.add_argument('--rolling-mean', type=int,
                        help='Rolling mean window size')
    parser.add_argument('--dpi', type=int, default=144)
    args = parser.parse_args()

    series = get_series(args.log_path)
    if args.rolling_mean:
        series = pd.Series(series).rolling(
            window=args.rolling_mean, center=False).mean()
    series = list(series)

    ax, fig = create_plot(series, args.dpi)

    if args.save_to_file:
        plt.savefig(args.save_to_file)
    else:
        plt.show()
