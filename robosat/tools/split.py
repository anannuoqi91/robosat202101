############################################################
#     张琪   2021-01-05
#     新增切割训练数据集的方法
############################################################

import os
import argparse
import random
import csv

from tqdm import tqdm

from robosat.tiles import tiles_from_csv


def add_parser(subparser):
    parser = subparser.add_parser(
        "split",
        help="divide images into three parts: training data set, validation data set and evaluation data set using a csv",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("tiles", type=str, help="csv to filter images by")
    parser.add_argument("training", type=int, help="percentage of training data set")
    parser.add_argument("validation", type=int, help="percentage of validation data set")
    parser.add_argument("evaluation", type=int, help="percentage of evaluation data set")
    parser.add_argument("out", type=str, help="directory to save filtered tiles to")

    parser.set_defaults(func=main)


def main(args):

    def difference_set(x, y):
        """
        x包含y(set)
        :param x:
        :param y:
        :return:
        """
        if y is not None:
            return x - y, None
        else:
            return x, None

    path = args.tiles
    label = ['training', 'validation', 'evaluation']
    data_set = {'training': None,
                'validation': None,
                'evaluation': None
                }
    data_rate = {'training': args.training,
                'validation': args.validation,
                'evaluation' : args.evaluation
                }

    tiles = set(tiles_from_csv(path))
    num = len(tiles)
    for data_label in tqdm(label, desc="split", ascii=True):
        out = os.path.join(args.out, 'csv_' + data_label + '.tiles')
        rate = data_rate[data_label]
        for i in data_set.keys():
            if i != data_label:
                tiles, data_set[i] = difference_set(tiles, data_set[i])
        tiles_list = list(tiles)
        tiles_out = random.sample(tiles_list, int(num * rate))
        with open(out, "w") as fp:
            writer = csv.writer(fp)
            writer.writerows(tiles_out)