#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
from data_util import load_data
from bert_embedding import bert


def main():
    """
    main function
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='./nela-17/whole')

    args = parser.parse_args()

    data_path = args.data_path
    print("data_path : {}".format(data_path))

    dataset = load_data(data_path)

    bert(dataset)


if __name__ == '__main__':
    main()
