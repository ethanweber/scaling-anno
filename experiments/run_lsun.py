import argparse
from lsun import LSUN
import numpy as np
import multiprocessing
import pprint
import json
import sys
sys.path.insert(0, "..")
from utils.common import load_from_json, write_to_json

#  0 : 'bed',
#  5 : 'table'
#  8 : 'car'
#  18: 'lamp'


def get_features(class_idx):
    filename = ("/data/vision/torralba/scratch/dimpapa/git_master_scaleade/scaleade/clustering/outputs/trainBval/"
                "001_evalTrainBval/simPlacesNew/fg_attention_coarse/{:03d}/008000000_0002_sampleData.npy").format(class_idx)
    return np.load(filename)


def run_lsun(arguments):
    class_idx, sample_size, iters = arguments

    print("Running LSUN experiment for class_idx {}".format(class_idx))

    features = get_features(class_idx)
    X_all = features[:, :2].astype("float32")
    ious = features[:, 2]
    gt = np.zeros_like(ious)
    gt[ious >= 0.75] = 1
    y_all = gt.astype("float32")
    assert X_all.shape[0] == y_all.shape[0]

    # run the experiment
    lsun = LSUN(X_all, y_all)
    data = lsun.run(iters=iters, sample_size=sample_size, ious=ious)

    # save the results
    filename = "./lsun_results/{:03d}.json".format(class_idx)
    write_to_json(filename, data)


parser = argparse.ArgumentParser(description="")
parser.add_argument('--step', type=str)


def main():
    args = parser.parse_args()
    pprint.pprint(args)

    sample_size = 4000
    iters = 100
    class_indices = [0, 5, 8, 18]

    arguments_list = []
    for class_idx in class_indices:
        arguments_list.append(
            (class_idx, sample_size, iters)
        )

    p = multiprocessing.Pool(4)
    _ = p.map(run_lsun, arguments_list)


if __name__ == '__main__':
    main()
