"""Run the experiments on ADE.
"""

import torch
import itertools
from tqdm import tqdm
import os
import argparse
import numpy as np
import copy
import json
import pprint
import time
from collections import defaultdict
from sklearn.metrics.pairwise import pairwise_distances
from scipy.spatial.distance import squareform
import sys
sys.path.insert(0, "../")


import multiprocessing
from src.search import Tree
from src.utils import load_from_json, write_to_json, get_chunks, make_dir, pjoin
from src.places import ExportPlaces, PlacesFolder
from src.coco_dataset import COCODataset
from src.cluster import Clusterer


# globl variables
FEATURE_NAMES = [
    "attention",
    "attention_quality_score",
    "backbone",
    "iou_embedding",
    "mask",
    "quality_score",
    "random",
    "pred_iou",
    "score"
]


def add_gt(dataset, filename):
    assert filename.endswith("instances_predictions.pth")
    gt_filename = filename.replace("instances_predictions.pth", "instances_predictions_with_gt.pth")
    instances = torch.load(filename)
    num_cpus = 18
    print("num_cpus,", num_cpus)
    chunks = get_chunks(instances, num_cpus)
    print("got chunks")
    p = multiprocessing.Pool(num_cpus)
    chunks = p.map(dataset.add_gt_to_inference_predictions, chunks)
    instances = list(itertools.chain.from_iterable(chunks))
    # save to gt
    print("Saving gt file.")
    torch.save(instances, gt_filename)


def export(filename, folder=None):
    assert folder is not None
    assert filename.endswith("instances_predictions_with_gt.pth")
    instances = torch.load(filename)

    # the instances:
    category_id_to_instances = defaultdict(list)
    # the features:
    # TODO(ethan): make this a concrete list
    category_id_to_features_attention = defaultdict(list)
    category_id_to_features_backbone = defaultdict(list)
    category_id_to_features_iou_embedding = defaultdict(list)
    category_id_to_features_mask = defaultdict(list)


    print("\n\nusing score\n\n")
    # process the instances
    count = 0
    for i in tqdm(range(len(instances))):
        for j in range(len(instances[i]["instances"])):
            score = instances[i]["instances"][j]["score"]
            category_id = instances[i]["instances"][j]["category_id"]
            if (category_id != 0 and score > 0.5) or (category_id == 0 and score > 0.90):
                # print(score)
                # return None
                category_id_to_instances[category_id].append(instances[i]["instances"][j])
                # add the features and remove them after

                category_id_to_features_attention[category_id].append(
                    instances[i]["instances"][j]["features_attention"])
                # category_id_to_features_backbone[category_id].append(
                #     instances[i]["instances"][j]["features_backbone"])
                # category_id_to_features_iou_embedding[category_id].append(
                #     instances[i]["instances"][j]["features_iou_embedding"])
                # category_id_to_features_mask[category_id].append(instances[i]["instances"][j]["features_mask"])

                # delete the features!
                del instances[i]["instances"][j]["features_attention"]
                # del instances[i]["instances"][j]["features_backbone"]
                # del instances[i]["instances"][j]["features_iou_embedding"]
                # del instances[i]["instances"][j]["features_mask"]

                count += 1

    print("num:", count)

    # save the results
    print("Saving the info.")
    # TODO(ethan): save this saving process
    for category_id in tqdm(category_id_to_instances.keys()):
        # save the instances
        instances_filename = os.path.join(folder, "instances", "{:03d}.json".format(category_id))
        make_dir(instances_filename)
        with open(instances_filename, "w") as outfile:
            json.dump(category_id_to_instances[category_id], outfile)

        # save the features
        features_filename = os.path.join(folder, "features", "attention", "{:03d}.npy".format(category_id))
        make_dir(features_filename)
        np.save(features_filename, category_id_to_features_attention[category_id])

        # features_filename = os.path.join(folder, "features", "backbone", "{:03d}.npy".format(category_id))
        # make_dir(features_filename)
        # np.save(features_filename, category_id_to_features_backbone[category_id])

        # features_filename = os.path.join(folder, "features", "iou_embedding", "{:03d}.npy".format(category_id))
        # make_dir(features_filename)
        # np.save(features_filename, category_id_to_features_iou_embedding[category_id])

        # features_filename = os.path.join(folder, "features", "mask", "{:03d}.npy".format(category_id))
        # make_dir(features_filename)
        # np.save(features_filename, category_id_to_features_mask[category_id])


def load_features(class_idx, name, folder=None):
    filename = os.path.join(folder, "features", name, "{:03d}.npy".format(class_idx))
    features = np.load(filename)
    return features


def get_distances(class_idx, name, instances, folder=None):
    """
    Returns the features as distances.
    Returns distances.
    """
    assert folder is not None
    distances = None
    if name == "attention":
        features = load_features(class_idx, "attention", folder=folder)
        distances = pairwise_distances(features, n_jobs=1)
        distances = squareform(distances)
    if name == "attention_quality_score":
        features = load_features(class_idx, "attention", folder=folder)

        featuresI = np.array([instances[i]["pred_iou"][0] for i in range(len(instances))]).astype("float32")
        featuresI = featuresI.reshape((featuresI.shape[0], 1))
        featuresS = np.array([instances[i]["score"] for i in range(len(instances))]).astype("float32")
        featuresS = featuresS.reshape((featuresS.shape[0], 1))

        features_quality_score = featuresI * featuresS  # combine

        distF = pairwise_distances(features, n_jobs=1)
        distI = pairwise_distances(features_quality_score, n_jobs=1)
        distF = distF - np.mean(distF)
        distF = distF / np.std(distF)
        distI = distI - np.mean(distI)
        distI = distI / np.std(distI)
        distances = distF + distI
        distances = distances + np.absolute(np.min(distances))
        distances = squareform(distances)

    if name == "backbone":
        features = load_features(class_idx, "backbone", folder=folder)
        distances = pairwise_distances(features, n_jobs=1)
        distances = squareform(distances)
    if name == "iou_embedding":
        features = load_features(class_idx, "iou_embedding", folder=folder)
        distances = pairwise_distances(features, n_jobs=1)
        distances = squareform(distances)
    if name == "mask":
        features = load_features(class_idx, "mask", folder=folder)
        distances = pairwise_distances(features, n_jobs=1)
        distances = squareform(distances)
    if name == "pred_iou":
        featuresI = np.array([instances[i]["pred_iou"][0] for i in range(len(instances))]).astype("float32")
        featuresI = featuresI.reshape((featuresI.shape[0], 1))
        distances = pairwise_distances(featuresI, n_jobs=1)
        distances = squareform(distances)
    if name == "score":
        featuresS = np.array([instances[i]["score"] for i in range(len(instances))]).astype("float32")
        featuresS = featuresS.reshape((featuresS.shape[0], 1))
        distances = pairwise_distances(featuresS, n_jobs=1)
        distances = squareform(distances)
    if name == "quality_score":
        featuresI = np.array([instances[i]["pred_iou"][0] for i in range(len(instances))]).astype("float32")
        featuresI = featuresI.reshape((featuresI.shape[0], 1))
        featuresS = np.array([instances[i]["score"] for i in range(len(instances))]).astype("float32")
        featuresS = featuresS.reshape((featuresS.shape[0], 1))
        features_quality_score = featuresI * featuresS  # combine
        distances = pairwise_distances(features_quality_score, n_jobs=1)
        distances = squareform(distances)
    if name == "random":
        features = np.random.rand(len(instances), 1).astype("float32")
        distances = pairwise_distances(features, n_jobs=1)
        distances = squareform(distances)

    assert type(distances) == np.ndarray
    return distances


def cluster(arguments):
    t0 = time.time()
    class_idx, name, folder = arguments
    print("Starting class {:03d}".format(class_idx))
    instances_filename = os.path.join(folder, "instances", "{:03d}.json".format(class_idx))
    instances = load_from_json(instances_filename)
    distances = get_distances(class_idx, name, instances, folder=folder)

    clusterer = Clusterer(None, distances=distances, cluster_method="hac", linkage_method="complete")
    clusterer.cluster()

    filename = os.path.join(folder, "trees", name, "{:03d}.npy".format(class_idx))
    make_dir(filename)
    np.save(filename, clusterer.Z)
    t1 = time.time()
    total = t1 - t0
    print("Finished clustering {:03d} in {} sec.".format(class_idx, total))


# parser = argparse.ArgumentParser(description="Run the experiments on ADE.")
# parser.add_argument('--step', type=str)
# parser.add_argument('--round-number', type=int, default=None)
# parser.add_argument('--place_indices', type=int, default=None)


from src.args import get_parsed_args


# for ADE:
# GT_DATA_FILENAME = "/data/vision/torralba/scratch/dimpapa/git_iccv21/scaleade/data/ade/classes80/trainB.json"
# for COCO:
# GT_DATA_FILENAME = "/data/vision/torralba/scratch/dimpapa/iccv_rebuttal/coco/annotations/val120k.json"
# TRAIN_GT_DATA_FILENAME = "/data/vision/torralba/scratch/dimpapa/iccv_rebuttal/coco/annotations/train2k.json"
# GT_DATA_FILENAME = "/data/vision/torralba/scratch/dimpapa/iccv_rebuttal/coco/annotations/val120k_rle.json"
# for openimages:
GT_DATA_FILENAME = "/data/vision/torralba/scratch/dimpapa/iccv_rebuttal/openimages/dim_val_coco_only60Classes.json"


# IMAGE_PATH = "/data/vision/torralba/ade20k-places/data/"
# IMAGE_PATH = "/data/vision/torralba/scratch/dimpapa/iccv_rebuttal/coco/"
IMAGE_PATH = "/data/vision/torralba/scratch/dimpapa/openImages/"


# gt_data_filename = "/data/vision/torralba/scratch/dimpapa/git_iccv21/scaleade/data/ade/classes80/trainA_1k.json"
# gt_data_filename = "/data/vision/torralba/scratch/dimpapa/git_iccv21/scaleade/data/ade/classes80/trainA_2k.json"
# gt_data_filename = "/data/vision/torralba/scratch/dimpapa/git_iccv21/scaleade/data/ade/classes80/trainA.json"
# gt_data_filename = "/data/vision/torralba/scratch/dimpapa/git_iccv21/scaleade/data/ade/classes80/trainA_500.json"
# gt_data_filename = "/data/vision/torralba/scratch/dimpapa/git_iccv21/scaleade/data/ade/classes80/trainA_5k.json"
# gt_data_filename = "/data/vision/torralba/scratch/dimpapa/git_iccv21/scaleade/data/ade/classes80/trainA_100.json"


# note
# score of 0.9 vs. 0.98

def main():
    # args = get_parsed_args(version="ade")
    args = get_parsed_args(version="coco")
    pprint.pprint(args)

    if args.step == "inference":
        #
        ###
        export_places = ExportPlaces(args.place_indices,
                                     args.places_round,
                                     export_places_path=args.export_places_path,
                                     config=args.config,
                                     model_weights=args.model_weights,
                                     places_dataset_prefix=args.places_dataset_prefix)
        print(export_places.get_cmd(0, -1))

    if args.step == "add_gt":
        print("Adding GT to the instances file!")
        places_folder = PlacesFolder(
            args.place_indices,
            args.places_round,
            category_indices=args.category_indices,
            export_places_path=args.export_places_path
        )
        print("got PlacesFolder")
        dataset = COCODataset(
            image_path=IMAGE_PATH,
            gt_data_filename=GT_DATA_FILENAME,
        )
        print("Loaded dataset.")
        filename = places_folder.get_instance_path(-1)
        print("Calling add_gt")
        add_gt(dataset, filename)

    if args.step == "export":
        print("Exporting the instances w/ GT to class-specific format.")

        places_folder = PlacesFolder(
            args.place_indices,
            args.places_round,
            category_indices=args.category_indices,
            export_places_path=args.export_places_path
        )
        filename = places_folder.get_gt_instance_path(-1)
        export(filename, folder=places_folder.round_dir)

    if args.step == "nn":
        print("Computing the NN for the instances.")

    if args.step == "cluster":
        # class_indices = list(range(0, 100))
        # class_indices = list(range(0, 1))
        # names = FEATURE_NAMES
        class_indices = args.category_indices
        names = ["attention_quality_score"]

        places_folder = PlacesFolder(
            args.place_indices,
            args.places_round,
            category_indices=class_indices,
            export_places_path=args.export_places_path
        )
        folder = places_folder.round_dir

        for name in names:
            num_cpus = 24
            p = multiprocessing.Pool(num_cpus)
            arguments = [(class_idx, name, folder) for class_idx in class_indices]
            chunks = p.map(cluster, arguments)
        # cluster((0, "attention_quality_score", folder))

    if args.step == "search":
        print("Searching!")

        class_indices = args.category_indices
        places_folder = PlacesFolder(
            args.place_indices,
            args.places_round,
            category_indices=class_indices,
            export_places_path=args.export_places_path
        )
        folder = places_folder.round_dir

        # feature_name = "attention_quality_score"

        feature_names = ["attention_quality_score"]
        # feature_names = FEATURE_NAMES

        # # # -- block ---
        # search_methods = [
        #     "universal_threshold",
        #     "bfs",
        #     "dfs",
        #     # "random_walk",
        #     "heuristics_only"
        # ]
        # heuristics = [
        #     "none",
        #     "rand",
        #     "score_iou",
        #     "real_iou"
        # ]
        # K_prior_1 = [0.0]
        # K_prior_2 = [0.0]
        # min_num_samples_list = [1]

        # -- block ---
        # search_methods = ["heuristics_only"]
        # heuristics = ["score_iou"]
        # K_prior_1 = [0.0]
        # K_prior_2 = [0.5, 0.7, 0.9]
        # min_num_samples_list = [1]

        # -- block ---
        search_methods = ["heuristics_only"]
        heuristics = ["score_iou"]
        K_prior_1 = [0.0]
        K_prior_2 = [0.7]
        # min_num_samples_list = [1, 10, 20, 50, 100]
        min_num_samples_list = [1]

        K_priors = []
        for k1 in K_prior_1:
            for k2 in K_prior_2:
                K_priors.append(
                    (k1, k2)
                )

        arguments = []
        for feature_name in feature_names:
            for class_idx in class_indices:
                for search_method in search_methods:
                    for heuristic in heuristics:
                        for K_prior in K_priors:
                            for min_num_samples in min_num_samples_list:
                                arguments.append(
                                    (
                                        folder,
                                        class_idx,
                                        feature_name,
                                        search_method,
                                        heuristic,
                                        K_prior[0],
                                        K_prior[1],
                                        min_num_samples
                                    )
                                )

        num_cpus = 24
        p = multiprocessing.Pool(num_cpus)
        _ = p.map(Tree.export_helper, arguments)

    if args.step == "export_datasets":
        print("Searching and exporting the datasets.")

        class_indices = args.category_indices
        places_folder = PlacesFolder(
            args.place_indices,
            args.places_round,
            category_indices=class_indices,
            export_places_path=args.export_places_path
        )
        folder = places_folder.round_dir

        # TODO: rethink the naming to define a search. should be a dictionary that we pass around?
        search_algorithm = "attention_quality_score-heuristics_only-real_iou-000_070-001"

        Tree.export_dataset_with_annotations(
            search_algorithm,
            input_folder=folder,
            image_path=IMAGE_PATH,
            train_gt_data_filename=TRAIN_GT_DATA_FILENAME,
            gt_data_filename=GT_DATA_FILENAME,
            class_indices=class_indices
        )

    if args.step == "export_datasets_oracle":
        input_folder = "/data/vision/torralba/scratch/ethanweber/scaleade/ade_rounds/round_00"
        search_algorithms = [
            "attention_quality_score-bfs-rand-naive-001",
            "attention_quality_score-heuristics_only-real_iou-020_070-001",
        ]

        for search_algorithm in search_algorithms:
            gt_data_filename = os.path.join(
                input_folder, "exported_datasets", search_algorithm + ".json"
            )
            print("Original:", gt_data_filename)
            assert os.path.exists(gt_data_filename)
            oracle_gt_data_filename = gt_data_filename.replace(".json", "-oracle.json")
            print("Oracle:", oracle_gt_data_filename)

            # load the dataset
            count = 0
            instances = load_from_json(gt_data_filename)
            for i in reversed(range(len(instances["annotations"]))):
                if "iou" in instances["annotations"][i]:
                    if instances["annotations"][i]["iou"][0] < 0.85:
                        del instances["annotations"][i]
                        count += 1
            print("Removed {} annotations w/ low quality.".format(count))

            # save the oracle dataset!
            with open(oracle_gt_data_filename, "w") as outfile:
                json.dump(instances, outfile)

    if args.step == "add_non_negatives":
        places_folder = PlacesFolder(
            args.place_indices,
            args.places_round,
            category_indices=args.category_indices,
            export_places_path=args.export_places_path
        )
        folder = places_folder.get_round_dir()
        gt_data_filename = pjoin(
            folder, "exported_datasets/attention_quality_score-heuristics_only-real_iou-000_070-001.json")

        dataset = COCODataset(
            image_path="/data/vision/torralba/ade20k-places/data/",
            gt_data_filename=gt_data_filename,
        )

        ade_train19K_dataset = COCODataset(
            image_path="/data/vision/torralba/ade20k-places/data/",
            gt_data_filename="/data/vision/torralba/scratch/dimpapa/git_iccv21/scaleade/data/ade/classes80/trainB.json",
        )

        # load all the instances...
        all_instances = []
        class_indices = args.category_indices
        for class_idx in tqdm(class_indices):
            filename = pjoin(folder, "instances/{:03d}.json".format(class_idx))
            if not os.path.exists(filename):
                print("Skipping filename: {}".format(filename))
                continue
            all_instances += load_from_json(filename)

        keys_already_added = set()
        used_image_filenames = set()
        for instance in tqdm(dataset.gt_data["annotations"]):
            # check if already in the dataset...
            image_filename = dataset.image_filename(instance)
            used_image_filenames.add(image_filename)
            if "instance_idx" in instance:
                # NOTE(ethan): hack!!
                key = str((image_filename, instance["bbox"]))
                keys_already_added.add(key)

        for instance in tqdm(all_instances):
            image_filename = ade_train19K_dataset.image_filename(instance)
            key = str((image_filename, instance["bbox"]))
            if key not in keys_already_added and image_filename in used_image_filenames:
                # print("addiing key", key)
                image_id = ade_train19K_dataset.image_id(instance)
                image_data = ade_train19K_dataset.image_id_to_image_data[image_id]
                ins = copy.deepcopy(instance)
                ins["accepted"] = 0
                dataset.add_instance_to_gt_data(ins, image_data)

        output_filename = gt_data_filename.replace("exported_datasets", "exported_datasets_with_non_neg")
        make_dir(output_filename)
        write_to_json(output_filename, dataset.gt_data)

    if args.step == "remove_overlapping":
        print("Removing overlapping instances from previous iterations!")

        # gt_data_filename = "/data/vision/torralba/scratch/ethanweber/scaleade/ade_rounds/round_01/exported_datasets/attention_quality_score-heuristics_only-real_iou-020_070-001.json"
        # gt_data_filename = "/data/vision/torralba/scratch/ethanweber/scaleade/ade_rounds/round_02/exported_datasets/attention_quality_score-heuristics_only-real_iou-020_070-001.json"
        # gt_data_filename = "/data/vision/torralba/scratch/ethanweber/scaleade/ade_rounds/round_03/exported_datasets/attention_quality_score-heuristics_only-real_iou-020_070-001.json"
        gt_data_filename = "/data/vision/torralba/scratch/ethanweber/scaleade/ade_rounds/round_04/exported_datasets/attention_quality_score-heuristics_only-real_iou-020_070-001.json"

        # temp
        # gt_data_filename = "/data/vision/torralba/scratch/ethanweber/scaleade/ade_rounds/round_01/exported_datasets/attention_quality_score-heuristics_only-real_iou-020_070-001-withoutduplicates.json"

        dataset = COCODataset(
            image_path="/data/vision/torralba/ade20k-places/data/",
            gt_data_filename=gt_data_filename,
        )

        # filename = "/data/vision/torralba/scratch/ethanweber/scaleade/ade_rounds/round_01/exported_datasets/attention_quality_score-heuristics_only-real_iou-020_070-001-withoutduplicates.json"
        # filename = "/data/vision/torralba/scratch/ethanweber/scaleade/ade_rounds/round_02/exported_datasets/attention_quality_score-heuristics_only-real_iou-020_070-001-withoutduplicates.json"
        # filename = "/data/vision/torralba/scratch/ethanweber/scaleade/ade_rounds/round_03/exported_datasets/attention_quality_score-heuristics_only-real_iou-020_070-001-withoutduplicates.json"
        filename = "/data/vision/torralba/scratch/ethanweber/scaleade/ade_rounds/round_04/exported_datasets/attention_quality_score-heuristics_only-real_iou-020_070-001-withoutduplicates.json"

        # temp
        # filename = "/data/vision/torralba/scratch/ethanweber/scaleade/ade_rounds/round_01/exported_datasets/attention_quality_score-heuristics_only-real_iou-020_070-001-withoutduplicates-temp.json"

        dataset.save_file_without_duplicates(filename)


if __name__ == "__main__":
    main()
