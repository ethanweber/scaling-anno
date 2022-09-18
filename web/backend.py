"""To quickly fetch dataset data needed for HITs.
"""

import sys
sys.path.insert(0, "../")
import argparse
from src.utils import load_from_json, get_mask_from_polygons, get_iou, get_polygons_from_rle
from src.coco_dataset import COCODataset
import os
import pickle
import pandas as pd
from flask_cors import CORS, cross_origin
from flask import (Flask,
                   render_template,
                   send_from_directory,
                   jsonify,
                   request)
import json
import glob
from os.path import join as pjoin


app = Flask(__name__, static_url_path="/static")
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'
app.jinja_env.filters['zip'] = zip

# the datasets we care about
datasets = {}

print("Loading ADE COCO dataset.")
ade_coco_dataset = COCODataset(
    image_path="/data/vision/torralba/ade20k-places/data/",
    gt_data_filename="/data/vision/torralba/scaleade/scaleade/data/ade/trainBval.json",
)

print("Loading Places COCO dataset.")
# places_coco_dataset = COCODataset(
#     image_path="/data/vision/torralba/ade20k-places/data/",
#     gt_data_filename="/data/vision/torralba/scratch/ethanweber/scaleade/data/places/merged.json",
# )

places_coco_dataset = COCODataset(
    image_path="/data/vision/torralba/ade20k-places/data/",
    gt_data_filename="/data/vision/torralba/scratch/ethanweber/scaleade/rounds/iccv/places_ade_1k_COCO/merged.json",
)
# PLACES_INSTANCES_PATH = "/data/vision/torralba/scratch/ethanweber/scaleade/places/places_round_00/all_15/instances/"
PLACES_INSTANCES_PATH = "/data/vision/torralba/scratch/ethanweber/scaleade/rounds/iccv/places_ade_1k_COCO/round_00/all/instances"


def get_dataset_key(dataset_name, class_idx):
    assert isinstance(dataset_name, str)
    assert isinstance(class_idx, str)
    return dataset_name + ";" + class_idx


def quickly_get_dataset(dataset_key,
                        places_instances_path=PLACES_INSTANCES_PATH):
    print("Loading {}".format(dataset_key))
    dataset_name, class_idx = dataset_key.split(";")
    if dataset_name == "ade":
        filename = "/data/vision/torralba/scratch/ethanweber/scaleade/rounds/cvpr/ade/instances/{:03d}.json".format(
            int(class_idx))
    elif dataset_name == "places":
        # NOTE(ethan): "merged" vs. "all"
        # "all" contains the nearest neighbors
        # with open("/data/vision/torralba/scratch/ethanweber/scaleade/places/places_round_00/all/instances/{:03d}.json".format(int(class_idx))) as f:
        filename = pjoin(places_instances_path, "{:03d}.json".format(int(class_idx)))

    # return the dataset if it exists!
    if os.path.exists(filename):
        instances = load_from_json(filename)
        dataset = instances
        print("Loaded {}".format(dataset_key))
        return dataset
    else:
        print(f"Filename {filename} does not exist. Returning None.")
        return None


# Returns either the annotation or image level data stored.
def return_data(dataset_key, idx, format=None):
    assert format is not None
    assert format in ["annotation", "image"]
    if dataset_key not in datasets.keys():
        return jsonify({"success": False, "data": "You can't access this right now."})
    elif datasets[dataset_key] is None:
        # TODO(ethan): add something that waits for the dataset to be loaded
        return jsonify({"success": False, "data": "Still loading dataset."})

    result = {}
    try:
        result["success"] = True
        dataset_name, class_idx = dataset_key.split(";")
        if format == "annotation":
            instance = datasets[dataset_key][int(idx)]
            if dataset_name == "ade":
                result["data"] = ade_coco_dataset.get_annotation_data(instance)
            elif dataset_name == "places":
                result["data"] = places_coco_dataset.get_annotation_data(instance)
        elif format == "image":
            if dataset_name == "ade":
                result["data"] = ade_coco_dataset.get_image_data(int(idx))
            elif dataset_name == "places":
                result["data"] = places_coco_dataset.get_image_data(int(idx))
    except:
        result["success"] = False
        result["data"] = "Uhh something broke."
    return jsonify(result)


# Returns number of annotations.
def return_num_annotations(dataset_key):
    if dataset_key not in datasets.keys():
        return jsonify({"success": False, "data": "Dataset is not loaded."})
    result = {}
    try:
        result["success"] = True
        result["data"] = len(datasets[dataset_key])
    except:
        result["success"] = False
        result["data"] = "Uhh something broke."
    return jsonify(result)


# Returns data for a specific dataset and annotation id.
@app.route('/annotations/<dataset_name>/<class_idx>/<annotation_id>', methods=["GET"])
@cross_origin()
def annotations(dataset_name, class_idx, annotation_id):
    dataset_key = get_dataset_key(dataset_name, class_idx)
    return return_data(dataset_key, annotation_id, format="annotation")


# Returns the number of annotation ids in a dataset.
@app.route('/num_annotations/<dataset_name>/<class_idx>', methods=["GET"])
@cross_origin()
def num_annotations(dataset_name, class_idx):
    dataset_key = get_dataset_key(dataset_name, class_idx)
    return return_num_annotations(dataset_key)


# Returns data on an image in a specified dataset.
@app.route('/images/<dataset_name>/<class_idx>/<image_id>', methods=["GET"])
@cross_origin()
def images(dataset_name, class_idx, image_id):
    dataset_key = get_dataset_key(dataset_name, class_idx)
    return return_data(dataset_key, image_id, format="image")


@app.route('/rle_to_polygon', methods=["GET", "POST"])
@cross_origin()
def rle_to_polygon():
    req_data = request.get_json()
    result = {}
    if request.method != 'POST':
        result["success"] = False
        result["data"] = "Use a POST request."
    else:
        try:
            polygons = get_polygons_from_rle(req_data["rle"])
            result["success"] = True
            result["data"] = polygons
        except:
            result["success"] = False
            result["data"] = "Failed to compute polygons."
    return jsonify(result)


@app.route('/iou_from_polygons', methods=["GET", "POST"])
@cross_origin()
def iou_from_polygons():
    req_data = request.get_json()
    result = {}
    if request.method != 'POST':
        result["success"] = False
        result["data"] = "Use a POST request."
    else:
        polygons1 = req_data["polygons1"]
        polygons2 = req_data["polygons2"]
        height = req_data["height"]
        width = req_data["width"]
        mask1 = get_mask_from_polygons(polygons1, height=height, width=width)
        mask2 = get_mask_from_polygons(polygons2, height=height, width=width)
        iou = get_iou(mask1, mask2)
        result["success"] = True
        result["data"] = iou
    return jsonify(result)


parser = argparse.ArgumentParser()
parser.add_argument('--start', type=int, default=0, help="Starting class to use.")
parser.add_argument('--end', type=int, default=100, help="Ending class to use.")
parser.add_argument('--round', type=int, default=0, help="Which round of the loop this is. This is for places.")
parser.add_argument('--port', type=int, default=8093, help="Port to use.")

if __name__ == '__main__':
    print("starting main")
    args = parser.parse_args()

    # Preload all datasets
    preload_datasets = ["ade", "places"]
    preload_classes = [i for i in range(args.start, args.end)]

    import multiprocessing
    from multiprocessing import Pool

    dataset_keys = []
    for preload_class in preload_classes:
        for preload_dataset in preload_datasets:
            dataset_keys.append(get_dataset_key(preload_dataset, str(preload_class)))
    with Pool(multiprocessing.cpu_count()) as p:
        quick_datasets = p.map(quickly_get_dataset, dataset_keys)
        for idx, dataset_key in enumerate(dataset_keys):
            datasets[dataset_key] = quick_datasets[idx]

    app.run(debug=False, threaded=True, host="0.0.0.0", port=args.port)
