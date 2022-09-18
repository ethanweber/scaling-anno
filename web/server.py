"""To run the mturk.py tasks locally.
"""

import sys
import json
import glob
import yaml

sys.path.insert(0, "../")

from flask import (Flask,
                   render_template,
                   send_from_directory,
                   jsonify,
                   request)
import pandas as pd
import os
from src.utils import load_from_json
from src.tree import ClusterHierarchy

# cluster_hierarchy = ClusterHierarchy(
#     cluster_folder_name="/data/vision/torralba/scratch/ethanweber/scaleade/places/places_round_00/merged/trees/")

CLUSTER_FOLDER_NAME = "/data/vision/torralba/scratch/ethanweber/scaleade/rounds/iccv/places_ade_1k_COCO/round_00/merged/trees"

cluster_hierarchy = ClusterHierarchy(cluster_folder_name=CLUSTER_FOLDER_NAME)

app = Flask(__name__, static_url_path="/static")
app.jinja_env.filters['zip'] = zip

global ROUND_NUMBER
ROUND_NUMBER = 6
global CONFIG_TYPE
CONFIG_TYPE = "binary"


# Get the children for a cluster.
@app.route('/children/<dataset_name>/<class_idx>/<cluster_id>', methods=["GET"])
def children(dataset_name, class_idx, cluster_id):
    return jsonify(
        cluster_hierarchy.get_children_from_class_idx_and_cluster_id(
            int(class_idx),
            int(cluster_id))
    )


# Get the parent for a cluster.
@app.route('/parent/<dataset_name>/<class_idx>/<cluster_id>', methods=["GET"])
def parent(dataset_name, class_idx, cluster_id):
    return jsonify(
        cluster_hierarchy.get_parent_from_class_idx_and_cluster_id(
            int(class_idx),
            int(cluster_id))
    )


# Get the purity for a cluster.
@app.route('/purity/<dataset_name>/<class_idx>/<cluster_id>/<round_number>', methods=["GET"])
def purity(dataset_name, class_idx, cluster_id, round_number):
    return jsonify(
        cluster_hierarchy.get_purity_from_class_idx_and_cluster_id(
            int(class_idx),
            int(cluster_id),
            round_number=int(round_number)
        )
    )

# Get the quality estimate for a cluster.


@app.route('/quality/<dataset_name>/<class_idx>/<cluster_id>/<round_number>', methods=["GET"])
def quality(dataset_name, class_idx, cluster_id, round_number):
    return jsonify(
        cluster_hierarchy.get_quality_from_class_idx_and_cluster_id(
            int(class_idx),
            int(cluster_id),
            round_number=int(round_number)
        )
    )

# Get the segmentation ids in a cluster.


@app.route('/segment_ids/<dataset_name>/<class_idx>/<cluster_id>', methods=["GET"])
def segment_ids(dataset_name, class_idx, cluster_id):
    return jsonify(
        cluster_hierarchy.get_segment_ids_from_class_idx_and_cluster_id(
            int(class_idx),
            int(cluster_id))
    )

# Get the size (total number of segments) in a cluster.


@app.route('/size/<dataset_name>/<class_idx>/<cluster_id>', methods=["GET"])
def size(dataset_name, class_idx, cluster_id):
    return jsonify(
        int(cluster_hierarchy.get_size_from_class_idx_and_cluster_id(
            int(class_idx),
            int(cluster_id)))
    )

# Get the dendogram to draw a nice tree.


@app.route('/dendogram/<dataset_name>/<class_idx>/<cluster_id>', methods=["GET"])
def dendogram(dataset_name, class_idx, cluster_id):
    global ROUND_NUMBER
    return jsonify(
        cluster_hierarchy.get_tree_from_class_idx_and_cluster_id(
            int(class_idx),
            int(cluster_id),
            round_number=ROUND_NUMBER)
    )


# # Save a submitted polygon annotation.
# @app.route('/submit_annotation/<cluster_name>/<segment_id>', methods=["POST"])
# def submit_annotation(cluster_name, segment_id):
#     polygons = request.form['polygons']

#     # turn polygon into gt annotation
#     filename = cluster_name + "_" + str(segment_id) + ".json"

#     # save file with cluster info
#     with open(os.path.join("results", filename), "w") as f:
#         json.dump(polygons, f)
#     return jsonify({"info": "awesome, thanks!"})


# Get submitted mturk data locally.
@app.route('/mturk/externalSubmit', methods=["POST"])
def submit_external_submit():
    import pprint
    pprint.pprint(request)
    print(request.headers)
    return jsonify({"info": "you still need to make this!"})


# Webpage for propagation.
@app.route('/polygon/<cluster_name>/<image_idx>', methods=["GET"])
def polygon(cluster_name, image_idx):
    with open("pages/polygon.html", 'r') as file:
        html_as_str = file.read()

    modified_html = html_as_str.replace("${CLUSTER_NAME}", cluster_name)
    modified_html = modified_html.replace("${IMAGE_IDX}", image_idx)
    return modified_html


# Look at the dataset, with current annotations and GT annotations.
@app.route('/dataset', methods=["GET"])
def dataset():
    with open("pages/dataset.html", 'r') as file:
        html_as_str = file.read()

    # TODO(ethan): replace this code with something more rebust
    dataset_name = request.args.get('dataset_name', default="places")
    class_idx = request.args.get('class_idx', default="83")
    annotation_id = request.args.get('annotation_id', default="100")
    image_id = request.args.get('image_id', default="5905")
    modified_html = html_as_str.replace("${DATASET_NAME}", dataset_name)
    modified_html = modified_html.replace("${CLASS_IDX}", class_idx)
    modified_html = modified_html.replace("${ANNOTATION_ID}", annotation_id)
    modified_html = modified_html.replace("${IMAGE_ID}", image_id)
    return modified_html


# Quickly inspect the whole tree.
@app.route('/tree', methods=["GET"])
def tree():
    with open("pages/tree.html", 'r') as file:
        html_as_str = file.read()

    # TODO(ethan): replace this code with something more rebust
    dataset_name = request.args.get('dataset_name', default="places")
    class_idx = request.args.get('class_idx', default="82")
    cluster_id = request.args.get('cluster_id', default="9601")
    modified_html = html_as_str.replace("${DATASET_NAME}", dataset_name)
    modified_html = modified_html.replace(
        "${CLASS_NAME}", cluster_hierarchy.class_idx_to_class_name[int(class_idx)])
    modified_html = modified_html.replace("${CLASS_IDX}", class_idx)
    modified_html = modified_html.replace("${CLUSTER_ID}", cluster_id)
    return modified_html


# Get the config paramters for HIT.
@app.route('/hits/<round_number>/<hit_name>', methods=["GET"])
def hits(round_number, hit_name):
    config_data = load_from_json(os.path.join(CLUSTER_FOLDER_NAME.replace("merged/trees", "mturk/hits"),
                                              #   CONFIG_TYPE,
                                              "{:03d}".format(int(round_number)),
                                              hit_name + ".json"))
    return jsonify(config_data)


# Get the config paramters for HIT.
@app.route('/get_responses/<round_number>/<hit_name>', methods=["GET"])
def get_responses(round_number, hit_name):

    config_data = load_from_json(os.path.join(CLUSTER_FOLDER_NAME.replace("merged/trees", "mturk/responses"),
                                              #   CONFIG_TYPE,
                                              "{:03d}".format(int(round_number)),
                                              hit_name + ".json"))
    return jsonify(config_data)


# The MTurk task. <config_name> is used to specify the .json file config, specifying the task.
@app.route('/mturk/<round_number>/<config_name>', methods=["GET"])
def mturk(round_number, config_name):
    with open("pages/mturk.html", 'r') as file:
        html_as_str = file.read()
    modified_html = html_as_str.replace("${CONFIG_TYPE}", CONFIG_TYPE)
    modified_html = modified_html.replace("${ROUND_NUMBER}", "{:03d}".format(int(round_number)))
    modified_html = modified_html.replace("${CONFIG_NAME}", config_name)
    return modified_html


# The MTurk task. <config_name> is used to specify the .json file config, specifying the task.
@app.route('/responses/<round_number>/<config_name>', methods=["GET"])
def responses(round_number, config_name):
    with open("pages/responses.html", 'r') as file:
        html_as_str = file.read()
    modified_html = html_as_str.replace("${CONFIG_TYPE}", CONFIG_TYPE)
    modified_html = modified_html.replace("${ROUND_NUMBER}", "{:03d}".format(int(round_number)))
    modified_html = modified_html.replace("${CONFIG_NAME}", config_name)
    return modified_html


if __name__ == '__main__':
    print("starting main")

    # TODO(ethan): set the round number!
    # global ROUND_NUMBER
    # notice that `round_number` (lowercase) is not being used and should be deleted
    ROUND_NUMBER = 6

    # also notice that we only use one config type now!
    CONFIG_TYPE = "binary"

    app.run(debug=False, threaded=True, host="0.0.0.0", port=8090)
