"""Code to compute the clusters, but not to do the adaptive thresholding.

"""
import sys
sys.path.insert(0, "..")
from src.utils import (
    load_from_json,
    write_to_json,
    get_class_name_to_class_idx_mapping,
    make_dir,
    get_project_root
)
import fastcluster
import numpy as np
from collections import defaultdict
import random
from multiprocessing import Pool
import glob
import pprint
import argparse
import pickle
import copy
from scipy.spatial.distance import pdist
import os
from src.hit_maker import HITMaker
from src.places_dataset import PlacesDataset
from src.places import PlacesFolder
from src.coco_dataset import COCODataset
from typing import List
from os.path import join as pjoin
import math
from tqdm import tqdm
import itertools
import logging

from detectron.datasets import (
    ICCV_PLACES_CLASS_INDICES
)

from matplotlib import cm
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
# mycolormap = cm.get_cmap("jet")
mycolormap_colors_size = 256
mycolormap = cm.get_cmap("RdYlGn", mycolormap_colors_size)
mycolormap_colors = mycolormap(np.linspace(0, 1, mycolormap_colors_size))


def get_filename_qualities(cluster_folder_name, class_idx):
    return pjoin(cluster_folder_name.replace("merged/trees", "mturk/qualities"), "*", "{:03d}.json".format(class_idx))


class ClusterHierarchy(object):
    """Code to handle the whole cluster tree.
    Takes the output of Clusterer and will allow for browsing the cluster tree.
    """

    def __init__(
        self,
        # cluster_folder_name="/data/vision/torralba/scratch/ethanweber/scaleade/places/places_round_00/merged/trees/"
        cluster_folder_name="/data/vision/torralba/scratch/ethanweber/scaleade/rounds/iccv/places_ade_1k_COCO/round_00/merged/trees"
    ):
        self.cluster_folder_name = cluster_folder_name
        self.pool_directory = cluster_folder_name.replace("merged/trees", "mturk/pool")
        self.responses_directory = cluster_folder_name.replace("merged/trees", "mturk/responses")
        self.qualities_directory = cluster_folder_name.replace("merged/trees", "mturk/qualities")

        # handy mapping of class name to class index
        self.class_name_to_class_idx = get_class_name_to_class_idx_mapping()
        self.class_idx_to_class_name = {v: k for k, v in self.class_name_to_class_idx.items()}

        # the annotation data (what we have mturk information on)
        self.class_idx_to_mturk_data = {}

        # store the qualities data
        self.class_idx_to_qualities_data = {}

        # this will store the loaded cluster data, from a filename
        self.class_idx_to_cluster_data = {}

        # this will store the tree of the cluster
        # class_idx_to_cluster_tree[class_idx]["children"][cluster_id]
        # class_idx_to_cluster_tree[class_idx]["parent"][cluster_id]
        self.class_idx_to_cluster_tree = {}

        self.class_idx_segment_ids = {}

        # cache purity responses
        self.cached_get_purity_from_class_idx_and_cluster_id = {}

        # cache segments
        self.cached_segment_ids_from_class_idx_and_cluster_id = {}

        # cache values
        self.cached_miou_from_class_idx_and_cluster_id = {}
        self.cached_score_from_class_idx_and_cluster_id = {}
        self.cached_size_from_class_idx_and_cluster_id = {}
        self.cached_prior_from_class_idx_and_cluster_id = {}
        # for the actual data (numpy arrays)
        self.miou = {}
        self.score = {}
        self.size = {}

    def get_loaded_cluster_tree_for_class_idx(self, class_idx):
        """Return the cluster tree.
        # class_idx_to_cluster_tree[class_idx]["children"][cluster_id] -> [2d list]
        # class_idx_to_cluster_tree[class_idx]["parent"][cluster_id] -> a single cluster id
        """
        filename = os.path.join(self.cluster_folder_name, "{:03d}.npy".format(class_idx))
        Z = np.load(filename)

        cluster_tree = {}
        cluster_tree["children"] = {}
        cluster_tree["parent"] = {}
        num_leaves = Z.shape[0]
        for idx in range(num_leaves):
            cluster_id = num_leaves + 1 + idx
            cluster_tree["children"][idx] = []  # no children if leaf
            # https://docs.scipy.org/doc/scipy/reference/generated/scipy.cluster.hierarchy.linkage.html
            cluster_tree["children"][cluster_id] = [int(Z[idx][0]), int(Z[idx][1])]
            for child_id in cluster_tree["children"][cluster_id]:
                cluster_tree["parent"][child_id] = cluster_id
        return cluster_tree

    def get_cluster_tree_from_class_idx(self, class_idx):
        if class_idx not in self.class_idx_to_cluster_tree:
            self.class_idx_to_cluster_tree[class_idx] = self.get_loaded_cluster_tree_for_class_idx(class_idx)
        return self.class_idx_to_cluster_tree[class_idx]

    def get_loaded_segment_ids_for_class_idx(self,
                                             class_idx):
        if class_idx not in self.class_idx_segment_ids:
            # load from the file
            prefix = self.cluster_folder_name.replace("merged/trees", "all/segment_ids")
            self.class_idx_segment_ids[class_idx] = load_from_json(pjoin(prefix, "{:03d}.json".format(class_idx)))
            # "/data/vision/torralba/scratch/ethanweber/scaleade/places/places_round_00/all/segment_ids/{:03d}.json".format(class_idx))
        return self.class_idx_segment_ids[class_idx]

    def get_segment_ids_from_class_idx_and_cluster_id(self,
                                                      class_idx: int,
                                                      cluster_id: int):
        """Recurse to leaves and return the segment ids.
        """
        key = (class_idx, cluster_id)
        if key in self.cached_segment_ids_from_class_idx_and_cluster_id:
            return self.cached_segment_ids_from_class_idx_and_cluster_id[key]
        children = self.get_children_from_class_idx_and_cluster_id(class_idx, cluster_id)
        if len(children) == 0:
            segment_ids = self.get_loaded_segment_ids_for_class_idx(class_idx)[cluster_id]
        else:
            segment_ids = self.get_segment_ids_from_class_idx_and_cluster_id(class_idx, children[0]) \
                + self.get_segment_ids_from_class_idx_and_cluster_id(class_idx, children[1])
        self.cached_segment_ids_from_class_idx_and_cluster_id[key] = segment_ids
        return segment_ids

    def get_miou_from_class_idx_and_cluster_id(self,
                                               class_idx: int,
                                               cluster_id: int):
        """Returns miou. NOTE(ethan): this is actually the predicted IOU! Note the mIoU.
        """
        if class_idx not in self.miou:
            self.miou[class_idx] = np.load(os.path.join(
                self.cluster_folder_name.replace("trees", "ious"), "{:03d}.npy".format(class_idx)))

        key = (class_idx, cluster_id)
        if key in self.cached_miou_from_class_idx_and_cluster_id:
            return self.cached_miou_from_class_idx_and_cluster_id[key]
        children = self.get_children_from_class_idx_and_cluster_id(class_idx, cluster_id)
        if len(children) == 0:
            miou = self.miou[class_idx][cluster_id]
        else:
            size_0 = self.get_size_from_class_idx_and_cluster_id(class_idx, children[0])
            size_1 = self.get_size_from_class_idx_and_cluster_id(class_idx, children[1])
            miou_0 = self.get_miou_from_class_idx_and_cluster_id(class_idx, children[0])
            miou_1 = self.get_miou_from_class_idx_and_cluster_id(class_idx, children[1])
            miou = ((size_0 * miou_0) + (size_1 * miou_1)) / (size_0 + size_1)
        self.cached_miou_from_class_idx_and_cluster_id[key] = miou
        return miou

    def get_mask_score_from_class_idx_and_cluster_id(self,
                                                     class_idx: int,
                                                     cluster_id: int):
        """Returns the mask score.
        """
        if class_idx not in self.score:
            self.score[class_idx] = np.load(os.path.join(self.cluster_folder_name.replace(
                "trees", "scores"), "{:03d}.npy".format(class_idx)))

        key = (class_idx, cluster_id)
        if key in self.cached_score_from_class_idx_and_cluster_id:
            return self.cached_score_from_class_idx_and_cluster_id[key]
        children = self.get_children_from_class_idx_and_cluster_id(class_idx, cluster_id)
        if len(children) == 0:
            score = self.score[class_idx][cluster_id]
        else:
            size_0 = self.get_size_from_class_idx_and_cluster_id(class_idx, children[0])
            size_1 = self.get_size_from_class_idx_and_cluster_id(class_idx, children[1])
            score_0 = self.get_mask_score_from_class_idx_and_cluster_id(class_idx, children[0])
            score_1 = self.get_mask_score_from_class_idx_and_cluster_id(class_idx, children[1])
            score = ((size_0 * score_0) + (size_1 * score_1)) / (size_0 + size_1)
        self.cached_score_from_class_idx_and_cluster_id[key] = score
        return score

    def get_quality_score_from_class_idx_and_cluster_id(self,
                                                        class_idx: int,
                                                        cluster_id: int):
        """Returns the quality score s.
        """

        key = (class_idx, cluster_id)
        if key in self.cached_prior_from_class_idx_and_cluster_id:
            return self.cached_prior_from_class_idx_and_cluster_id[key]
        children = self.get_children_from_class_idx_and_cluster_id(class_idx, cluster_id)
        if len(children) == 0:
            # iou * score
            prior = self.get_miou_from_class_idx_and_cluster_id(
                class_idx, cluster_id) * self.get_mask_score_from_class_idx_and_cluster_id(class_idx, cluster_id)
        else:
            size_0 = self.get_size_from_class_idx_and_cluster_id(class_idx, children[0])
            size_1 = self.get_size_from_class_idx_and_cluster_id(class_idx, children[1])
            prior_0 = self.get_quality_score_from_class_idx_and_cluster_id(class_idx, children[0])
            prior_1 = self.get_quality_score_from_class_idx_and_cluster_id(class_idx, children[1])
            prior = ((size_0 * prior_0) + (size_1 * prior_1)) / (size_0 + size_1)
        self.cached_prior_from_class_idx_and_cluster_id[key] = prior
        return prior

    def get_size_from_class_idx_and_cluster_id(self,
                                               class_idx: int,
                                               cluster_id: int):
        """Returns size (number elements ).
        """
        if class_idx not in self.size:
            self.size[class_idx] = np.load(os.path.join(self.cluster_folder_name.replace(
                "trees", "leaf_size"), "{:03d}.npy".format(class_idx)))

        key = (class_idx, cluster_id)
        if key in self.cached_size_from_class_idx_and_cluster_id:
            return self.cached_size_from_class_idx_and_cluster_id[key]
        children = self.get_children_from_class_idx_and_cluster_id(class_idx, cluster_id)
        if len(children) == 0:
            size = self.size[class_idx][cluster_id]
        else:
            size_0 = self.get_size_from_class_idx_and_cluster_id(class_idx, children[0])
            size_1 = self.get_size_from_class_idx_and_cluster_id(class_idx, children[1])
            size = size_0 + size_1
        self.cached_size_from_class_idx_and_cluster_id[key] = size
        return size

    def get_loaded_mturk_data_for_class_idx(self, class_idx, round_number=None):
        """Load the mturk data for a given class_idx and return the dictionary.
        """
        # choose the last filename
        pool_filenames = sorted(glob.glob(pjoin(self.pool_directory, "*", "{:03d}.json".format(class_idx))))
        filename = pool_filenames[round_number]
        mturk_data = load_from_json(filename)
        return mturk_data

    def get_mturk_data_from_class_idx(self, class_idx, round_number=None):
        if class_idx not in self.class_idx_to_mturk_data:
            self.class_idx_to_mturk_data[class_idx] = {}
        if round_number not in self.class_idx_to_mturk_data[class_idx]:
            self.class_idx_to_mturk_data[class_idx][round_number] = self.get_loaded_mturk_data_for_class_idx(class_idx,
                                                                                                             round_number=round_number)
        return self.class_idx_to_mturk_data[class_idx][round_number]

    def get_mturk_data_from_class_idx_and_segment_id(self,
                                                     class_idx: int,
                                                     segment_id: int,
                                                     round_number=None):
        mturk_data = self.get_mturk_data_from_class_idx(class_idx, round_number=round_number)
        try:
            return mturk_data[str(segment_id)]
        except:
            return {}

    def get_loaded_qualities_data_for_class_idx(self, class_idx, round_number=None):
        """Load the qualities for a given class_idx and return the dictionary.
        """
        # choose the last filename
        qualities_filenames = sorted(glob.glob(pjoin(self.qualities_directory, "*", "{:03d}.json".format(class_idx))))
        filename = qualities_filenames[round_number]
        qualities_data = load_from_json(filename)
        return qualities_data

    def get_qualities_data_from_class_idx(self, class_idx, round_number=None):
        if class_idx not in self.class_idx_to_qualities_data:
            self.class_idx_to_qualities_data[class_idx] = {}
        if round_number not in self.class_idx_to_qualities_data[class_idx]:
            self.class_idx_to_qualities_data[class_idx][round_number] = \
                self.get_loaded_qualities_data_for_class_idx(class_idx, round_number=round_number)
        return self.class_idx_to_qualities_data[class_idx][round_number]

    def get_mturk_data_information_list_from_class_idx_and_cluster_id(self,
                                                                      class_idx: int,
                                                                      cluster_id: int,
                                                                      round_number=None):
        """Returns the mturk info for a (class_idx, cluster_id).
        """

        mturk_data_information_list = []
        segment_ids = self.get_segment_ids_from_class_idx_and_cluster_id(class_idx, cluster_id)
        for segment_id in segment_ids:
            # get the data for all the clusters
            mturk_data = self.get_mturk_data_from_class_idx_and_segment_id(class_idx,
                                                                           segment_id,
                                                                           round_number=round_number)
            if len(mturk_data) != 0:  # if we have information, append the segment id
                mturk_data_information_list.append(segment_id)
        return mturk_data_information_list

    def get_num_mturk_questions_asked(self, class_indices, round_number=None):
        """Returns the total number of MTurk questions asked.
        """
        count = 0
        for class_idx in tqdm(class_indices):
            cluster_id = self.get_root_cluster_id_from_class_idx(class_idx)
            count += len(self.get_mturk_data_from_class_idx(class_idx, round_number=round_number))
        return count

    def get_num_clusters_statistics(self, class_indices, round_number=None):
        # all clusters
        # TODO(ethan): decide on better thresholds to use for this call
        round_data = self.get_all_class_puritites_in_threshold(
            class_indices, round_number=round_number, min_purity=-1, max_purity=-1)
        num_total_clusters = sum(len(x) for x in round_data.values())
        # accepted clusters
        round_data = self.get_all_class_puritites_in_threshold(
            class_indices, round_number=round_number, min_purity=0.85)
        num_accepted_clusters = sum(len(x) for x in round_data.values())
        return num_accepted_clusters, num_total_clusters

    def get_purity_from_class_idx_and_cluster_id_recursive(self,
                                                           class_idx: int,
                                                           cluster_id: int,
                                                           round_number=None):
        """Calculates and returns the purity of the cluster.
        This score is comuted with MTurk data.
        Args:
            class_idx (int): The class index.
            cluster_id (int): The cluster id, for the specified class.
        Returns:
            float: The purity.
        """

        key = (class_idx, cluster_id, round_number)
        if key in self.cached_get_purity_from_class_idx_and_cluster_id:
            return self.cached_get_purity_from_class_idx_and_cluster_id[key]

        # TODO(ethan): speed this up!
        segment_ids = self.get_segment_ids_from_class_idx_and_cluster_id(class_idx, cluster_id)
        freq_dict = defaultdict(int)
        for segment_id in segment_ids:
            mturk_data = self.get_mturk_data_from_class_idx_and_segment_id(class_idx,
                                                                           segment_id,
                                                                           round_number=round_number)
            if bool(mturk_data):  # if it has data
                if mturk_data["binary"] not in [-1, 1]:
                    print(mturk_data["binary"])
                    print(type(mturk_data["binary"]))
                    raise ValueError("your labels aren't in the right range")
                freq_dict[mturk_data["binary"]] += 1
        total_with_labels = freq_dict[1] + freq_dict[-1]  # yes + no labels
        purity = freq_dict[1] / total_with_labels if total_with_labels != 0 else None
        response = (purity, total_with_labels)

        # cache responses
        self.cached_get_purity_from_class_idx_and_cluster_id[key] = response
        return response

    def get_purity_from_class_idx_and_cluster_id(self,
                                                 class_idx: int,
                                                 cluster_id: int,
                                                 round_number=None,
                                                 min_num_samples=15):
        purity, total_with_labels = self.get_purity_from_class_idx_and_cluster_id_recursive(class_idx,
                                                                                            cluster_id,
                                                                                            round_number=round_number)

        # TODO(ethan): set `min_num_samples` in a cleaner way
        if total_with_labels < min_num_samples:
            response = (None, total_with_labels)
        else:
            response = (purity, total_with_labels)
        return response

    def get_purity_from_class_idx_and_cluster_id(self,
                                                 class_idx: int,
                                                 cluster_id: int,
                                                 round_number=None,
                                                 min_num_samples=15):
        purity, total_with_labels = self.get_purity_from_class_idx_and_cluster_id_recursive(class_idx,
                                                                                            cluster_id,
                                                                                            round_number=round_number)

        # TODO(ethan): set `min_num_samples` in a cleaner way
        if total_with_labels < min_num_samples:
            response = (None, total_with_labels)
        else:
            response = (purity, total_with_labels)
        return response

    def get_quality_from_class_idx_and_cluster_id(self,
                                                  class_idx: int,
                                                  cluster_id: int,
                                                  round_number=None):
        """Return the quality for the cluster. None if no estimation.
        """
        assert round_number is not None
        qualities_data = self.get_qualities_data_from_class_idx(class_idx, round_number=round_number)
        if str(cluster_id) in qualities_data:
            return qualities_data[str(cluster_id)]
        else:
            return None

    def get_children_from_class_idx_and_cluster_id(self,
                                                   class_idx: int,
                                                   cluster_id: int):
        """Return the children ids."""
        try:
            cluster_tree = self.get_cluster_tree_from_class_idx(class_idx)
            children = cluster_tree["children"][cluster_id]
            return children
        except:
            return []

    def get_parent_from_class_idx_and_cluster_id(self,
                                                 class_idx: int,
                                                 cluster_id: int):
        """Return the parent id."""
        try:
            cluster_tree = self.get_cluster_tree_from_class_idx(class_idx)
            children = cluster_tree["parent"][cluster_id]
            return children
        except:
            return None

    def get_root_cluster_id_from_class_idx(self, class_idx: int):
        """Return the root id."""
        cluster_id = 0
        parent_id = self.get_parent_from_class_idx_and_cluster_id(class_idx, cluster_id)
        while parent_id:
            cluster_id = parent_id
            parent_id = self.get_parent_from_class_idx_and_cluster_id(class_idx, cluster_id)
        return cluster_id

    def get_segment_ids_used_for_quality_estimate(self, class_idx, cluster_id, latest_round_number=None):
        assert latest_round_number is not None
        round_numbers = list(range(0, latest_round_number + 1))
        for round_number in round_numbers:
            purity, total_with_labels = self.get_purity_from_class_idx_and_cluster_id(
                class_idx, cluster_id, round_number=round_number)
            if total_with_labels >= 15:
                break
        if total_with_labels < 15:
            raise ValueError("This shouldn't happen!")

        print("round_number: ", round_number)

        # now get the segment ids with labels
        segment_ids = self.get_segment_ids_from_class_idx_and_cluster_id(class_idx, cluster_id)
        random.shuffle(segment_ids)
        used_segment_ids = []
        used_responses = []
        for segment_id in segment_ids:
            mturk_data = self.get_mturk_data_from_class_idx_and_segment_id(class_idx,
                                                                           segment_id,
                                                                           round_number=round_number)
            if bool(mturk_data):  # if it has data
                if mturk_data["binary"] not in [-1, 1]:
                    print(mturk_data["binary"])
                    print(type(mturk_data["binary"]))
                    raise ValueError("your labels aren't in the right range")

                # append the data
                used_segment_ids.append(segment_id)
                used_responses.append(mturk_data["binary"])
        return used_segment_ids[:15], used_responses[:15], purity

    def get_tree_from_class_idx_and_cluster_id(self,
                                               class_idx: int,
                                               cluster_id: int,
                                               K_a=0.85,
                                               K_r=0.15,
                                               is_pruned=False,
                                               round_number=None):
        """Returns the tree, rooted, at these elements. This is used by the frontend server.
        """
        children = self.get_children_from_class_idx_and_cluster_id(class_idx, cluster_id)
        num_segments = len(self.get_segment_ids_from_class_idx_and_cluster_id(class_idx, cluster_id))

        data = {}
        data["cluster_id"] = cluster_id
        data["num_segments"] = num_segments

        def get_interpolated_color(start, end, factor):
            color = ((end - start) * factor) + start
            return color

        # TODO(ethan): change "purity" to quality_est
        # also, check if "purity" is used by the website and change to "quality_est"
        # purity = None
        quality_est = self.get_quality_from_class_idx_and_cluster_id(class_idx,
                                                                     cluster_id,
                                                                     round_number=round_number)

        edge_color = "rgba(0,0,0,1)"

        if quality_est and quality_est != -1 and not is_pruned:
            color = list(
                (255.00 * np.array(mycolormap_colors[int((mycolormap_colors_size - 1.0) * quality_est)])).astype("int"))[:3]
            alpha = 1.0
            stroke = "black"
        else:
            color = list(np.array([0, 0, 0]))
            alpha = 0.0
            stroke = "none"
            edge_color = "rgba(0,0,0,0.5)"

        # overwrite if needed
        if is_pruned:
            edge_color = "#fff"
            stroke = "none"

        data["purity"] = quality_est
        data["color_string"] = "rgba({},{},{},{})".format(color[0], color[1], color[2], alpha)
        data["stroke"] = stroke
        data["children"] = []

        set_is_pruned = is_pruned
        if quality_est and (quality_est != -1) and (quality_est >= K_a or quality_est < K_r):
            # means prune, since accepted or rejected
            # print(quality_est)
            set_is_pruned = True

        for child_id in children:
            data["children"].append(self.get_tree_from_class_idx_and_cluster_id(
                class_idx, child_id, is_pruned=set_is_pruned, round_number=round_number))

        data["edge_color"] = edge_color
        return data

    def get_segment_ids_to_label_from_class_idx_and_cluster_id(self,
                                                               class_idx,
                                                               cluster_id,
                                                               round_number=None,
                                                               num_segment_ids_needed=None,
                                                               num_questions_per_cluster=15):
        """Returns a list of segment ids to label, from a list of cluster ids. [cluster_id1, ...]
        """

        segment_ids_to_label = []
        segment_ids = self.get_segment_ids_from_class_idx_and_cluster_id(class_idx, cluster_id)
        assert isinstance(segment_ids, list)
        assert isinstance(segment_ids[0], int)

        # get mturk data already labeled
        mturk_data_information_list = self.get_mturk_data_information_list_from_class_idx_and_cluster_id(class_idx,
                                                                                                         cluster_id,
                                                                                                         round_number=round_number)
        mturk_data_information_set = set(mturk_data_information_list)
        if len(mturk_data_information_set) >= num_questions_per_cluster:
            # TODO: actually add a check here!
            print("{} should be less than {}".format(len(mturk_data_information_set), num_questions_per_cluster))
            print(cluster_id)
            print()
            raise ValueError("Fix this before continuing! Exiting.")

        # let's sample from from data points where we don't have information
        possible_ids_to_use = list(set(segment_ids) - mturk_data_information_set)
        segment_ids_to_label = random.sample(possible_ids_to_use, min(num_segment_ids_needed, len(possible_ids_to_use)))
        return segment_ids_to_label

    def get_segment_ids_to_label_from_class_idx_and_round_number(self,
                                                                 class_idx: int,
                                                                 round_number: int,
                                                                 #  K_prior_1=0.2,
                                                                 K_prior_1=0.0,
                                                                 K_prior_2=0.7,
                                                                 K_a=0.85,
                                                                 # K_r=0.15, # TODO(ethan): update this in paper or change back!
                                                                 # K_r=0.00,
                                                                 num_questions_per_cluster=15,
                                                                 min_num_samples=15,
                                                                 max_num_clusters=30):  # was 100 for ICCV
        """Start at the root cluster id and return the segments that need to be labeled.
        Returns {
            cluster_id: [ ..., segment_id_i, ... ]
        }
        """

        # TODO(ethan): implement this!!!

        assert round_number is not None

        root_cluster_id = self.get_root_cluster_id_from_class_idx(class_idx)
        clusters = [root_cluster_id]  # the root node
        cluster_ids_to_segment_ids = {}  # this is what we'll be labeling. keyed by cluster id, to know how many clusters are being labeled

        while clusters:
            cluster_id = clusters.pop(0)
            # print(cluster_id)

            segment_ids = self.get_segment_ids_from_class_idx_and_cluster_id(class_idx, cluster_id)

            if len(segment_ids) < min_num_samples:  # terminate if too few
                continue

            quality_est = self.get_quality_from_class_idx_and_cluster_id(class_idx,
                                                                         cluster_id,
                                                                         round_number=round_number)

            # this means we have information below, so split
            if quality_est == -1:
                clusters += self.get_children_from_class_idx_and_cluster_id(class_idx, cluster_id)
                continue

            if quality_est is not None:  # means it's already done!
                if quality_est >= K_a:  # accept
                    continue
                # otherwise mixed, so split
                clusters += self.get_children_from_class_idx_and_cluster_id(class_idx, cluster_id)
                continue

            else:  # we don't know the quality, so label it!

                quality_score = self.get_quality_score_from_class_idx_and_cluster_id(class_idx,
                                                                                     cluster_id)
                if K_prior_1 and quality_score < K_prior_1:  # reject
                    continue
                if K_prior_2 and len(segment_ids) > num_questions_per_cluster and quality_score < K_prior_2:  # split!
                    clusters += self.get_children_from_class_idx_and_cluster_id(class_idx, cluster_id)
                    continue

                # otherwise we should label it!
                # figure out how many questions we need to ask
                # print("cluster_id:", cluster_id)
                segment_ids_to_label = self.get_segment_ids_to_label_from_class_idx_and_cluster_id(class_idx,
                                                                                                   cluster_id,
                                                                                                   round_number=round_number,
                                                                                                   num_segment_ids_needed=num_questions_per_cluster)

                if len(segment_ids_to_label) == 0:  # nothing left to label here
                    continue

                cluster_ids_to_segment_ids[cluster_id] = segment_ids_to_label
                continue

        # do the filter for good quality clusters
        filtered_cluster_ids_to_segment_ids = {}
        cluster_ids_list = list(cluster_ids_to_segment_ids.keys())
        items = []
        for cluster_id in cluster_ids_list:
            quality_score = self.get_quality_score_from_class_idx_and_cluster_id(class_idx, cluster_id)
            items.append((cluster_id, quality_score))
        items = sorted(items, key=lambda x: x[1], reverse=True)
        cluster_ids_list = [x[0] for x in items]
        # filter
        cluster_ids_list = cluster_ids_list[:min(len(cluster_ids_list), max_num_clusters)]
        for cluster_id in cluster_ids_list:
            filtered_cluster_ids_to_segment_ids[cluster_id] = cluster_ids_to_segment_ids[cluster_id]
        return filtered_cluster_ids_to_segment_ids

    def get_segment_ids_rounded_up_to_nearest_value(self,
                                                    segment_ids,
                                                    class_idx,
                                                    round_number):
        new_segment_ids = [x for x in segment_ids]
        num_segment_ids = ((len(segment_ids) // 50) + 1) * 50
        num_to_add = num_segment_ids - len(new_segment_ids)

        root_cluster_id = self.get_root_cluster_id_from_class_idx(class_idx)
        all_segment_ids = self.get_segment_ids_from_class_idx_and_cluster_id(class_idx, root_cluster_id)

        mturk_data = self.get_mturk_data_from_class_idx(class_idx,
                                                        round_number=round_number)  # keys are the segment ids where we have info
        # now add this number of points
        set_already_used = set(list(mturk_data.keys()))
        set_already_used.update(set(segment_ids))

        items_to_sample_from = list(set(all_segment_ids) - set_already_used)
        new_segment_ids += random.sample(items_to_sample_from, min(num_to_add, len(items_to_sample_from)))
        return new_segment_ids

    def get_hits_for_class_idx_conditioned_on_round_number(self,
                                                           class_idx=None,
                                                           round_number=None):
        # hits_directory = "/data/vision/torralba/scratch/ethanweber/scaleade/annotate/hits/binary"
        hits_directory = self.cluster_folder_name.replace("merged/trees", "mturk/hits")
        return glob.glob(os.path.join(hits_directory,
                                      "{:03d}".format(round_number),
                                      "binary_places_{:03d}_{:03d}_*.json".format(int(class_idx), int(round_number))))

    def get_hits_for_class_indices_conditioned_on_round_number(self,
                                                               class_indices=None,
                                                               round_number=None):
        hits = []
        for class_idx in class_indices:
            hits += self.get_hits_for_class_idx_conditioned_on_round_number(class_idx, round_number)
        return hits

    def get_hit_names_for_class_indices_conditioned_on_round_number(self,
                                                                    class_indices=None,
                                                                    round_number=None):
        hits = self.get_hits_for_class_indices_conditioned_on_round_number(class_indices=class_indices,
                                                                           round_number=round_number)
        hit_names = []
        for hit in hits:
            hit_names.append(os.path.basename(hit)[:-5])
        return hit_names

    def make_hits_for_class_idx_conditioned_on_round_number(self,
                                                            class_idx=None,
                                                            round_number=None,
                                                            safety_check=True,
                                                            get_cluster_ids_only=False):
        assert class_idx is not None
        assert round_number is not None

        # make sure we haven't done this before
        if safety_check:
            assert len(self.get_hits_for_class_idx_conditioned_on_round_number(class_idx=class_idx,
                                                                               round_number=round_number)) == 0

        def divide_chunks(l, n):
            # looping till length l
            newlist = []
            for i in range(0, len(l), n):
                newlist.append(l[i:i + n])
            return newlist

        cluster_ids_to_segment_ids = self.get_segment_ids_to_label_from_class_idx_and_round_number(class_idx,
                                                                                                   round_number)
        print("Number of clusters:", len(cluster_ids_to_segment_ids.keys()))

        cluster_ids = []
        segment_ids = []
        for key, val in cluster_ids_to_segment_ids.items():
            cluster_ids.append(key)
            segment_ids += val

        if get_cluster_ids_only:
            return cluster_ids

        # l = self.get_segment_ids_rounded_up_to_nearest_value(segment_ids, class_idx, round_number)
        # TODO(ethan): decide what the above was used for!
        l = segment_ids
        random.shuffle(l)
        segment_chunks = divide_chunks(l, 50)

        print("num segments:", len(segment_ids))
        print("num hits:", len(segment_chunks))
        if len(segment_ids) == 0:
            print("Skipping this HITs for class_idx", class_idx)
            return

        # load the hit maker to make these hits
        maker = HITMaker(class_idx)
        for idx, chunk_segment_ids in tqdm(enumerate(segment_chunks)):
            hit = maker.get_binary_hit(chunk_segment_ids)
            filename = os.path.join(
                # "/data/vision/torralba/scratch/ethanweber/scaleade/annotate/hits/binary",
                self.cluster_folder_name.replace("merged/trees", "mturk/hits"),
                "{:03d}".format(round_number),
                "binary_places_{:03d}_{:03d}_{:06d}.json".format(
                    class_idx,
                    round_number,
                    idx
                )
            )
            # the cluster ids used in this round
            assert "CLUSTER_IDS_IN_ROUND" not in hit
            hit["CLUSTER_IDS_IN_ROUND"] = cluster_ids
            make_dir(filename)
            write_to_json(
                filename,
                hit
            )

        # print(len(cluster_ids_to_segment_ids))

    def update_label_pool_with_responses_conditioned_class_idx_and_round_number(self,
                                                                                class_idx=None,
                                                                                round_number=None,
                                                                                min_seconds=35):
        """
        TODO(ethan): move the min_seconds param elsewhere!
        """
        assert class_idx is not None
        assert round_number is not None

        # load current pool
        pool_filenames_path = os.path.join(
            self.pool_directory,
            "{:03d}".format(round_number),
            "{:03d}.json".format(class_idx))
        print()
        pool_filenames = sorted(glob.glob(pool_filenames_path))
        assert len(pool_filenames) == 1
        pool = load_from_json(pool_filenames[0])
        pool_original = pool.copy()
        print(len(pool))

        # get the relevant configs to load from, to update the pool with
        response_filenames = sorted(
            glob.glob(pjoin(
                self.responses_directory,
                "{:03d}".format(round_number),
                "binary_places_{:03d}_{:03d}_*.json".format(class_idx, round_number))))

        num_duplicates = 0
        num_added = 0
        for response_filename in response_filenames:
            assert "_{:03d}_".format(int(class_idx)) in os.path.basename(response_filename)

            responses = load_from_json(response_filename)

            hit_time = responses["TEST_TIME"]
            if min_seconds and hit_time < min_seconds:
                print("Skipping {} with time {}".format(response_filename, hit_time))
                continue

            # use this to update the pool of information
            for example in responses["QUERY_EXAMPLES"]:
                annotation_id = example["annotation_id"]
                assert example["class_idx"] == str(class_idx)
                assert example["dataset"] == "places"
                response = int(example["response"])
                if response == 0:
                    response = -1

                # make sure we aren't overwriting anything
                if str(annotation_id) in pool_original:
                    num_duplicates += 1
                    print("WHOOPS: {} was already annotated!".format(str(annotation_id)))
                num_added += 1
                pool[str(annotation_id)] = {"binary": response}

        if len(response_filenames) == 0:
            print("no response filenames for class idx: {:03d}".format(class_idx))
            # copy over the pool to the next found (since nothing changed), and return
        else:
            pass
            # assert num_duplicategets != num_added, "yo, you already updated this."

        # write to a new pool, for a new iteration of the algorithm
        next_round_number = round_number + 1
        directory = pjoin(self.pool_directory, "{:03d}".format(next_round_number))
        if not os.path.exists(directory):
            os.makedirs(directory)
        print(len(pool))
        new_pool_filename = os.path.join(directory, "{:03d}.json".format(class_idx))
        write_to_json(new_pool_filename, pool)
        if class_idx in self.class_idx_to_mturk_data:
            del self.class_idx_to_mturk_data[class_idx]  # refresh the cache

    def update_label_pool_with_responses_conditioned_class_indices_and_round_number(self,
                                                                                    class_indices=None,
                                                                                    round_number=None):
        assert class_indices is not None
        assert round_number is not None
        for class_idx in class_indices:
            self.update_label_pool_with_responses_conditioned_class_idx_and_round_number(class_idx=class_idx,
                                                                                         round_number=round_number)

    def update_qualities_conditioned_class_idx_and_round_number(self,
                                                                class_idx=None,
                                                                round_number=None):
        """Update the qualities for (round_number + 1) | round_number responses. Fix the quality as it's traversed.
        """
        assert class_idx is not None
        assert round_number is not None

        # NOTE(ethan): purities come from `next_round_number`!
        next_round_number = round_number + 1

        # dictionary we want to update with the current mturk pool!
        cluster_id_to_quality = self.get_qualities_data_from_class_idx(class_idx, round_number=round_number)
        cluster_id_to_quality = copy.deepcopy(cluster_id_to_quality)

        # get the cluster ids that we checked this round
        filenames = self.get_hits_for_class_idx_conditioned_on_round_number(class_idx=class_idx,
                                                                            round_number=round_number)
        hit_datas = [load_from_json(x) for x in filenames]
        cluster_ids_checked_during_round = []
        for hit_data in hit_datas:
            cluster_ids_checked_during_round += hit_data["CLUSTER_IDS_IN_ROUND"]

        # filter just in case
        cluster_ids_checked_during_round = sorted(list(set(cluster_ids_checked_during_round)))
        # if len(cluster_ids_checked_during_round) != 0:
        #     print("We should know which cluster ids were checked!")
        #     print("Assuming none were checked.")
        #     return None
        # print(cluster_ids_checked_during_round)

        # keep track of number of updates
        num_updates = 0

        # simple tree traversal to update the quality scores where possible
        # notice that this starts with the cluster_ids that we labeled
        clusters = cluster_ids_checked_during_round
        while clusters:
            cluster_id = clusters.pop(0)
            if str(cluster_id) in cluster_id_to_quality:
                print(f"cluster_id {cluster_id} already labeled, so splitting.")
                clusters += self.get_children_from_class_idx_and_cluster_id(class_idx, cluster_id)
                continue

            # try to assign a quality score, if possible
            quality_est, _ = self.get_purity_from_class_idx_and_cluster_id(
                class_idx, cluster_id, round_number=next_round_number)

            if quality_est is not None:
                # means we have enough samples to set the quality
                cluster_id_to_quality[str(cluster_id)] = quality_est
                num_updates += 1

                # go through the parents and set to -1 (skipped) if not already having a value!
                parent_id = self.get_parent_from_class_idx_and_cluster_id(class_idx, cluster_id)
                while parent_id:
                    # set to -1 is no value
                    if str(parent_id) not in cluster_id_to_quality:
                        cluster_id_to_quality[str(parent_id)] = -1
                    parent_id = self.get_parent_from_class_idx_and_cluster_id(class_idx, parent_id)

            # always split in case there are more to label due to over sampling
            clusters += self.get_children_from_class_idx_and_cluster_id(class_idx, cluster_id)

        # save the results
        filename = pjoin(self.qualities_directory, "{:03d}".format(next_round_number), "{:03d}.json".format(class_idx))
        make_dir(filename)
        write_to_json(filename, cluster_id_to_quality)
        if class_idx in self.class_idx_to_qualities_data:
            del self.class_idx_to_qualities_data[class_idx]  # refresh the cache

    def update_qualities_conditioned_class_indices_and_round_number(self,
                                                                    class_indices=None,
                                                                    round_number=None):
        assert class_indices is not None
        assert round_number is not None
        for class_idx in tqdm(class_indices):
            self.update_qualities_conditioned_class_idx_and_round_number(class_idx=class_idx,
                                                                         round_number=round_number)

    def get_cluster_ids_from_root_in_purity_range(self,
                                                  class_idx: int,
                                                  round_number=None,
                                                  min_purity=0.9,
                                                  max_purity=1.0):
        """Start at the root cluster id (9999) and return the cluster ids and purity.
        """
        assert round_number is not None

        root_cluster_id = self.get_root_cluster_id_from_class_idx(class_idx)
        clusters = [root_cluster_id]  # the root node
        cluster_id_to_quality_est = {}  # this is what we'll be labeling. keyed by cluster id, to know how many clusters are being labeled
        while clusters:
            cluster_id = clusters.pop(0)
            quality_est = self.get_quality_from_class_idx_and_cluster_id(class_idx,
                                                                         cluster_id,
                                                                         round_number=round_number)
            if quality_est is None:
                # means we haven't explored here, so terminate
                continue
            elif min_purity <= quality_est <= max_purity:
                cluster_id_to_quality_est[cluster_id] = quality_est
                continue

            # otherwise keep splitting
            clusters += self.get_children_from_class_idx_and_cluster_id(class_idx, cluster_id)

        return cluster_id_to_quality_est

    def call_get_cluster_ids_from_root_in_purity_range(self, arguments):
        class_idx, round_number, min_purity, max_purity = arguments
        return self.get_cluster_ids_from_root_in_purity_range(class_idx,
                                                              round_number=round_number,
                                                              min_purity=min_purity,
                                                              max_purity=max_purity)

    def get_all_class_puritites_in_threshold(self,
                                             class_indices,
                                             round_number=None,
                                             min_purity=0.9,
                                             max_purity=1.0,
                                             num_cores=1):
        rounds = {}
        from multiprocessing import Pool

        values = []
        for class_idx in class_indices:
            values.append((class_idx, round_number, min_purity, max_purity))

        if num_cores > 1:
            with Pool(num_cores) as p:
                results = p.map(self.call_get_cluster_ids_from_root_in_purity_range, values)
        else:
            results = []
            for value in tqdm(values):
                results.append(self.call_get_cluster_ids_from_root_in_purity_range(value))

        for idx, class_idx in enumerate(class_indices):
            rounds[class_idx] = results[idx]

        return rounds

    def get_urls_for_all_filtered_data(self, filtered_data):
        urls = []
        for class_idx in filtered_data:
            for cluster_id in filtered_data[class_idx]:
                urls.append("https://adeserver.ethanweber.me/tree"
                            "?dataset_name=places&class_idx={}&cluster_id={}".format(class_idx, cluster_id))

        return urls

    def get_count_from_round_data(self, round_data, round_number=None):
        """Get then number of items in something returned from obtaining the round data, which has class_idx -> cluster ids.
        """
        count = 0
        for class_idx, data in round_data.items():
            for cluster_id in data.keys():
                count += self.get_size_from_class_idx_and_cluster_id(class_idx, cluster_id)
        return count

    def get_segment_ids_from_round_data(self, round_data):
        class_idx_to_segment_ids = defaultdict(list)
        for class_idx in round_data:
            for cluster_id in round_data[class_idx]:
                class_idx_to_segment_ids[class_idx] += self.get_segment_ids_from_class_idx_and_cluster_id(
                    class_idx, cluster_id)
        # sort the indices
        for class_idx in round_data:
            class_idx_to_segment_ids[class_idx] = sorted(class_idx_to_segment_ids[class_idx])
        return class_idx_to_segment_ids

    def get_all_segment_ids(self):
        # TODO: ethan, maybe remove this. we did this for the CVPR 2021 rebuttal
        class_idx_to_segment_ids = defaultdict(list)
        for class_idx in round_data:
            root_cluster_id = self.get_root_cluster_id_from_class_idx(class_idx)
            class_idx_to_segment_ids[class_idx] = sorted(
                self.get_segment_ids_from_class_idx_and_cluster_id(class_idx, root_cluster_id))
        return class_idx_to_segment_ids

    def get_url_from_class_idx_and_cluster_id(self, class_idx, cluster_id):
        url = "https://adeserver.ethanweber.me/tree?dataset_name={}&class_idx={}&cluster_id={}".format("places",
                                                                                                       str(class_idx),
                                                                                                       str(cluster_id))
        return url

    def get_urls_and_quality_est_from_round_data(self,
                                                 round_data,
                                                 sort_by_quality_est=True,
                                                 num_to_sample=None):
        """Returns HTML to check quality of the obtained data!
        """
        items = []
        for class_idx in round_data.keys():
            for cluster_id in round_data[class_idx].keys():
                quality_est = round_data[class_idx][cluster_id]
                url = self.get_url_from_class_idx_and_cluster_id(class_idx, cluster_id)
                html_str = """<a href="{}">class_idx: {:03d}, cluster_id: {:09d}, quality_est: {:.3f}</a><br>""".format(
                    url, class_idx, cluster_id, quality_est)
                items.append((html_str, quality_est))
        if sort_by_quality_est:
            items = sorted(items, key=lambda x: x[1], reverse=True)
        if num_to_sample:
            items = random.sample(items, k=num_to_sample)
        html_str = "".join([x[0] for x in items])
        return html_str


parser = argparse.ArgumentParser(description="Process the Places data.")
parser.add_argument('--step', type=str)
parser.add_argument('--round_number', type=int, default=None)
parser.add_argument('--disable_safety_check', action='store_true')
parser.add_argument('--export_places_path',
                    type=str,
                    default=pjoin(get_project_root(), "rounds/iccv/places_ade_1k_COCO"))
parser.add_argument('--places_round', type=int, default=0)


def main():
    args = parser.parse_args()
    pprint.pprint(args)

    PLACES_ROUND = 0
    CLUSTER_FOLDER_NAME = "/data/vision/torralba/scratch/ethanweber/scaleade/rounds/iccv/places_ade_1k_COCO/round_{:02d}/merged/trees".format(
        PLACES_ROUND)
    DATABASE_FILENAME = CLUSTER_FOLDER_NAME.replace("merged/trees", "mturk_database.pkl")

    class_indices = ICCV_PLACES_CLASS_INDICES
    # class_indices = [35, 36]
    # class_indices = [9, 10]

    if args.step == "update_qualities":

        # update the qualities as we get more information
        round_numbers = list(range(4, args.round_number + 1))  # [0, 1, 2]
        # class_indices = ICCV_PLACES_CLASS_INDICES
        # class_indices = [35, 36]

        # class_indices = [60]
        # print(round_numbers)
        for class_idx in class_indices:
            print("Updating class_idx {}".format(class_idx))
            for round_number in tqdm(round_numbers):
                hier = ClusterHierarchy(cluster_folder_name=CLUSTER_FOLDER_NAME)
                # first update the mturk responses
                hier.update_label_pool_with_responses_conditioned_class_idx_and_round_number(class_idx,
                                                                                             round_number=round_number)
                # next update the qualities
                hier.update_qualities_conditioned_class_idx_and_round_number(class_idx=class_idx,
                                                                             round_number=round_number)

    if args.step == "make_hits":

        # for mturk
        # round_number = 1

        assert args.round_number is not None
        round_number = args.round_number

        hier = ClusterHierarchy(cluster_folder_name=CLUSTER_FOLDER_NAME)

        # class_indices = [31]
        # class_indices = ICCV_PLACES_CLASS_INDICES

        # class_indices = [60]
        for idx, class_idx in enumerate(class_indices):
            print(f"\n\nIdx {idx}: Making hit for {class_idx}")
            hier.make_hits_for_class_idx_conditioned_on_round_number(
                class_idx=class_idx,
                round_number=round_number,
                safety_check=not args.disable_safety_check
            )

    if args.step == "export_dataset":
        pass
        # TODO: change this
        place_indices = [i for i in range(0, 11)]

        hier = ClusterHierarchy(cluster_folder_name=CLUSTER_FOLDER_NAME)

        min_purity = 0.85

        print("Finding high quality clusters.")
        round_data = hier.get_all_class_puritites_in_threshold(
            class_indices,
            round_number=args.round_number,
            min_purity=min_purity,
            max_purity=1.0,
            num_cores=8
        )

        # print("Found clusters with high quality.")
        # pprint.pprint(round_data)

        print("Getting the segment ids.")
        class_idx_to_segment_ids = hier.get_segment_ids_from_round_data(round_data)
        # NOTE(ethan): some segment_ids will extend behind the representative leafs in MTurk!!!

        export_places_round_path = CLUSTER_FOLDER_NAME.replace("merged/trees", "")
        # prepare data we want to export
        arguments_list = []
        for class_idx in class_indices:

            # add the positively labeled instances too!
            # don't simply add the clusters but also the labels from mturk
            mturk_data = hier.get_mturk_data_from_class_idx(class_idx, round_number=args.round_number)
            list_of_all_segments_correct = []
            for segment_id, info in mturk_data.items():
                if int(info["binary"]) == 1:
                    list_of_all_segments_correct.append(int(segment_id))
            class_idx_to_segment_ids[class_idx] += list_of_all_segments_correct
            class_idx_to_segment_ids[class_idx] = sorted(list(set(class_idx_to_segment_ids[class_idx])))

            arguments_list.append(
                (
                    class_idx,
                    place_indices,
                    export_places_round_path,
                    class_idx_to_segment_ids[class_idx]  # make sure no duplicates are here
                )
            )

        # get the collated results!!
        print("Loading the Places Datasets (to combine nn in one place).")
        p = Pool(20)
        collated_instances = p.map(PlacesDataset.get_instances_from_class_idx, arguments_list)

        # save the per instances results
        class_idx_to_instances = {}
        for idx, class_idx in enumerate(class_indices):
            class_idx_to_instances[class_idx] = collated_instances[idx]

        # now save per class results!
        for class_idx in class_indices:
            filename = pjoin(CLUSTER_FOLDER_NAME.replace("merged/trees", "exported_places"),
                             "mturk_round_{:03d}".format(args.round_number),
                             "{:03d}.json".format(class_idx))
            make_dir(filename)
            write_to_json(filename, class_idx_to_instances[class_idx])

    if args.step == "merge_instances":
        # pass

        all_instances = []
        for class_idx in tqdm(class_indices):
            filename = pjoin(CLUSTER_FOLDER_NAME.replace("merged/trees", "exported_places"),
                             "mturk_round_{:03d}".format(args.round_number),
                             "{:03d}.json".format(class_idx))
            instances = load_from_json(filename)
            all_instances += instances

        # now write to a file
        filename = pjoin(CLUSTER_FOLDER_NAME.replace("merged/trees", "exported_places"),
                         "mturk_round_{:03d}.json".format(args.round_number))
        make_dir(filename)
        write_to_json(filename, all_instances)

    if args.step == "combine_with_ade":
        """Here we add the new_dataset to the base_dataset.
        """

        # load the previous dataset
        # TODO: change the name here
        base_dataset_filename = "/data/vision/torralba/scaleade/scaleade/data/ade/trainA_1k.json"
        # base_dataset_filename = "/data/vision/torralba/scratch/dimpapa/git_iccv21/scaleade/data/ade/trainA_1k.json"
        base_dataset = COCODataset(
            image_path="/data/vision/torralba/ade20k-places/data/",
            gt_data_filename=base_dataset_filename,
        )

        # this will always be the `new_dataset` just to get the image_id and image_data
        new_dataset_filename = "/data/vision/torralba/scratch/ethanweber/scaleade/rounds/iccv/places_ade_1k_COCO/merged.json"
        new_dataset = COCODataset(
            image_path="/data/vision/torralba/ade20k-places/data/",
            gt_data_filename=new_dataset_filename
        )

        # load the dataset we just labeled (unlabeled pool)
        instances_filename = pjoin(
            CLUSTER_FOLDER_NAME.replace("merged/trees", "exported_places"),
            "mturk_round_{:03d}.json".format(args.round_number))
        instances = load_from_json(instances_filename)

        print("Adding the instances")
        for idx in range(len(instances)):

            instance = instances[idx]

            # don't do this since it's already in the COCODataset class
            # instance["category_id"] += 1

            # get image_data
            image_id = new_dataset.image_id(instance)
            image_data = new_dataset.image_id_to_image_data[image_id]
            # add the instance
            base_dataset.add_instance_to_gt_data(instance, image_data)

        print("Removing duplicates and saving.")
        # remove the duplicates and save to file
        filename = pjoin(
            CLUSTER_FOLDER_NAME.replace("merged/trees", "exported_places"),
            "merged_round_{:03d}.json".format(args.round_number))
        make_dir(filename)
        write_to_json(filename, base_dataset.gt_data)

        # base_dataset.save_file_without_duplicates(filename)

    if args.step == "rebalance_dist":

        # raise NotImplementedError("This is not done yet.")
        #
        # load the previous dataset
        # TODO: change the name here
        base_dataset_filename = "/data/vision/torralba/scaleade/scaleade/data/ade/trainA_1k.json"
        base_dataset = COCODataset(
            image_path="/data/vision/torralba/ade20k-places/data/",
            gt_data_filename=base_dataset_filename,
        )

        # load the dataset we want to rebalance
        dataset_filename = pjoin(
            CLUSTER_FOLDER_NAME.replace("merged/trees", "exported_places"),
            "merged_round_{:03d}.json".format(args.round_number))
        new_dataset = COCODataset(
            image_path="/data/vision/torralba/ade20k-places/data/",
            gt_data_filename=dataset_filename
        )

        # keep track of everything to delete
        annotation_ids_to_delete = []

        # clip above 10% if big increase
        # calculate the number to remove per class
        num_to_remove = {}
        for class_idx in class_indices:
            class_id = class_idx + 1

            old_num = len(base_dataset.class_id_to_annotation_ids[class_id])
            old_den = len(base_dataset.annotation_id_to_image_id.keys())

            new_num = len(new_dataset.class_id_to_annotation_ids[class_id])
            new_den = len(new_dataset.annotation_id_to_image_id.keys())

            old_ratio = old_num / old_den
            new_ratio = new_num / new_den

            ratio = new_ratio / old_ratio

            # max 10% increase
            target_ratio = 1.1
            if ratio > target_ratio:
                # target_ratio = (new_num / new_den) * ?? / (old_num / old_den)
                # target_ratio = (new_ratio) * ?? / (old_ratio)
                perc_to_keep = (target_ratio * old_ratio) / new_ratio
                num_to_remove = int(new_num - (perc_to_keep * new_num))

                # remove num_to_remove for this class
                print(f"Removing # from class_id: {class_id}", num_to_remove)

                annotation_ids = new_dataset.class_id_to_annotation_ids[class_id]
                annotation_ids_to_delete += random.sample(annotation_ids, num_to_remove)

        annotation_ids_to_delete = set(annotation_ids_to_delete)
        annotation_indices = []
        for idx, annotation in enumerate(new_dataset.gt_data["annotations"]):
            if annotation["id"] in annotation_ids_to_delete:
                annotation_indices.append(idx)

        annotation_indices = sorted(annotation_indices, reverse=True)
        for idx in annotation_indices:
            del new_dataset.gt_data["annotations"][idx]

        # now save the final file
        filename = pjoin(
            CLUSTER_FOLDER_NAME.replace("merged/trees", "exported_places"),
            "rebalanced_merged_round_{:03d}.json".format(args.round_number))
        make_dir(filename)
        write_to_json(filename, new_dataset.gt_data)

    if args.step == "add_non_negatives":
        # go through all the images
        # add all other detections to the images
        pass

        # all the Places images
        # NOTE that this keeps track of everything
        dataset_merged = COCODataset(
            image_path="/data/vision/torralba/ade20k-places/data/",
            gt_data_filename="/data/vision/torralba/scratch/ethanweber/scaleade/rounds/iccv/places_ade_1k_COCO/merged.json",
        )

        # load the dataset we want to rebalance
        print("Loading the dataset that we want to add to.")
        dataset_filename = pjoin(
            CLUSTER_FOLDER_NAME.replace("merged/trees", "exported_places"),
            "rebalanced_merged_round_{:03d}.json".format(args.round_number))
        dataset = COCODataset(
            image_path="/data/vision/torralba/ade20k-places/data/",
            gt_data_filename=dataset_filename
        )

        # we need to add all the instances...
        place_indices = [i for i in range(0, 11)]
        places_folder = PlacesFolder(
            place_indices,
            args.places_round,
            category_indices=class_indices,
            export_places_path=args.export_places_path)

        print("Setting the accepted instances.")
        accepted_instance_keys = []
        image_filenames_that_exist = set()
        for instance in tqdm(dataset.gt_data["annotations"]):
            # NOTE: this assumes that "instance_idx" is not tampered with!
            # we operate under the assumption that "instance_idx" with image_filename creates a unique identifier
            if "instance_idx" in instance:
                # NOTE: split_idx should solve the issue of using the same images in the validation set
                split_idx = instance["split_idx"]
                image_filename = dataset.image_filename(instance)
                key = str((image_filename, instance["bbox"]))
                # assert key not in accepted_instances
                accepted_instance_keys.append(key)
                image_filenames_that_exist.add(image_filename)

        print("Number of added places instances: {}".format(len(accepted_instance_keys)))

        # duplicates would be due to the evaluation set?
        # TODO: figure out why there is a descrepancy here?
        accepted_instance_keys = set(accepted_instance_keys)
        print("Number of added places after filtering: {}".format(len(accepted_instance_keys)))

        # need to add instances to dataset
        num_already_accepted = 0
        non_accepted_instances = []
        print("Going over all the places indices.")
        for place_idx in tqdm(place_indices):
            for category_idx in tqdm(class_indices):
                instance_filename = places_folder.get_instances_filename(
                    place_idx, category_idx
                )
                instances = load_from_json(instance_filename)

                for instance in instances:

                    # check if instance is in the dataset already
                    assert "instance_idx" in instance
                    image_filename = dataset_merged.image_filename(instance)
                    split_idx = instance["split_idx"]
                    key = str((image_filename, instance["bbox"]))
                    if key in accepted_instance_keys:
                        num_already_accepted += 1
                    elif image_filename in image_filenames_that_exist:  # only add for images we can
                        new_instance = copy.deepcopy(instance)
                        new_instance["accept"] = 0  # don't keep this!
                        non_accepted_instances.append(new_instance)
                    # break
                # break
            # break

        print("num_already_accepted: ", num_already_accepted)
        print("num non_accepted_instances: ", len(non_accepted_instances))

        print("\nAdding instances to the dataset.")
        thresh = 0
        num_added = 0
        for instance in tqdm(non_accepted_instances):
            # check if
            score = instance["score"] * instance["pred_iou"][0]
            if score >= thresh * 0.1:
                image_id = dataset_merged.image_id(instance)
                image_data = dataset_merged.image_id_to_image_data[image_id]
                dataset.add_instance_to_gt_data(instance, image_data)
                num_added += 1
            # pass

        print("number added above thresh, ", num_added)
        # now go through and add instances IF

        filename = pjoin(
            CLUSTER_FOLDER_NAME.replace("merged/trees", "exported_places"),
            "rebalanced_merged_round_{:03d}_non_neg_score_{:02d}.json".format(args.round_number, thresh))
        make_dir(filename)
        write_to_json(filename, dataset.gt_data)

    # todo:
if __name__ == '__main__':
    main()
