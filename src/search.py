"""Code for the search algorithm!
"""

import json
import os
import numpy as np
import random
from src.coco_dataset import COCODataset
from src.cluster import Clusterer
from src.utils import make_dir, write_to_json, pjoin, load_from_json
import pprint
from tqdm import tqdm
import multiprocessing


class Tree:
    """This class takes a Z and will search properly.
    """

    def __init__(
        self,
        class_idx=None,
        feature_name=None,
        folder=None
    ):
        self.class_idx = class_idx
        self.feature_name = feature_name
        assert folder is not None
        self.folder = folder

        instances_filename = pjoin(self.folder, "instances", "{:03d}.json".format(self.class_idx))
        if not os.path.exists(instances_filename):
            print("No instance file for class_idx {:03d}".format(class_idx))
        self.instances = load_from_json(instances_filename)

        self.Z = np.load(os.path.join(self.folder, "trees", self.feature_name, "{:03d}.npy".format(self.class_idx)))
        self.tree = None

        # values to cache to avoid recompute
        self.cached_segment_ids = {}
        self.cached_quality = {}

        # initialization!
        self.init_tree()

    def get_universal_thresh(self, K_a=0.85):
        """Returns the universal threshold progress.
        """
        progress = {
            "num_clusters_visited": [],
            "num_accepted_clusters": [],
            "num_accepted_segments": [],
            "miou_of_accepted_segments": []
        }
        clusterer = Clusterer()
        clusterer.setZ(self.Z)

        num_intervals = 100
        print("Running universal threshold experiments for {} intervals".format(num_intervals))
        intervals = list(np.logspace(0, np.log10(len(self.Z) - 1), num_intervals).astype(int))

        for max_num_clusters in tqdm(intervals):
            clusterer.compute_flat_clusters(max_num_clusters=max_num_clusters)
            clusters = clusterer.extract_clusters()

            num_clusters_visited = 0
            num_accepted_clusters = 0
            accepted_segment_ids = []

            for segment_ids in clusters:
                quality = self.quality(None, segment_ids=segment_ids)
                num_clusters_visited += 1
                if quality >= K_a:
                    num_accepted_clusters += 1
                    accepted_segment_ids += segment_ids

            progress["num_clusters_visited"].append(num_clusters_visited)
            progress["num_accepted_clusters"].append(num_accepted_clusters)
            progress["num_accepted_segments"].append(len(accepted_segment_ids))
            progress["miou_of_accepted_segments"].append(self.miou(None, segment_ids=accepted_segment_ids))

        return progress

    def init_tree(self):
        tree = {}
        tree["children"] = {}
        tree["parent"] = {}
        num_leaves = self.Z.shape[0]
        for idx in range(num_leaves):
            cluster_id = num_leaves + 1 + idx
            tree["children"][idx] = []  # no children if leaf
            # https://docs.scipy.org/doc/scipy/reference/generated/scipy.cluster.hierarchy.linkage.html
            tree["children"][cluster_id] = [int(self.Z[idx][0]), int(self.Z[idx][1])]
            for child_id in tree["children"][cluster_id]:
                tree["parent"][child_id] = cluster_id
        self.tree = tree

    def children(self, cluster_id):
        try:
            children = self.tree["children"][cluster_id]
            return children
        except:
            return []

    def parent(self, cluster_id):
        try:
            parent = self.tree["parent"][cluster_id]
            return parent
        except:
            return None

    def root(self):
        cluster_id = 0
        parent_id = self.parent(cluster_id)
        while parent_id:
            cluster_id = parent_id
            parent_id = self.parent(cluster_id)
        return cluster_id

    def segment_ids(self, cluster_id):
        """Recurse to leaves and return the segment ids.
        """
        if cluster_id in self.cached_segment_ids:
            return self.cached_segment_ids[cluster_id]
        children = self.children(cluster_id)
        if len(children) == 0:
            segment_ids = [cluster_id]
        else:
            segment_ids = self.segment_ids(children[0]) + self.segment_ids(children[1])
        self.cached_segment_ids[cluster_id] = segment_ids
        return segment_ids

    def prior(self, cluster_id):
        segment_ids = self.segment_ids(cluster_id)
        l = []
        for segment_id in segment_ids:
            l.append(
                self.instances[segment_id]["pred_iou"][0] *
                self.instances[segment_id]["score"]
            )
        return np.array(l).mean()

    def quality(self, cluster_id, K_iou=0.75, segment_ids=None):
        if segment_ids is None:
            segment_ids = self.segment_ids(cluster_id)
        l = []
        for segment_id in segment_ids:
            value = 1 if self.instances[segment_id]["iou"][0] > K_iou else 0
            l.append(value)
        return np.array(l).mean()

    def score(self, cluster_id):
        segment_ids = self.segment_ids(cluster_id)
        l = []
        for segment_id in segment_ids:
            l.append(self.instances[segment_id]["score"])
        return np.array(l).mean()

    def miou(self, cluster_ids, segment_ids=None):
        if segment_ids is None:
            segment_ids = []
            for cluster_id in cluster_ids:
                segment_ids += self.segment_ids(cluster_id)
        ious = []
        for segment_id in segment_ids:
            ious.append(self.instances[segment_id]["iou"][0])
        return np.array(ious).mean() if len(ious) > 0 else 0.0

    def get_sorted_cluster_ids(self, cluster_ids, heuristic, reverse=True):
        # assert heuristic in ["prior"]
        if heuristic == "none":
            return cluster_ids

        # otherwise sort
        items = []
        for cluster_id in cluster_ids:
            if heuristic == "rand":
                value = random.uniform(0, 1)
            elif heuristic == "iou":
                raise ValueError("Not implemented!")
            elif heuristic == "score":
                value = self.score(cluster_id)
            elif heuristic == "score_iou":
                value = self.prior(cluster_id)
            elif heuristic == "size_cluster":
                value = len(self.segment_ids(cluster_id))
            elif heuristic == "real_iou":
                value = self.miou([cluster_id])
            elif heuristic == "real_quality":
                raise ValueError("Not implemented!")
            elif heuristic == "score_iou_size":
                raise ValueError("Not implemented!")
            else:
                raise ValueError("Not implemented!")
            items.append(
                (cluster_id, value)
            )
        items = sorted(items, key=lambda x: x[1], reverse=reverse)
        items = [x[0] for x in items]
        return items

    def save_progress(self, search_method, heuristic, K_prior_1, K_prior_2, min_num_samples, progress):

        if K_prior_1 is None and K_prior_2 is None:
            active_section = "naive"
        else:
            active_section = "{:03d}_{:03d}".format(int(K_prior_1 * 100), int(K_prior_2 * 100))

        filename = os.path.join(self.folder, "search", self.feature_name, search_method,
                                heuristic, active_section, "{:03d}".format(min_num_samples), "{:03d}.json".format(self.class_idx))
        make_dir(filename)
        write_to_json(filename, progress)
        print("Saving to ", filename)

    def search(
            self,
            K_prior_1=0.2,
            K_prior_2=0.7,
            K_a=0.85,
            K_r=0.15,
            search_method="heuristics_only",
            heuristic="score_iou",
            min_num_samples=1,
            budget=None):

        assert K_prior_1 == 0.0

        assert search_method in [
            "universal_threshold",
            "bfs",
            "dfs",
            "random_walk",
            "heuristics_only"
        ]
        assert heuristic in [
            "none",
            "rand",
            "iou",
            "score",
            "score_iou",
            "size_cluster",
            "real_iou",
            "real_quality",
            "score_iou_size"
        ]

        if search_method == "universal_threshold":
            return None, None, self.get_universal_thresh(K_a=K_a)

        # store the progress
        progress = {
            "num_clusters_visited": [],
            "num_accepted_clusters": [],
            "num_accepted_segments": [],
            "miou_of_accepted_segments": [],
            "num_questions_asked": [],
        }

        num_clusters_visited = 0
        num_accepted_clusters = 0
        num_accepted_segments = 0
        miou_of_accepted_segments = 0
        num_questions_asked = 0

        accepted = []
        rejected = []
        queue = [self.root()]
        while queue:

            # grab an element from the queue
            if search_method == "bfs":  # FIFO
                cluster_id = queue.pop(0)
            elif search_method == "dfs":  # LIFO
                cluster_id = queue.pop()
            elif search_method == "random_walk":
                random.shuffle(queue)
                cluster_id = queue.pop()
            elif search_method == "heuristics_only":
                queue = self.get_sorted_cluster_ids(queue, heuristic)
                cluster_id = queue.pop(0)

            # if too few examples
            if len(self.segment_ids(cluster_id)) < min_num_samples:  # terminate if too few!
                progress["num_clusters_visited"].append(num_clusters_visited)
                progress["num_accepted_clusters"].append(num_accepted_clusters)
                progress["num_accepted_segments"].append(num_accepted_segments)
                progress["miou_of_accepted_segments"].append(miou_of_accepted_segments)
                progress["num_questions_asked"].append(num_questions_asked)
                continue

            # # terminate if you reach a leaf node and low est quality
            # if K_prior_2 and len(self.segment_ids(cluster_id)) == min_num_samples and est_quality < K_prior_2:
            #     progress["num_clusters_visited"].append(num_clusters_visited)
            #     progress["num_accepted_clusters"].append(num_accepted_clusters)
            #     progress["num_accepted_segments"].append(num_accepted_segments)
            #     progress["miou_of_accepted_segments"].append(miou_of_accepted_segments)
            #     continue

            est_quality = self.prior(cluster_id)
            if K_prior_1 and est_quality < K_prior_1:  # terminate

                progress["num_clusters_visited"].append(num_clusters_visited)
                progress["num_accepted_clusters"].append(num_accepted_clusters)
                progress["num_accepted_segments"].append(num_accepted_segments)
                progress["miou_of_accepted_segments"].append(miou_of_accepted_segments)
                progress["num_questions_asked"].append(num_questions_asked)
                continue

            if K_prior_2 and len(self.segment_ids(cluster_id)) > min_num_samples and est_quality < K_prior_2:  # split

                children = self.children(cluster_id)
                queue += self.get_sorted_cluster_ids(children, heuristic, reverse=False)

                progress["num_clusters_visited"].append(num_clusters_visited)
                progress["num_accepted_clusters"].append(num_accepted_clusters)
                progress["num_accepted_segments"].append(num_accepted_segments)
                progress["miou_of_accepted_segments"].append(miou_of_accepted_segments)
                progress["num_questions_asked"].append(num_questions_asked)

                continue

            # ask questions!
            quality = self.quality(cluster_id)
            num_clusters_visited += 1  # means we visited it!!!
            # num_questions_asked += min(15, len(self.segment_ids(cluster_id)))
            num_questions_asked += max(0, min(7.5, len(self.segment_ids(cluster_id)) - 7.5))

            if quality >= K_a:  # accept
                accepted.append(cluster_id)
                num_accepted_clusters += 1
                num_accepted_segments += len(self.segment_ids(cluster_id))
                miou_of_accepted_segments = self.miou(accepted)  # TODO(ethan): this could be cleaner

                progress["num_clusters_visited"].append(num_clusters_visited)
                progress["num_accepted_clusters"].append(num_accepted_clusters)
                progress["num_accepted_segments"].append(num_accepted_segments)
                progress["miou_of_accepted_segments"].append(miou_of_accepted_segments)
                progress["num_questions_asked"].append(num_questions_asked)
                continue

            if quality < K_r:  # reject
                rejected.append(cluster_id)
                progress["num_clusters_visited"].append(num_clusters_visited)
                progress["num_accepted_clusters"].append(num_accepted_clusters)
                progress["num_accepted_segments"].append(num_accepted_segments)
                progress["miou_of_accepted_segments"].append(miou_of_accepted_segments)
                progress["num_questions_asked"].append(num_questions_asked)
                continue

            # otherwise split
            children = self.children(cluster_id)
            queue += self.get_sorted_cluster_ids(children, heuristic, reverse=False)

        # TODO(ethan): make sure methods that call this are correct!!
        return accepted, rejected, progress

    def cluster_ids_to_instances(self, cluster_ids):
        """Returns an "instances" object with the good detected instances.
        """
        instances = []
        segment_ids = []
        for cluster_id in cluster_ids:
            segment_ids += self.segment_ids(cluster_id)

        # sanity check so we don't double count
        assert len(segment_ids) == len(set(segment_ids))

        for segment_id in segment_ids:
            instances.append(self.instances[segment_id])

        return instances

    @staticmethod
    def export_helper(arguments):
        folder, class_idx, feature_name, search_method, heuristic, K_prior_1, K_prior_2, min_num_samples = arguments
        tree = Tree(class_idx=class_idx, feature_name=feature_name, folder=folder)
        _, _, progress = tree.search(
            search_method=search_method,
            heuristic=heuristic,
            K_prior_1=K_prior_1,
            K_prior_2=K_prior_2,
            min_num_samples=min_num_samples
        )
        tree.save_progress(search_method, heuristic, K_prior_1, K_prior_2, min_num_samples, progress)

    @staticmethod
    def fast_export_dataset_with_annotations():
        pass

    @staticmethod
    def fast_get_accepted_rejected_instances(arguments):
        search_algorithm, input_folder, class_idx = arguments
        feature_name, search_method, heuristic, active_section, min_num_samples = search_algorithm.split("-")
        K_prior_1 = None
        K_prior_2 = None
        if active_section != "naive":
            K_prior_1, K_prior_2 = [float(x) / 100.0 for x in active_section.split("_")]
        min_num_samples = int(min_num_samples)

        # TODO(ethan): remove this! this is in case some instances don't have any data
        filename = os.path.join(input_folder, "trees", feature_name, "{:03d}.npy".format(class_idx))
        if not os.path.exists(filename):
            print("Skipping search for", filename)
            return [], []

        print("Starting search for", filename)
        tree = Tree(class_idx=class_idx, feature_name=feature_name, folder=input_folder)
        accepted, rejected, progress = tree.search(
            search_method=search_method,
            heuristic=heuristic,
            K_prior_1=K_prior_1,
            K_prior_2=K_prior_2,
            min_num_samples=min_num_samples,
            budget=None  # TODO(ethan): change this!
        )
        accepted_instances = tree.cluster_ids_to_instances(accepted)
        rejected_instances = tree.cluster_ids_to_instances(rejected)
        print("Finished search for", filename)
        return accepted_instances, rejected_instances

    @staticmethod
    def export_dataset_with_annotations(
        search_algorithm,
        input_folder=None,  # "/data/vision/torralba/scratch/ethanweber/scaleade/ade_rounds/round_00",
        image_path=None,
        train_gt_data_filename=None,
        gt_data_filename=None,
        add_to_filename=None,
        class_indices=None,
    ):
        assert input_folder is not None
        assert gt_data_filename is not None

        print(f"Input folder: {input_folder} with {search_algorithm}")

        # TODO(ethan): add a parameter here!

        # TODO(ethan): change the name of the following!!
        ade_train1K_dataset = COCODataset(
            image_path=image_path,
            gt_data_filename=train_gt_data_filename,
        )
        print("NOTE!!\n\n Using dataset with name: \n{}\n\n ".format(gt_data_filename))

        ade_train19K_dataset = COCODataset(
            image_path=image_path,
            gt_data_filename=gt_data_filename,
        )

        arguments = []
        for class_idx in class_indices:
            arguments.append(
                (search_algorithm, input_folder, class_idx)
            )

        accepted_instances = []
        class_idx_to_rejected_instances = {}
        print("Running on everything.")
        num_cpus = 24
        p = multiprocessing.Pool(num_cpus)
        class_idx_to_acc_rej = p.map(Tree.fast_get_accepted_rejected_instances, arguments)
        for idx, class_idx in enumerate(class_indices):
            acc, rej = class_idx_to_acc_rej[idx]
            accepted_instances += acc
            class_idx_to_rejected_instances[class_idx] = rej

        for instance in accepted_instances:
            image_id = ade_train19K_dataset.image_id(instance)
            image_data = ade_train19K_dataset.image_id_to_image_data[image_id]
            ade_train1K_dataset.add_instance_to_gt_data(instance, image_data)

        # now save the dataset!
        output_filename = os.path.join(input_folder, "exported_datasets", f"{search_algorithm}.json")
        make_dir(output_filename)
        with open(output_filename, "w") as outfile:
            json.dump(ade_train1K_dataset.gt_data, outfile)

        # also save the rejected instances!
        output_filename = os.path.join(input_folder, "rejected_instances", f"{search_algorithm}.json")
        make_dir(output_filename)
        with open(output_filename, "w") as outfile:
            json.dump(class_idx_to_rejected_instances, outfile)
