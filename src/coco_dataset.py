"""Class to use a coco-formatted dataset.
"""
import logging
import numpy as np
from tqdm import tqdm
import json
import pprint
import os
import random
from collections import defaultdict
import cv2
import fastcluster
from PIL import Image, ImageDraw, ImageFont
from PIL import ImagePath
from src.utils import get_polygons_from_mask, draw_polygon_on_image, load_from_json
from pycocotools import mask as pycocomask
from goat.view import utils as view_utils


def get_partial_coco_dataset(coco_dataset, num_images=100):
    """Returns a partial COCO dataset from an original one."""
    partial_coco_dataset = {}
    partial_coco_dataset["categories"] = coco_dataset["categories"]
    partial_coco_dataset["images"] = coco_dataset["images"][:num_images]
    image_ids = set([x["id"] for x in partial_coco_dataset["images"]])
    partial_coco_dataset["annotations"] = []
    for annotation in coco_dataset["annotations"]:
        if annotation["image_id"] in image_ids:
            partial_coco_dataset["annotations"].append(annotation)
    return partial_coco_dataset


def get_image_with_mask_overlayed(image, mask, color=(1.0, 0, 0), alpha=0.5):
    """Apply the given mask to the image.
    Args:
        color (tuple): (r, g, b) with range (0, 1)
    """
    im = image.copy()
    for c in range(3):
        im[:, :, c] = np.where(
            mask == 1, im[:, :, c] * (1 - alpha) + alpha * color[c] * 255, im[:, :, c])
    return im


def get_expanded_bounding_box(bbox, image_size, perc_exp=0.1):
    """Returns the expanded bounding box.
    Args:
        perc_exp: percent expansion
    """
    h, w = image_size
    bbox = [int(round(b)) for b in bbox]  # [x, y, width, height]
    bbox[0] = max(int(round(bbox[0] - bbox[2] * perc_exp)), 0)
    bbox[1] = max(int(round(bbox[1] - bbox[3] * perc_exp)), 0)
    bbox[2] = min(int(round(bbox[2] + bbox[2] * perc_exp * 2)), w - bbox[0])
    bbox[3] = min(int(round(bbox[3] + bbox[3] * perc_exp * 2)), h - bbox[1])
    return bbox


class COCODataset:
    def __init__(self,
                 image_path="/data/vision/torralba/ade20k-places/data/ade_challenge/images/",
                 gt_data_filename="/data/vision/torralba/scratch/ethanweber/scaleade/data/ade/train.json"):
        assert os.path.exists(image_path)
        assert os.path.exists(gt_data_filename)
        self.image_path = image_path
        self.gt_data_filename = gt_data_filename
        # load the data
        self.gt_data = load_from_json(self.gt_data_filename)
        # dictionary mappings
        self.image_filename_to_image_id = None  # stores image filename -> image id
        self.image_id_to_image_filename = None  # stores image image id -> filename
        # stores image id -> list of annotation ids
        self.image_id_to_annotation_ids = None
        self.image_id_to_image_size = None
        self.image_id_to_image_data = None
        # stores class id -> list of annotation ids
        self.class_id_to_annotation_ids = None
        self.annotation_id_to_image_id = None  # stores annotation id -> image id
        self.annotation_id_to_annotation = None
        self.class_id_to_class_name = None  # stores class id -> class name
        self.class_name_to_class_id = None  # stores class name -> class id
        self.set_dictionary_mappings()

    def set_dictionary_mappings(self):
        # print("Setting dictionary mappings.")
        self.image_filename_to_image_id = {}  # one to one mapping
        self.image_id_to_image_filename = {}
        self.image_id_to_annotation_ids = {}
        self.image_id_to_image_size = {}
        self.image_id_to_image_data = {}
        for im_data in self.gt_data["images"]:
            image_filename = im_data["file_name"]
            image_id = im_data["id"]
            self.image_id_to_annotation_ids[image_id] = []
            self.image_id_to_image_size[image_id] = (im_data["height"], im_data["width"])
            self.image_id_to_image_data[image_id] = im_data
            self.image_filename_to_image_id[image_filename] = image_id
            self.image_id_to_image_filename[image_id] = image_filename
        self.class_id_to_annotation_ids = defaultdict(list)
        self.annotation_id_to_image_id = {}
        self.annotation_id_to_annotation = {}

        # delete from gt_data annotations where images do not exist
        annotation_idxs_to_delete = []
        for idx, annotation in enumerate(self.gt_data["annotations"]):
            image_id = annotation["image_id"]
            class_id = annotation["category_id"]
            annotation_id = annotation["id"]
            if image_id not in self.image_id_to_annotation_ids:
                annotation_idxs_to_delete.append(idx)
            else:
                self.image_id_to_annotation_ids[image_id].append(annotation_id)
                self.class_id_to_annotation_ids[class_id].append(annotation_id)
                assert annotation_id not in self.annotation_id_to_image_id
                assert annotation_id not in self.annotation_id_to_annotation
                self.annotation_id_to_image_id[annotation_id] = image_id
                self.annotation_id_to_annotation[annotation_id] = annotation

        # delete annotations where they don't have an image!
        if len(annotation_idxs_to_delete) > 0:
            print("Deleting {} annotations that don't belong to an image.".format(len(annotation_idxs_to_delete)))
            for idx in sorted(annotation_idxs_to_delete, reverse=True):
                del self.gt_data["annotations"][idx]

        self.class_id_to_class_name = {}
        self.class_name_to_class_id = {}
        for category_data in self.gt_data["categories"]:
            class_id = category_data["id"]
            class_name = category_data["name"]
            self.class_id_to_class_name[class_id] = class_name
            self.class_name_to_class_id[class_name] = class_id

        # set some specifics for places
        self.using_places = "split_id_image_id_to_global_image_id" in self.gt_data
        self.split_id_image_id_to_global_image_id = self.gt_data["split_id_image_id_to_global_image_id"] if self.using_places else None
        self.global_image_id_to_split_id_image_id = self.gt_data["global_image_id_to_split_id_image_id"] if self.using_places else None

        self.image_id_max = sorted(list(self.image_id_to_image_filename.keys())
                                   )[-1] if len(self.image_id_to_image_filename) != 0 else 0
        self.annotation_id_max = sorted(list(self.annotation_id_to_image_id.keys())
                                        )[-1] if len(self.annotation_id_to_image_id) != 0 else 0

    def set_accept_to_zeros_below_thresh(self, thresh=0.8):
        num_removed = 0
        for idx, annotation in enumerate(self.gt_data["annotations"]):
            accept_label = annotation.get("accepted", annotation.get("accept", 1))
            if accept_label == 1 and "score" in annotation:
                if annotation["score"] > thresh:
                    self.gt_data["annotations"][idx]["accept"] = 0
                    self.gt_data["annotations"][idx]["accepted"] = 0
                    num_removed += 1
        print("Changed {} values".format(num_removed))
        self.set_dictionary_mappings()

    def remove_images_without_annotations(self):

        image_ids_to_remove = set()
        for image_id, annotation_ids in tqdm(self.image_id_to_annotation_ids.items()):
            num_accepted = 0
            for annotation_id in annotation_ids:
                annotation = self.annotation_id_to_annotation[annotation_id]
                accept_label = annotation.get("accepted", annotation.get("accept", 1))
                num_accepted += accept_label
            if num_accepted == 0:
                image_ids_to_remove.add(image_id)

        image_indices_to_remove = set()
        for idx, image_data in enumerate(self.gt_data["images"]):
            if image_data["id"] in image_ids_to_remove:
                image_indices_to_remove.add(idx)

        print("Removing num images", len(image_indices_to_remove))

        for idx in sorted(list(image_indices_to_remove), reverse=True):
            del self.gt_data["images"][idx]

        self.set_dictionary_mappings()

    def add_instance_to_gt_data(self, instance, image_data):
        image_filename = image_data["file_name"]
        if image_filename not in self.image_filename_to_image_id:
            # need to add the image
            image_data["id"] = self.image_id_max + 1
            self.gt_data["images"].append(image_data)

            self.image_filename_to_image_id[image_filename] = image_data["id"]
            self.image_id_to_image_filename[image_data["id"]] = image_filename
            self.image_id_to_image_data[image_data["id"]] = image_data
            self.image_id_to_annotation_ids[image_data["id"]] = []

            self.image_id_max = image_data["id"]
        else:
            image_id = self.image_filename_to_image_id[image_filename]
            image_data = self.image_id_to_image_data[image_id]
            assert image_id == image_data["id"]

        # add the annotation
        instance["category_id"] += 1  # TODO(ethan): fix this discrepancy!
        instance["segmentation"] = instance["segmentation"][0]
        instance["image_id"] = image_data["id"]
        instance["id"] = self.annotation_id_max + 1

        self.annotation_id_to_annotation[instance["id"]] = instance
        self.gt_data["annotations"].append(instance)
        self.image_id_to_annotation_ids[image_data["id"]].append(instance["id"])
        self.annotation_id_max = instance["id"]

    def get_duplicate_annotation_ids(self, image_id, thresh=0.5):
        """Find the duplicates in the image!
        """
        annotation_ids = sorted(self.image_id_to_annotation_ids[image_id])
        duplicates = []
        # prioritize keeping the first ones!
        for i in range(len(annotation_ids)):
            for j in range(i + 1, len(annotation_ids)):
                ann_id_i = annotation_ids[i]
                ann_id_j = annotation_ids[j]
                ann_i = self.annotation_id_to_annotation[ann_id_i]
                ann_j = self.annotation_id_to_annotation[ann_id_j]
                # if not the same category, then skip
                if ann_i["category_id"] != ann_j["category_id"]:
                    continue

                maski = pycocomask.decode(ann_i["segmentation"])
                maskj = pycocomask.decode(ann_j["segmentation"])
                inter = np.logical_and(maski, maskj)
                union = np.logical_or(maski, maskj)
                iou = np.sum(inter) / np.sum(union)
                if iou > thresh:
                    duplicates.append(ann_j["id"])
        duplicates = list(set(duplicates))
        return duplicates

    def get_all_duplicate_annotation_ids(self, thresh=0.5):
        duplicates = []
        image_ids = set(self.image_id_to_image_filename.keys())
        for image_id in tqdm(image_ids):
            duplicates += self.get_duplicate_annotation_ids(image_id, thresh=thresh)
        return sorted(duplicates)

    def save_file_without_duplicates(self, filename, thresh=0.5):
        duplicates = self.get_all_duplicate_annotation_ids(thresh=thresh)
        assert len(duplicates) == len(set(duplicates))
        duplicates = set(duplicates)
        print("Found {} duplicates!".format(len(duplicates)))
        annotations = []
        for annotation in self.gt_data["annotations"]:
            if annotation["id"] in duplicates:
                continue
            annotations.append(annotation)

        self.gt_data["annotations"] = annotations
        with open(filename, "w") as outfile:
            json.dump(self.gt_data, outfile)

    def get_category_id_to_annotations(self, image_id):
        """
        """
        d = defaultdict(list)
        for annotation_id in self.image_id_to_annotation_ids[image_id]:
            annotation = self.annotation_id_to_annotation[annotation_id]
            d[annotation["category_id"]].append(annotation)
        return d

    def image(self, image_id):
        filename = os.path.join(
            self.image_path,
            self.image_id_to_image_filename[image_id]
        )
        return cv2.imread(filename)[:, :, ::-1]

    def image_detection(
        self,
        instance,
        crop=False,
        size=None,
        border_size=2,
        perc_exp=0.2,
        color=(0, 255, 0),
        oveylay_on_gt=False
    ):
        # TODO(ethan): speed this up with crop first!

        image_id = self.image_id(instance)
        if oveylay_on_gt:
            image = self.get_gt_image(image_id)
        else:
            image = self.image(image_id)
        segmentation = instance["segmentation"]
        if isinstance(segmentation, list):
            segmentation = segmentation[0]
        mask = pycocomask.decode(segmentation)

        # potentially crop
        if crop:
            bbox = [int(x) for x in instance["bbox"]]  # x, y, w, h
            image_size = self.image_id_to_image_size[image_id]
            bbox = get_expanded_bounding_box(bbox, image_size, perc_exp=perc_exp)
            x, y, w, h = bbox
            image = image[y:y + h, x:x + w]
            mask = mask[y:y + h, x:x + w]

        if size:
            mask = cv2.resize(mask, size, interpolation=cv2.INTER_NEAREST)
            image = cv2.resize(image, size, interpolation=cv2.INTER_LINEAR)

        # draw polygons
        # image = get_image_with_mask_overlayed(image, mask)
        polygons = get_polygons_from_mask(mask)
        for polygon in polygons:
            image = draw_polygon_on_image(image, polygon, radius=border_size, point_color=color)

        return image

    def image_detections(
        self,
        instances,
        crop=False,
        size=None,
        border_size=2,
        perc_exp=0.2,
        color=(0, 255, 0)
    ):
        # TODO(ethan): speed this up with crop first!

        image_id = self.image_id(instances[0])
        image = self.image(image_id)

        for instance in instances:

            segmentation = instance["segmentation"]
            if isinstance(segmentation, list):
                segmentation = segmentation[0]
            mask = pycocomask.decode(segmentation)

            # potentially crop
            if crop:
                bbox = [int(x) for x in instance["bbox"]]  # x, y, w, h
                image_size = self.image_id_to_image_size[image_id]
                bbox = get_expanded_bounding_box(bbox, image_size, perc_exp=perc_exp)
                x, y, w, h = bbox
                image = image[y:y + h, x:x + w]
                mask = mask[y:y + h, x:x + w]

            if size:
                mask = cv2.resize(mask, size, interpolation=cv2.INTER_NEAREST)
                image = cv2.resize(image, size, interpolation=cv2.INTER_LINEAR)

            # draw polygons

            color = tuple(list((255.0 * np.random.rand(3)).astype("int")))  # RGB, alpha
            image = get_image_with_mask_overlayed(image, mask, color=tuple((np.array(color) / 255.0)))
            polygons = get_polygons_from_mask(mask)
            for polygon in polygons:
                image = draw_polygon_on_image(image, polygon, radius=border_size, point_color=(255, 255, 255))

        return image

    def image_id(self, instance):
        if self.using_places:
            key = "{},{}".format(instance["split_idx"], instance["image_id"])
            image_id = self.split_id_image_id_to_global_image_id[key]
        else:
            image_id = instance["image_id"]
        return image_id

    def image_filename(self, instance):
        image_id = self.image_id(instance)
        return self.image_id_to_image_filename[image_id]

    def image_detection_temp(
        self,
        instance
    ):
        image_id = self.image_id(instance)
        image = self.image(image_id)
        mask = pycocomask.decode(instance["segmentation"][0])
        image = get_image_with_mask_overlayed(image, mask)
        return image

    def get_gt_image(
        self,
        image_id,
        show_colors=False,
        show_class_name=True,
        thresh=None
    ):

        image = self.image(image_id)
        annotation_ids = self.image_id_to_annotation_ids[image_id]
        used_box_strs = set()
        for annotation_id in annotation_ids:

            annotation = self.annotation_id_to_annotation[annotation_id]
            # if "score" in annotation:

            bbox_str = str(annotation["bbox"])
            assert bbox_str not in used_box_strs, "You probably have a duplicate instances..."
            used_box_strs.add(bbox_str)

            mask = pycocomask.decode(annotation["segmentation"])
            color = tuple(list((255.0 * np.random.rand(3)).astype("int")) + [int(200)])  # RGB, alpha
            polygons = get_polygons_from_mask(mask)
            im_pil = Image.fromarray(image)
            for polygon_temp in polygons:
                polygon = [tuple(x) for x in polygon_temp]
                im_dra = ImageDraw.Draw(im_pil, "RGBA")
                im_dra.polygon(polygon, fill=color, outline=None)
                im_dra.line(polygon, fill="white", width=4)
            # try to draw text the category name
            # try:
            #     point = polygons[0][0]
            #     class_name = self.class_id_to_class_name[annotation["category_id"]]
            #     # font = ImageFont.load("arial.pil")
            #     # font = ImageFont.truetype(r'C:\Users\System-Pc\Desktop\arial.ttf', 20)
            #     font = ImageFont.truetype("arial.ttf", 15)
            #     im_dra.text(point, class_name, font=font)
            # except:
            #     pass

            image = np.asarray(im_pil)

            # draw a bbox if
            accept_label = annotation.get("accepted", annotation.get("accept", 1))
            # print(accept_label)
            if show_colors:
                color = (0, 255, 0) if accept_label else (255, 0, 0)
                x, y, w, h = [int(x) for x in annotation["bbox"]]  # x, y, w, h
                start_point = (x, y)
                end_point = (x + w, y + h)
                thickness = 2
                image = cv2.rectangle(image, start_point, end_point, color, thickness)

            # print(accept_label)
            # if "accept" in annotation:
            #     accept_label = annotation["accept"]
            #     if accept_label == 0:
            #         print(accept_label)

            # image = np.asarray(im_pil)
            if show_class_name:
                try:
                    point = tuple(polygons[0][0])
                    class_name = self.class_id_to_class_name[annotation["category_id"]]
                    # font = ImageFont.load("arial.pil")
                    # font = ImageFont.truetype(r'C:\Users\System-Pc\Desktop\arial.ttf', 20)
                    # im_dra.text(point, class_name, size=24)
                    fontScale = 0.6
                    color = (255, 255, 255)
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    thickness = 2
                    image = cv2.putText(image, class_name, point, font, fontScale, color, thickness, cv2.LINE_AA)
                except:
                    pass

        return image

    def add_gt_to_inference_predictions(
        self,
        inference_predictions
    ):
        """
        Adds the following fields to inference_predictions:
        - "gt_id"
        - "gt_segmentation"
        - "iou"
        """
        # iterate over predicted instances
        for i in tqdm(range(len(inference_predictions))):
            image_id = inference_predictions[i]["image_id"]
            instances = inference_predictions[i]["instances"]
            category_id_to_gt_annotations = self.get_category_id_to_annotations(image_id)
            for j in range(len(instances)):
                annotation = instances[j]
                category_id = annotation["category_id"] + 1  # TODO(ethan): fix this + 1 offset error!
                # VALUES WE WANT TO ADD
                assert "gt_id" not in inference_predictions[i]["instances"][j]
                assert "gt_segmentation" not in inference_predictions[i]["instances"][j]
                assert "iou" not in inference_predictions[i]["instances"][j]
                inference_predictions[i]["instances"][j]["gt_id"] = []
                inference_predictions[i]["instances"][j]["gt_segmentation"] = []
                inference_predictions[i]["instances"][j]["iou"] = []
                # iterate over the multiple masks
                for k in range(len(annotation["segmentation"])):
                    mask = pycocomask.decode(annotation["segmentation"][k])
                    ### --- START OF VALUES WE WANT TO ADD --- ###
                    gt_id = None
                    gt_segmentation = None
                    iou = 0.0
                    ### -- END --- ###
                    # iterate over the gt masks
                    for gt_annotation in category_id_to_gt_annotations[category_id]:
                        temp_gt_segmentation = gt_annotation["segmentation"]
                        temp_gt_mask = pycocomask.decode(temp_gt_segmentation)
                        inter = np.logical_and(temp_gt_mask, mask)
                        union = np.logical_or(temp_gt_mask, mask)
                        temp_iou = np.sum(inter) / np.sum(union)
                        if iou is None or temp_iou > iou:
                            gt_id = gt_annotation["id"]
                            gt_segmentation = temp_gt_segmentation
                            iou = temp_iou
                    # TODO: now add these to the coco formatted dataset
                    inference_predictions[i]["instances"][j]["gt_id"].append(gt_id)
                    inference_predictions[i]["instances"][j]["gt_segmentation"].append(gt_segmentation)
                    inference_predictions[i]["instances"][j]["iou"].append(iou)

        return inference_predictions

    def add_gt_to_instances(self, instances, places_coco_dataset=None, ignore_zeros=True):
        """Add GT to a list of instances!
        """
        # assert places_coco_dataset is not None
        instances_with_gt = instances  # NOTE(ethan): this is not a deep copy!
        count = 0
        for i in tqdm(range(len(instances))):
            instance = instances[i]
            category_id = instance["category_id"] + 1  # TODO(ethan): fix this + 1 offset error!
            
            if isinstance(places_coco_dataset, type(None)):
                image_filename = self.image_filename(instance)
            else:
                image_filename = places_coco_dataset.image_filename(instance)

            gt_id = None
            gt_segmentation = None
            iou = None  # NOTE(ethan): this is None since we don't assume the images are fully annotated

            if image_filename in self.image_filename_to_image_id:
                image_id = self.image_filename_to_image_id[image_filename]
                category_id_to_gt_annotations = self.get_category_id_to_annotations(image_id)

                mask = pycocomask.decode(instance["segmentation"][0])
                ### --- START OF VALUES WE WANT TO ADD --- ###
                ### -- END --- ###
                # iterate over the gt masks
                for gt_annotation in category_id_to_gt_annotations[category_id]:
                    temp_gt_segmentation = gt_annotation["segmentation"]
                    temp_gt_mask = pycocomask.decode(temp_gt_segmentation)
                    inter = np.logical_and(temp_gt_mask, mask)
                    union = np.logical_or(temp_gt_mask, mask)
                    temp_iou = np.sum(inter) / np.sum(union)
                    if ignore_zeros and temp_iou == 0.0:
                        continue  # don't count this!
                    if iou is None or temp_iou > iou:
                        gt_id = gt_annotation["id"]
                        gt_segmentation = temp_gt_segmentation
                        iou = temp_iou

            instances_with_gt[i]["gt_id"] = gt_id
            instances_with_gt[i]["gt_segmentation"] = gt_segmentation
            instances_with_gt[i]["iou"] = iou

        return instances_with_gt

    @staticmethod
    def gt_id_to_instance(
        inference_predictions
    ):
        """Returns the gt id mapped to the instance.
        """
        d = {}
        for i in range(len(inference_predictions)):
            image_id = inference_predictions[i]["image_id"]
            instances = inference_predictions[i]["instances"]
            for j in range(len(instances)):
                for k in range(len(inference_predictions[i]["instances"][j]["gt_id"])):
                    gt_id = inference_predictions[i]["instances"][j]["gt_id"][k]
                    if gt_id is None:
                        continue
                    # if nowhere, add it
                    if gt_id not in d:
                        d[gt_id] = (inference_predictions[i]["instances"][j], k)
                    else:
                        # potentially replace current instance
                        prev_inst, prev_k = d[gt_id]
                        prev_iou = prev_inst["iou"][prev_k]
                        curr_iou = inference_predictions[i]["instances"][j]["iou"][k]
                        if curr_iou > prev_iou:
                            d[gt_id] = (inference_predictions[i]["instances"][j], k)
        return d

    def get_annotation_data(self, instance):
        """Annotation data for backend interface.
        # TODO(ethan): zero indexing since used to be for diversity
        """
        image_id = self.image_id(instance)
        bbox = [int(x) for x in instance["bbox"]]  # x, y, w, h
        image_size = self.image_id_to_image_size[image_id]
        bbox = get_expanded_bounding_box(bbox, image_size, perc_exp=0.4)
        polygons = get_polygons_from_mask(
            pycocomask.decode(instance["segmentation"][0]),
            thresh=1
        )
        data = {
            "image_filename": self.image_id_to_image_filename[image_id],
            "image_id": image_id,
            "bbox": bbox,
            "segmentation": instance["segmentation"][0],
            "polygons": polygons
        }
        if "gt_segmentation" in instance:
            data["gt_segmentation"] = instance["gt_segmentation"][0]
            try:
                data["gt_polygons"] = get_polygons_from_mask(
                    pycocomask.decode(instance["gt_segmentation"][0]),
                    thresh=1)
            except:
                data["gt_polygons"] = polygons
        return data

    def get_image_data(self, image_id):
        """Image data for backend interface.
        """
        data = {
            "image_filename": self.image_id_to_image_filename[image_id]
        }
        # TODO(ethan): need to implement this!
        data["segmentations"] = []
        data["gt_segmentations"] = []
        return data
