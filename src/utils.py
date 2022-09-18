import json
from pycocotools import mask as pycocomask
import numpy as np
import cv2
from pathlib import Path
import os
import GPUtil
import ast
from os.path import join as pjoin
import argparse

def get_project_root() -> Path:
    return Path(__file__).parent.parent.absolute()


def get_absolute_path(path):
    """
    Returns the full, absolute path.
    Relative paths are assumed to start at the repo directory.
    """
    absolute_path = path
    if absolute_path[0] != "/":
        absolute_path = os.path.join(
            get_project_root(), absolute_path
        )
    return absolute_path


def get_polygon_from_mask(mask, thresh=10, largest_only=True):
    """Get the polygon from a binary mask.
    Inputs:
        mask: (h, w) 0/1
        threshold: threshold is in pixel euclidean distance
    Return:
        polygon (np array of x,y points)
    """
    # https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_contours/py_contours_begin/py_contours_begin.html
    original_contours, hierarchy = cv2.findContours(
        mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)  # RETR_EXTERNAL if only care about outer contours

    h, w = mask.shape
    lengths = np.zeros(64)  # assume no more than 64 polygons
    polygon = []
    i = 0

    largest_polygon = None
    largest_polygon_area = 0
    largest_polygon_length = None

    for original_contour in original_contours:

        original_polygon = np.squeeze(original_contour)
        temp_polygon = [original_polygon[0]]
        for point in original_polygon[1:]:
            dist = np.linalg.norm(point - temp_polygon[-1])
            if dist >= thresh:
                temp_polygon.append(point)
        temp_polygon = np.array(temp_polygon)
        if len(temp_polygon) < 3:  # can't have a polygon less than 3 points
            continue

        contour_area = cv2.contourArea(original_contour)
        if contour_area > largest_polygon_area:
            largest_polygon_area = contour_area
            largest_polygon = temp_polygon
            largest_polygon_length = len(largest_polygon)

        lengths[i] = len(temp_polygon)
        i += 1
        polygon.append(temp_polygon)
    if len(polygon) == 0:
        raise ValueError("Must have polygon array of nonzero length.")
    polygon = np.vstack(polygon)

    if largest_only:
        new_lengths = np.zeros(64)
        new_lengths[0] = largest_polygon_length
        return largest_polygon, new_lengths

    return polygon, lengths


def load_from_json(filename: str):
    assert filename.endswith(".json")
    with open(filename, "r") as f:
        return json.load(f)


def write_to_json(filename: str, content: dict):
    assert filename.endswith(".json")
    with open(filename, "w") as f:
        json.dump(content, f)


def get_formatted_mask(mask):
    seg = pycocomask.encode(np.asfortranarray(mask.astype("uint8")))
    seg['counts'] = seg['counts'].decode('ascii')
    return seg


# Functon to return polygons from rle encoded segment.
def get_polygons_from_rle(rle):
    mask = pycocomask.decode(rle)
    return get_polygons_from_mask(mask)


def get_polygons_from_mask(mask, thresh=1):
    try:
        points, lengths = get_polygon_from_mask(mask, thresh=thresh, largest_only=False)

        points = points.tolist()
        lengths = lengths.tolist()

        polygons = []
        curr_length = 0
        for length in lengths:
            if int(length) == 0:
                break
            polygons.append(points[curr_length:curr_length + int(length)])
            curr_length += int(length)
    except:
        polygons = []
    return polygons


def get_mask_from_polygons(polygons,
                           height=513,
                           width=513):
    # create the segmentation, in the full image
    image = np.zeros((height, width, 3), np.uint8)
    contours = []
    for polygon in polygons:
        points = np.array(polygon, dtype=np.uint32)
        points[:, 0] = points[:, 0]
        points[:, 1] = points[:, 1]
        contours.append(np.expand_dims(points, axis=1).astype(np.int32))
    im = cv2.drawContours(image, contours, -1, (255, 255, 255), -1)
    im = (im[:, :, 0] / 255).astype("uint8")
    return im


def get_iou(mask1, mask2):
    inter = np.logical_and(mask1, mask2)
    union = np.logical_or(mask1, mask2)
    return np.sum(inter) / np.sum(union)


def get_iou_from_rles(rle1, rle2):
    mask1 = pycocomask.decode(rle1)
    mask2 = pycocomask.decode(rle2)
    return get_iou(mask1, mask2)


def get_class_name_to_class_idx_mapping():
    # TODO(ethan): move these mapping somewhere to avoid redundant code
    category_info = load_from_json("/data/vision/torralba/scaleade/data/Places/categories100.json")
    class_name_to_class_idx = {}
    for x in category_info:
        # TODO(ethan): deal with this off by 1! especially with COCO datasets...
        class_name_to_class_idx[x["name"]] = x["id"] - 1
    return class_name_to_class_idx


def get_three_channel_image(image):
    """
    If single channel (2 dim), convert to a three dimensional image and return.
    TODO(ethan): add support for (h, w, 1)
    """
    im = image.copy()
    # if binary, convert to rgb.
    # if rgb, leave as is
    if len(im.shape) == 2:
        im = np.stack((im, im, im), axis=2) * 255
    return im


def draw_polygon_on_image(image, polygon, mask_color=(0, 255, 0), radius=4, point_color=(255, 0, 0)):
    """Draws polygon points on image.
    """

    im = get_three_channel_image(image)

    # im[:, :, 0][im[:, :, 0] == 255] = mask_color[0]
    # im[:, :, 1][im[:, :, 1] == 255] = mask_color[1]
    # im[:, :, 2][im[:, :, 2] == 255] = mask_color[2]

    for point in polygon:
        x, y = point  # TODO(ethan): make sure this is an integer
        try:
            im = cv2.circle(im, (x, y), radius, point_color, -1)
        except:
            pass
    return im


def get_bbox_iou(in_boxA, in_boxB, mode="xyxy"):
    """
    #Determine the (x, y)-coordinates of the intersection rectangle
    https://gist.github.com/meyerjo/dd3533edc97c81258898f60d8978eddc
    """

    if mode == "xyxy":
        boxA = in_boxA
        boxB = in_boxB
    elif mode == "xywh":
        boxA = in_boxA
        boxA[2] += boxA[0]  # w + x
        boxA[3] += boxA[1]  # h + y
        boxB = in_boxB
        boxB[2] += boxB[0]
        boxB[3] += boxB[1]
    else:
        raise ValueError("Mode not implemented.")

    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    # compute the area of intersection rectangle
    interArea = abs(max((xB - xA, 0)) * max((yB - yA), 0))
    if interArea == 0:
        return 0
    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = abs((boxA[2] - boxA[0]) * (boxA[3] - boxA[1]))
    boxBArea = abs((boxB[2] - boxB[0]) * (boxB[3] - boxB[1]))

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)

    # return the intersection over union value
    return iou


def get_image_with_mask_overlayed(image, mask, color, alpha=0.5):
    """Apply the given mask to the image.
    Args:
        color (tuple): (r, g, b) with range (0, 1)
    """
    im = image.copy()
    for c in range(3):
        im[:, :, c] = np.where(
            mask == 1, im[:, :, c] * (1 - alpha) + alpha * color[c] * 255, im[:, :, c])
    return im


def get_gpus(maxLoad=0.5):
    """Returns the available GPUs."""
    deviceIDs = GPUtil.getAvailable(
        order='first',
        limit=8,
        maxLoad=maxLoad,
        maxMemory=0.5,
        includeNan=False,
        excludeID=[],
        excludeUUID=[])
    return deviceIDs


def make_dir(filename_or_folder):
    """Make the directory for either the filename or folder.
    Note that filename_or_folder currently needs to end in / for it to be recognized as a folder.
    """
    folder = os.path.dirname(filename_or_folder)
    if not os.path.exists(folder):
        try:
            os.makedirs(folder)
        except Exception as e:
            print(f"Couldn't create folder: {folder}. Maybe due to a parallel process?")
            print(e)


def get_chunks(lst, n):
    """Yield n successive chunks from lst."""
    size = len(lst) // n
    chunks = []
    for i in range(0, len(lst), size):
        chunks.append(lst[i:i + size])
    return chunks


def arg_as_list(s):
    v = ast.literal_eval(s)
    if type(v) is not list:
        raise argparse.ArgumentTypeError("Argument \"%s\" is not a list" % (s))
    return v
