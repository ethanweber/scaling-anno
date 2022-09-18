"""
Keep track of the args used for the repo.

"""


from detectron.datasets import ICCV_PLACES_FILENAMES, ICCV_PLACES_CLASS_INDICES
from src.utils import arg_as_list, pjoin
import argparse


def get_parser():
    """Return the parser.
    """

    export_places_path = None
    # export_places_path = pjoin(get_project_root(), "rounds/iccv/places_ade_1k_COCO")

    places_dataset_prefix = None

    # step_choices = ["inference", "aggregate", "merge", "merge_datasets", "cluster", "mturk", "nn", "iou_and_score"]

    parser = argparse.ArgumentParser(description="Process the Places data.")
    parser.add_argument('--step',
                        type=str)  # choices=step_choices)
    # For CVPR we used [0, 81], but for ICCV we are using [0, 11] where 11 and 81 are the same.
    # See detectron/datasets.py for more details.
    parser.add_argument('--place_indices',
                        type=arg_as_list,
                        default=list(range(0, 11)),
                        help="[0, N], where N should always be the validation set")
    parser.add_argument('--category_indices',
                        type=arg_as_list,
                        default=[], # ICCV_PLACES_CLASS_INDICES # list(range(0, 100)),
                        help="the ADE classes")
    parser.add_argument('--places_round',
                        type=int,
                        default=0)
    parser.add_argument('--export_places_path',
                        type=str,
                        default=export_places_path)
    parser.add_argument('--config',
                        type=str,
                        default="/data/vision/torralba/scratch/dimpapa/git_iccv21/scaleade/detectron/outputs/ade_1k_ImageNet/config.yaml")
    parser.add_argument('--model_weights',
                        type=str,
                        default="/data/vision/torralba/scratch/dimpapa/git_iccv21/scaleade/detectron/outputs/ade_1k_ImageNet/model_final.pth")
    parser.add_argument('--places_dataset_prefix',
                        type=str,
                        default=None)
    return parser


def get_parsed_args(version=None):
    parser = get_parser()
    args = parser.parse_args()
    assert args.export_places_path is not None
    assert args.places_dataset_prefix is not None

    if version:
        assert version in ["ade", "places", "coco"]
        if version == "ade" and len(args.category_indices) == 0:
            args.category_indices = list(range(0, 80))
        if version == "places" and len(args.category_indices) == 0:
            args.category_indices = ICCV_PLACES_CLASS_INDICES
        if version == "coco" and len(args.category_indices) == 0:
            args.category_indices = list(range(0, 90))

    if len(args.place_indices) > 0:
        args.place_indices = [int(x) for x in args.place_indices]
    else:
        # trick to avoid using splits. then don't use the prefix_ notation
        args.place_indices = [-1]

    return args
