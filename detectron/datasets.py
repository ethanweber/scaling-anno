"""
Register datasets in this file.
"""

import os
import numpy as np
from detectron2.data.datasets import register_coco_instances
# TODO(ethan): fix this with a util function
PROJECT_ROOT = "/data/vision/torralba/scratch/ethanweber/scaleade/"

register_coco_instances("ade_train",
                        {},
                        os.path.join(PROJECT_ROOT, "data/ade/train.json"),
                        os.path.join(PROJECT_ROOT, "data/ade/images/"))

register_coco_instances("ade_val",
                        {},
                        os.path.join(PROJECT_ROOT, "data/ade/val.json"),
                        os.path.join(PROJECT_ROOT, "data/ade/images/"))
