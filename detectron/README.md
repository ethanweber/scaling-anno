
# Detection code

This folder contains the code used for detection, which is the first step of the `ScaleADE` project. We make use of detectron2 for our detection code.

#### <hr> Relevant configs

- configs/point_rend/
    - **pointrend_rcnn_R_50_FPN_1x_ade.yaml**: Config for PointRend training on ADE train and evaluating on ADE val.

#### <hr> Loading custom datasets

See the file [datasets.py] to register a dataset with detectron2. This file must be imported before attempting to use these custom datasets with the detectron2 environment. Below are some commands to confirm your dataset has been registered correctly.

```
# visualize the dataset annotations
python visualize_data.py \
    --config-file configs/point_rend/trainA_004.yaml \
    --output-dir outputs/visualize_data/annotation \
    --source annotation

# visualize the dataloader
python visualize_data.py \
    --config-file configs/point_rend/trainA_004.yaml \
    --output-dir outputs/visualize_data/dataloader \
    --source dataloader
```

#### <hr> Training, evaluation, and inference (exporting features)

You can specify a config, the number of GPUs, and override values in the config file, as shown in the following commands.

```
# train
python train_net.py \
    --config configs/point_rend/trainA_004.yaml \
    --num-gpus 1
    # (Optionally) add the following to set batch size and learning rate.
    SOLVER.IMS_PER_BATCH 2 SOLVER.BASE_LR 0.0025

# evaluate
python train_net.py \
    --config configs/point_rend/trainA_004.yaml \
    --num-gpus 4 \
    --eval-only \
    MODEL.WEIGHTS outputs/point_rend/trainA/004/model_0018999.pth \
    TEST.BATCH_SIZE 8 \
    # MODEL.DEVICE cpu

# run with CPU
https://github.com/facebookresearch/detectron2/issues/374#issuecomment-557518709

# export features
python train_net.py \
    --config configs/point_rend/trainA_004.yaml \
    --num-gpus 4 \
    --eval-only \
    TEST.DIVERSITY.EXPORT True \
    MODEL.WEIGHTS outputs/point_rend/trainA/004/model_0018999.pth \
    TEST.BATCH_SIZE 8

# visualizing the tensorboard
tensorboard \
    --samples_per_plugin scalars=100,images=10 \
    --port 8880 \
    --bind_all \
    --logdir outputs/point_rend/trainA/004
```
