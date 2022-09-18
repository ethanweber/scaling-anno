# scaling-anno
Code and data for "Scaling up instance annotation via label propagation" in ICCV 2021. The goal of this project is the efficiently annotate a set of images with instance segmentation labels.

> This repo is a WOP and won't quite work out the box, but we are open-sourcing as-is for people to reference our code if trying to implement our pipeline method to scale up instance segmentation. Note that the pipeline has a lot of moving parts and some of the documentation here may be outdated. You can find our final created dataset JSON file [here](https://drive.google.com/drive/folders/1T8z-1jxWZclkSoWSIjgUNVzYmafy6LWr?usp=sharing) in the COCO format but this will require access to the Places images to use.

#### <hr> Installation

```
# clone the repo
git clone --recurse-submodules git@github.com:ethanweber/scaling-anno.git

# install dependencies and configure ipykernel for use with jupyter notebooks
conda create -n scaleade python=3.8.2
conda activate scaleade
pip install -r requirements.txt
python -m ipykernel install --user --name "scaling-anno" --display-name "scaling-anno"
cd external/
python -m pip install -e fvcore
cd external/goat
python setup.py develop
```

#### <hr> Repo structure

Below we outline the general structure of the repo. Note that some files or folders may not be mentioned, but this does not necessarily mean that they are not important.

```
- data/                                 # SYMBOLIC LINKS TO NEEDED DATA # TODO: populate this
    |- ade/
        |- images/
            |- training/                # symlink to ADE training images
            |- validation/              # symlink to ADE validation images
        |- train.json                   # train annotations
        |- val.json                     # val annotations
- detectron/                            # DETECTION CODE BEFORE CLUSTERING
    |- configs/                         # specify dataset, network, etc.
    |- train_net.py                     # code to train detection network
    |- README.md                        # instructions for detection model
- experiments/                          # EXPERIMENTS FROM THE PAPER
- rounds/                               # WHERE WEIGHTS AND ANNOTATIONS ARE STORED
- src/                                  # PYTHON CODE USED THROUGHOUT PROJECT
- web/                                  # WEB VISUALIZATION CODE
    |- pages/                           # HTML interfaces
    |- static/                          # JavaScript files
    |- backend.py                       # server for access to backend annotation data
    |- server.py                        # server for access to frontent interfaces at URL
```

#### <hr> Procedure for running the code

1. **Detection**: The first step of our pipeline is to detect instances. We run an instance segmentation network on a pool of images, and we save the results.

    See additional details at [detectron/README.md](detectron/README.md).

2. **Clustering, search, and label propagation**: The purpose of this step is create a class-specific hierarchical clustering.

    See experiments/run_ade.py for more details. Note that data is populated into a `rounds/` folder. The naming is because our pipeline could theoretically be performed for multiple rounds.

    To, visualize the result of clustering, see our interface in `web/`.
    ```
    cd web/
    python backend.py
    python server.py
    ```

3. **Annotation**: With detections clustered and trees created, we perform annotation. In this step, we use a binary annotation task to efficiently find high and low quality detections. We aren't including the MTurk pipeline in this code release, but following the structure of `run_ade.py` provides the necessary framework.

#### <hr> Notebooks

```
notebooks/
    |- TODO.ipynb               # run binary task
```

#### <hr> Helpful commands

```
# start tensorboard
tensorboard --samples_per_plugin scalars=100,images=10 --port 8880 --bind_all --logdir path/to/folder/
```

#### <hr> Download the Places annotations

```
TODO
```
