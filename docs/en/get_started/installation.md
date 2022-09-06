# Installation

## Prepare Environment

Create a conda virtual environment and activate it.

```Python
conda create -n openmmlab python=3.7 -y
conda activate openmmlab
```

Install PyTorch and torchvision following the [official instructions](https://pytorch.org/).

```{note}
Make sure that your compilation CUDA version and runtime CUDA version match. You can check the supported CUDA version for precompiled packages on the [PyTorch website](https://pytorch.org/). If you build PyTorch from source instead of installing the prebuilt package, you can use more CUDA versions such as 9.0.
```

## Customize Installation

It is recommended to install MMRazor with [MIM](https://github.com/open-mmlab/mim), which automatically handles the dependencies of OpenMMLab projects, including mmcv and other python packages.

Or you can still install MMRazor manually

1. Install mmcv.

You can install mmcv with MIM, pip, or build it from source.

- Install mmcv with MIM (recommend).

```Python
pip install openmim
mim install 'mmcv>=2.0.0rc1'
```

- Install mmcv with pip.

```Python
pip install 'mmcv>=2.0.0rc1' -f https://download.openmmlab.com/mmcv/dist/{cu_version}/{torch_version}/index.html
```

Please replace `{cu_version}` and `{torch_version}` in the url to your desired one. For example, to install the latest `mmcv` with `CUDA 10.2` and `PyTorch 1.10.0`, use the following command:

```Python
pip install 'mmcv>=2.0.0rc1' -f https://download.openmmlab.com/mmcv/dist/cu102/torch1.10.0/index.html
```

See [here](https://github.com/open-mmlab/mmcv#installation) for different versions of MMCV compatible to different PyTorch and CUDA versions.

- Build mmcv from source.

```bash
MMCV_WITH_OPS=0 pip install -e . -v
# install mmcv-lite, do not compile operators
MMCV_WITH_OPS=1 pip install -e . -v
# install mmcv (originally called mmcv-full), compile operators
pip install -e . -v
# install mmcv with compiled operators，
```

- For windows platform, try `set MMCV_WITH_OPS=1` instead.

2. Install MMEngine.

You can install mmengine with MIM or build it from source.

- Install MMEngine with MIM.

```bash
pip install openmim
mim install mmengine
```

- Compile MMEngine from source.

```Python
git clone https://github.com/open-mmlab/mmengine.git
cd mmengine
pip install -v -e .
```

3. Install MMRazor.

If you would like to install MMRazor in `dev` mode, run following:

```Python
git clone https://github.com/open-mmlab/mmrazor.git
cd mmrazor
git fetch origin
git checkout -b dev-1.x origin/dev-1.x
# The new version is released in branch ``dev-1.x``
pip install -v -e .
# "-v" means verbose, or more output
# "-e" means installing a project in editable mode,
# thus any local modifications made to the code will take effect without reinstallation.
```

```{note}
When MMRazor is installed on `dev` mode, any local modifications made to the code will take effect without the need to reinstall it.
```

## A from-scratch Setup Script

```Python
conda create -n openmmlab python=3.7 -y
conda activate openmmlab

conda install pytorch torchvision cudatoolkit=10.2 -c pytorch
# install the latest mmcv
pip install 'mmcv>=2.0.0rc1' -f https://download.openmmlab.com/mmcv/dist/cu102/torch1.10.0/index.html
# install mmrazor
git clone https://github.com/open-mmlab/mmrazor.git
cd mmrazor
git fetch origin
git checkout -b dev-1.x origin/dev-1.x
pip install -v -e .
```

## Install Other Libraries

MMRazor can easily collaborate with other OpenMMLab libraries. MMRazor requires the use of other libraries for different tasks. For example, `MMClassification` is required for image classification tasks, `MMDetection` for object detection, and `MMSegmentation` for semantic segmentation.

We provide the installation of the above three libraries using `MIM`.

```bash
pip install openmim
# mmcv is required for all libraries
mim install 'mmcv>=2.0.0rc1'
# install mmcls
mim install 'mmcls>=1.0.0rc0'
# install mmdet
mim install 'mmdet>=3.0.0rc0'
# install mmseg
mim install 'mmseg>=1.0.0rc0'
```

```{note}
Not all of above libraries are required by MMRazor. Please install according to your requirements.
```
