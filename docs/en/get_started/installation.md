# Installation

## Prepare Environment

Create a conda virtual environment and activate it.

```Python
conda create -n openmmlab python=3.7 -y
conda activate openmmlab
```

Install PyTorch and torchvision following the [official instructions](https://pytorch.org/).

Note: Make sure that your compilation CUDA version and runtime CUDA version match. You can check the supported CUDA version for precompiled packages on the [PyTorch website](https://pytorch.org/).

`E.g.1` If you have CUDA 10.2 installed under `/usr/local/cuda` and would like to install PyTorch 1.10, you need to install the prebuilt PyTorch with CUDA 10.2.

```Python
conda install pytorch torchvision torchaudio cudatoolkit=10.2 -c pytorch
```

`E.g.2` If you have CUDA 9.2 installed under `/usr/local/cuda` and would like to install PyTorch 1.5.1, you need to install the prebuilt PyTorch with CUDA 9.2.

```Python
conda install pytorch==1.5.1 torchvision==0.6.1 cudatoolkit=9.2 -c pytorch
```

- If you build PyTorch from source instead of installing the prebuilt package, you can use more CUDA versions such as 9.0.

## Customize Installation

It is recommended to install MMRazor with [MIM](https://github.com/open-mmlab/mim), which automatically handles the dependencies of OpenMMLab projects, including mmcv and other python packages.

```Python
pip install openmim
mim install git+https://github.com/open-mmlab/mmrazor.git@1.0.0rc0
```

Or you can still install MMRazor manually

1. Install mmcv.

```Python
pip install 'mmcv>=2.0.0rc1' -f https://download.openmmlab.com/mmcv/dist/{cu_version}/{torch_version}/index.html
```

Please replace `{cu_version}` and `{torch_version}` in the url to your desired one. For example, to install the latest `mmcv` with `CUDA 10.2` and `PyTorch 1.10.0`, use the following command:

```Python
pip install 'mmcv>=2.0.0rc1' -f https://download.openmmlab.com/mmcv/dist/cu102/torch1.10.0/index.html
```

See [here](https://github.com/open-mmlab/mmcv#installation) for different versions of MMCV compatible to different PyTorch and CUDA versions.

Optionally, you can compile mmcv from source.

```
MMCV_WITH_OPS=0 pip install -e . -v
# install mmcv-lite, do not compile operators
MMCV_WITH_OPS=1 pip install -e . -v
# install mmcv (originally called mmcv-full), compile operators
pip install -e . -v
# install mmcv with compiled operators，
```

2. Install MMEngine.

Compile MMEngine from source.

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
git checkout -b 1.0.0rc0 origin/1.0.0rc0
# The new version is released in branch ``1.0.0rc0``
pip install -v -e .
# "-v" means verbose, or more output
# "-e" means installing a project in editable mode,
# thus any local modifications made to the code will take effect without reinstallation.
```

**Note:**

- When MMRazor is installed on `dev` mode, any local modifications made to the code will take effect without the need to reinstall it.

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
git checkout -b 1.0.0rc0 origin/1.0.0rc0
pip install -v -e .
```
