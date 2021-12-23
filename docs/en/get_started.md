## Prerequisites

- Linux

- Python 3.7+

- PyTorch 1.5+

- CUDA 9.2+ (If you build PyTorch from source, CUDA 9.0 is also compatible)

- GCC 5+

- [MMCV](https://mmcv.readthedocs.io/en/latest/#installation)

**Note:** You need to run `pip uninstall mmcv` first if you have mmcv installed. If mmcv and mmcv-full are both installed, there will be `ModuleNotFoundError`

## Installation

### Prepare environment

1. Create a conda virtual environment and activate it.

    ```Bash
    conda create -n openmmlab python=3.7 -y
    conda activate openmmlab
    ```

2. Install PyTorch and torchvision following the [official instructions](https://pytorch.org/).

    Note: Make sure that your compilation CUDA version and runtime CUDA version match. You can check the supported CUDA version for precompiled packages on the [PyTorch website](https://pytorch.org/).

    `E.g.1` If you have CUDA 10.2 installed under `/usr/local/cuda` and would like to install PyTorch 1.10, you need to install the prebuilt PyTorch with CUDA 10.2.

    ```Bash
    conda install pytorch torchvision torchaudio cudatoolkit=10.2 -c pytorch
    ```

    `E.g.2` If you have CUDA 9.2 installed under `/usr/local/cuda` and would like to install PyTorch 1.3.1, you need to install the prebuilt PyTorch with CUDA 9.2.

    ```Bash
    conda install pytorch==1.5.1 torchvision==0.6.1 cudatoolkit=9.2 -c pytorch
    ```

    If you build PyTorch from source instead of installing the prebuilt package, you can use more CUDA versions such as 9.0.

### Install MMRazor

It is recommended to install MMRazor with [MIM](https://github.com/open-mmlab/mim), which automatically handles the dependencies of OpenMMLab projects, including mmcv and other python packages.

```Bash
pip install openmin
min install mmrazor
```

Or you can still install MMRazor manually

1. Install mmcv-full.

    ```Bash
    pip install mmcv-full -f https://download.openmmlab.com/mmcv/dist/{cu_version}/{torch_version}/index.html
    ```

    Please replace `{cu_version}` and `{torch_version}` in the url to your desired one. For example, to install the latest `mmcv-full` with `CUDA 10.2` and `PyTorch 1.10.0`, use the following command:

    ```Bash
    pip install mmcv-full -f https://download.openmmlab.com/mmcv/dist/cu102/torch1.10.0/index.html
    ```

    See [here](https://github.com/open-mmlab/mmcv#installation) for different versions of MMCV compatible to different PyTorch and CUDA versions.

    Optionally, you can compile mmcv from source if you need to develop both mmcv and mmdet. Refer to the [guide](https://github.com/open-mmlab/mmcv#installation) for details.

2. Install MMRazor.

    You can simply install mmrazor with the following command:

    ```Bash
    pip install mmrazor
    ```

    ​    or:

    ```Bash
    pip install git+https://github.com/open-mmlab/mmrazor.git # install the master branch
    ```

    ​    Instead, if you would like to install MMRazor in `dev` mode, run following:

    ```Bash
    git clone https://github.com/open-mmlab/mmrazor.git
    cd mmrazor
    pip install -v -e .  # or "python setup.py develop"
    ```

    **Note:**

    - When MMRazor is installed on `dev` mode, any local modifications made to the code will take effect without the need to reinstall it.
    - Currently, running `pip install -v -e .` will install `mmcls`, `mmdet`, `mmsegmentation`. We will work on minimum runtime requirements in future.

### A from-scratch setup script

```Bash
conda create -n openmmlab python=3.7 -y
conda activate openmmlab

conda install pytorch torchvision cudatoolkit=10.2 -c pytorch
# install the latest mmcv
pip install mmcv-full -f https://download.openmmlab.com/mmcv/dist/cu102/torch1.9.0/index.html
# install mmrazor
git clone https://github.com/open-mmlab/mmrazor.git
cd mmrazor
pip install -v -e .
```
