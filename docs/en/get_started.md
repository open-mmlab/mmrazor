## Prerequisites

- Linux

- Python 3.7+

- PyTorch 1.5+

- CUDA 9.2+ (If you build PyTorch from source, CUDA 9.0 is also compatible)

- GCC 5+

- [MMCV](https://mmcv.readthedocs.io/en/latest/#installation)

**Note:** You need to run `pip uninstall mmcv` first if you have mmcv installed. If mmcv and mmcv-full are both installed, there will be `ModuleNotFoundError`.

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

    `E.g.2` If you have CUDA 9.2 installed under `/usr/local/cuda` and would like to install PyTorch 1.5.1, you need to install the prebuilt PyTorch with CUDA 9.2.

    ```Bash
    conda install pytorch==1.5.1 torchvision==0.6.1 cudatoolkit=9.2 -c pytorch
    ```

    If you build PyTorch from source instead of installing the prebuilt package, you can use more CUDA versions such as 9.0.

### Install MMRazor

It is recommended to install MMRazor with [MIM](https://github.com/open-mmlab/mim), which automatically handles the dependencies of OpenMMLab projects, including mmcv and other python packages.

```Bash
pip install openmim
mim install mmrazor
```

Or you can still install MMRazor manually

1. Install mmcv-full, we recommend you to install the pre-build package as below.

    ```shell
    # pip install mmcv-full -f https://download.openmmlab.com/mmcv/dist/{cu_version}/{torch_version}/index.html
    pip install mmcv-full -f https://download.openmmlab.com/mmcv/dist/cu102/torch1.10.0/index.html
    ```

    mmcv-full is only compiled on PyTorch 1.x.0 because the compatibility usually holds between 1.x.0 and 1.x.1. If your PyTorch version is 1.x.1, you can install mmcv-full compiled with PyTorch 1.x.0 and it usually works well.

    ```
    # We can ignore the micro version of PyTorch
    pip install mmcv-full -f https://download.openmmlab.com/mmcv/dist/cu102/torch1.10/index.html
    ```

    See [here](https://mmcv.readthedocs.io/en/latest/get_started/installation.html) for different versions of MMCV compatible to different PyTorch and CUDA versions.
    Optionally you can choose to compile mmcv from source by the following command

    ```shell
    git clone https://github.com/open-mmlab/mmcv.git
    cd mmcv
    MMCV_WITH_OPS=1 pip install -e .  # package mmcv-full will be installed after this step
    cd ..
    ```

2. Install MMRazor.

    You can simply install mmrazor with the following command:

    ```Bash
    pip install mmrazor
    ```

    ​or:

    ```Bash
    pip install git+https://github.com/open-mmlab/mmrazor.git # install the master branch
    ```

    ​Instead, if you would like to install MMRazor in `dev` mode, run following:

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
pip install mmcv-full -f https://download.openmmlab.com/mmcv/dist/cu102/torch1.10.0/index.html
# install mmrazor
git clone https://github.com/open-mmlab/mmrazor.git
cd mmrazor
pip install -v -e .
```
