# Copyright (c) OpenMMLab. All rights reserved.
"""impl folder is an experimental file structure to store algorithm
implementations.

Previous file structure splits the files of an algorithm into different folders
according to the types of these files. It may make it hard to understand an
algorithm. So we add the impl folder, where all files of an algorithm are
stored in one folder. As this structure is experimental, it may change rapidly.
"""

from . import pruning  # noqa

__all__ = ['pruning']
