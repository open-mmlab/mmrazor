# Copyright (c) OpenMMLab. All rights reserved.
def get_placeholder(string: str) -> object:
    """Get placeholder instance which can avoid raising errors when down-stream
    dependency is not installed properly.

    Args:
        string (str): the dependency's name, i.e. `mmcls`

    Raises:
        ImportError: raise it when the dependency is not installed properly.

    Returns:
        object: PlaceHolder instance.
    """

    class PlaceHolder:

        def __init__(self, *args, **kwargs) -> None:
            raise ImportError(
                f'`{string}` is not installed properly, plz check.')

    return PlaceHolder
