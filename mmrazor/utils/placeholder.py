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

    def raise_import_error(package_name):
        raise ImportError(
            f'`{package_name}` is not installed properly, plz check.')

    class PlaceHolder():

        def __init__(self) -> None:
            raise_import_error(string)

    return PlaceHolder
