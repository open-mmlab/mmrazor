# Copyright (c) OpenMMLab. All rights reserved.
def get_placeholder(string):

    class PlaceHolder:

        def __init__(self, *args, **kwargs) -> None:
            raise ImportError(
                f'`{string}` is not installed properly, plz check.')

    return PlaceHolder
