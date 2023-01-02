import os, sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import itertools

import global_utils
# from PlainNet import basic_blocks, super_blocks, SuperResKXKX, SuperResK1KXK1
from PlainNet import basic_blocks

seach_space_block_type_list_list = [
    [basic_blocks.SuperResK1K3K1, basic_blocks.SuperResK1K5K1, basic_blocks.SuperResK1K7K1],
    [basic_blocks.SuperResK3K3, basic_blocks.SuperResK5K5, basic_blocks.SuperResK7K7],
]

__block_type_round_channels_base_dict__ = {
    basic_blocks.SuperResK3K3: 8,
    basic_blocks.SuperResK5K5: 8,
    basic_blocks.SuperResK7K7: 8,
    basic_blocks.SuperResK1K3K1: 8, 
    basic_blocks.SuperResK1K5K1: 8, 
    basic_blocks.SuperResK1K7K1: 8,
}

__block_type_min_channels_base_dict__ = {
    basic_blocks.SuperResK3K3: 8,
    basic_blocks.SuperResK5K5: 8,
    basic_blocks.SuperResK7K7: 8,
    basic_blocks.SuperResK1K3K1: 8,
    basic_blocks.SuperResK1K5K1: 8,
    basic_blocks.SuperResK1K7K1: 8,
}


def get_select_student_channels_list(out_channels):
    the_list = [out_channels * 2.5, out_channels * 2, out_channels * 1.5, out_channels * 1.25,
                out_channels,
                out_channels / 1.25, out_channels / 1.5, out_channels / 2, out_channels / 2.5]
    the_list = [min(2048, max(8, x)) for x in the_list]
    the_list = [global_utils.smart_round(x, base=8) for x in the_list]
    the_list = list(set(the_list))
    the_list.sort(reverse=True)
    return the_list


def get_select_student_sublayers_list(sub_layers):
    the_list = [sub_layers,
                sub_layers + 1, sub_layers + 2,
                sub_layers - 1, sub_layers - 2, ]
    the_list = [max(0, round(x)) for x in the_list]
    the_list = list(set(the_list))
    the_list.sort(reverse=True)
    return the_list


def gen_search_space(block_list, block_id):
    the_block = block_list[block_id]
    student_blocks_list_list = []

    if isinstance(the_block, basic_blocks.SuperConvKXBNRELU):
        student_blocks_list = []
        student_out_channels_list = get_select_student_channels_list(the_block.out_channels)
        for student_out_channels in student_out_channels_list:
            tmp_block_str = type(the_block).__name__ + '({},{},{},1)'.format(
                the_block.in_channels, student_out_channels, the_block.stride)
            student_blocks_list.append(tmp_block_str)
        pass
        student_blocks_list = list(set(student_blocks_list))
        assert len(student_blocks_list) >= 1
        student_blocks_list_list.append(student_blocks_list)
    else:
        for student_block_type_list in seach_space_block_type_list_list:
            student_blocks_list = []
            student_out_channels_list = get_select_student_channels_list(the_block.out_channels)
            student_sublayers_list = get_select_student_sublayers_list(sub_layers=the_block.sub_layers)
            student_bottleneck_channels_list = get_select_student_channels_list(the_block.bottleneck_channels)
            for student_block_type in student_block_type_list:
                for student_out_channels, student_sublayers, student_bottleneck_channels in itertools.product(
                        student_out_channels_list, student_sublayers_list, student_bottleneck_channels_list):

                    # filter smallest possible channel for this block type
                    min_possible_channels = __block_type_round_channels_base_dict__[student_block_type]
                    channel_round_base = __block_type_round_channels_base_dict__[student_block_type]
                    student_out_channels = global_utils.smart_round(student_out_channels, channel_round_base)
                    student_bottleneck_channels = global_utils.smart_round(student_bottleneck_channels,
                                                                           channel_round_base)

                    if student_out_channels < min_possible_channels or student_bottleneck_channels < min_possible_channels:
                        continue
                    if student_sublayers <= 0:  # no empty layer
                        continue
                    tmp_block_str = student_block_type.__name__ + '({},{},{},{},{})'.format(
                        the_block.in_channels, student_out_channels, the_block.stride, student_bottleneck_channels,
                        student_sublayers)
                    student_blocks_list.append(tmp_block_str)
                pass
                student_blocks_list = list(set(student_blocks_list))
                assert len(student_blocks_list) >= 1
                student_blocks_list_list.append(student_blocks_list)
            pass
        pass  # end for student_block_type_list in seach_space_block_type_list_list:
    pass
    return student_blocks_list_list
