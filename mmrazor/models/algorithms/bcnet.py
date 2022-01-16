# Copyright (c) OpenMMLab. All rights reserved.
from heapq import *  # noqa: F401,F403

from mmrazor.models.builder import ALGORITHMS
from mmrazor.models.utils import add_prefix
from .autoslim import AutoSlim


class LossRecord(object):

    def __init__(self, loss, subnet_l, subnet_r):
        super().__init__()
        self.loss = loss
        self.subnet_l = subnet_l
        self.subnet_r = subnet_r

    def __lt__(self, other):
        return self.loss > other.loss


@ALGORITHMS.register_module()
class BCNet(AutoSlim):

    def __init__(self, loss_rec_num, use_complementary=True, **kwargs):
        super(BCNet, self).__init__(**kwargs)
        self.loss_rec = []
        self.loss_rec_num = loss_rec_num

        # Whether to use the complementary subnet of a sampled subnet in this
        # iteration. If ``use_complementary`` is True, the
        # ``_complementary_switch`` will always be False during training.
        self._complementary_switch = False
        self.use_complementary = use_complementary
        if use_complementary:
            self._subnet_dict_l_comp = None  # the complementary left subnet

    def get_space_flops(self):
        flops = {space_id: 0 for space_id in self.pruner.channel_spaces.keys()}
        for name, module in self.architecture.model.named_modules():
            space_id = self.pruner.get_space_id(name)
            if space_id not in self.pruner.channel_spaces:
                continue

            flops[space_id] += module.__flops__
        return flops

    def record_loss(self, loss, subnet_left, subnet_right):
        rec = LossRecord(loss, subnet_left, subnet_right)
        if len(self.loss_rec) < self.loss_rec_num:
            heappush(self.loss_rec, rec)  # noqa: F405
        else:
            heappushpop(self.loss_rec, rec)  # noqa: F405

    def train_step(self, data, optimizer):
        if self.retraining:
            return super().train_step(data, optimizer)

        optimizer.zero_grad()

        losses = dict()
        if not self._complementary_switch:
            subnet_dict_l = self.pruner.sample_subnet()
            subnet_dict_r = self.pruner.reverse_subnet(subnet_dict_l)
            if self.use_complementary:
                # get the corresponding complementary subnet
                self._subnet_dict_l_comp = \
                    self.pruner.get_complementary_subnet(subnet_dict_l)
                self._complementary_switch = not self._complementary_switch
        else:
            subnet_dict_l = self._subnet_dict_l_comp
            subnet_dict_r = self.pruner.reverse_subnet(subnet_dict_l)
            self._complementary_switch = not self._complementary_switch

        for subnet_l, subnet_r in zip(subnet_dict_l.values(),
                                      subnet_dict_r.values()):
            for _, __ in zip(subnet_l[0], subnet_r[0]):
                print(_[0, 0], __[0, 0])
            assert subnet_l.sum() == subnet_r.sum()
            break

        self.pruner.set_subnet(subnet_dict_l)
        losses_l = self(**data)
        losses.update(add_prefix(losses_l, 'losses_l'))
        loss_l, _ = self._parse_losses(losses_l)
        (loss_l * 0.5).backward()

        self.pruner.set_subnet(subnet_dict_r)
        losses_r = self(**data)
        losses.update(add_prefix(losses_r, 'losses_r'))
        loss_r, _ = self._parse_losses(losses_r)
        (loss_r * 0.5).backward()

        optimizer.step()

        loss, log_vars = self._parse_losses(losses)
        self.record_loss(log_vars['loss'], subnet_dict_l, subnet_dict_r)

        outputs = dict(
            loss=loss, log_vars=log_vars, num_samples=len(data['img'].data))

        return outputs
