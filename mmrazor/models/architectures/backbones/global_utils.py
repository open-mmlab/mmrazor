import os
import distutils.dir_util
import pprint, ast, argparse, logging
import numpy as np
import torch

def load_py_module_from_path(module_path, module_name=None):
    if module_path.find(':') > 0:
        split_path = module_path.split(':')
        module_path = split_path[0]
        function_name = split_path[1]
    else:
        function_name = None

    if module_name is None:
        module_name = module_path.replace('/', '_').replace('.', '_')

    assert os.path.isfile(module_path)

    import importlib.util
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    any_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(any_module)
    if function_name is None:
        return any_module
    else:
        return getattr(any_module, function_name)

def mkfilepath(filename):
    distutils.dir_util.mkpath(os.path.dirname(filename))

def mkdir(dirname):
    distutils.dir_util.mkpath(dirname)

def smart_round(x, base=None):
    if base is None:
        if x > 32 * 8:
            round_base = 32
        elif x > 16 * 8:
            round_base = 16
        else:
            round_base = 8
    else:
        round_base = base

    return max(round_base, round(x / float(round_base)) * round_base)

def save_pyobj(filename, pyobj):
    mkfilepath(filename)
    the_s = pprint.pformat(pyobj, indent=2, width=120, compact=True)
    with open(filename, 'w') as fid:
        fid.write(the_s)

def load_pyobj(filename):
    with open(filename, 'r') as fid:
        the_s = fid.readlines()

    if isinstance(the_s, list):
        the_s = ''.join(the_s)

    the_s = the_s.replace('inf', '1e20')
    pyobj = ast.literal_eval(the_s)
    return pyobj

def parse_cmd_options(argv):

    parser = argparse.ArgumentParser(description='Default command line parser.')

    parser.add_argument('--evaluate_only', action='store_true', help='Only evaluation.')

    # apex support
    parser.add_argument('--apex', action='store_true', help='Mixed precision training using apex.')
    parser.add_argument('--apex_loss_scale', type=str, default='dynamic', help='loss scale for apex.')
    parser.add_argument('--apex_opt_level', type=str, default='O1')
    parser.add_argument('--fp16', action='store_true', help='Using FP16.')

    # distributed training
    parser.add_argument('--dist_mode', type=str, default='cpu', help='Distribution mode, could be cpu, single, horovod, mpi, auto.')
    parser.add_argument('--independent_training', action='store_true', help='When distributed training, use each gpu separately.')
    parser.add_argument('--world-size', default=1, type=int, help='number of nodes for distributed training')
    parser.add_argument('--rank', default=-1, type=int, help='node rank for distributed training')

    parser.add_argument('--gpu', default=None, type=int, help='GPU id to use. Used by torch.distributed package')
    parser.add_argument('--sync_bn',  action='store_true', help='Use synchronized BN.')

    parser.add_argument('--num_job_splits', default=None, type=str, help='Split jobs into multiple groups.')
    parser.add_argument('--job_id', default=None, type=int, help='The id of this job node.')

    # horovod setting
    parser.add_argument('--fp16_allreduce', action='store_true', help='use fp16 compression during allreduce.')
    parser.add_argument('--batches_per_allreduce',
                        type=int,
                        default=1,
                        help='number of batches processed locally before '
                        'executing allreduce across workers; it multiplies '
                        'total batch size.')

    # learning rate setting
    parser.add_argument('--lr', default=None, type=float, help='initial learning rate per 256 batch size')
    parser.add_argument('--target_lr', default=None, type=float, help='target learning rate')
    parser.add_argument('--lr_per_256', default=0.1, type=float, help='initial learning rate per 256 batch size')
    parser.add_argument('--target_lr_per_256', default=0.0, type=float, help='target learning rate')
    parser.add_argument('--lr_mode', default=None, type=str, help='learning rate decay mode.')
    parser.add_argument('--warmup', default=0, type=int, help='epochs for warmup.')
    parser.add_argument('--epoch_offset', default=0.0, type=float, help='Make the learning rate decaying as epochs + epoch_offset but start from epoch_offset. ')

    parser.add_argument('--lr_stage_list', default=None, type=str, help='stage-wise learning epoch list.')
    parser.add_argument('--lr_stage_decay', default=None, type=float, help='stage-wise learning epoch list.')

    # optimizer
    parser.add_argument('--optimizer', default='sgd', type=str, help='sgd optimizer')
    parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
    parser.add_argument('--adadelta_rho', default=0.9, type=float)
    parser.add_argument('--adadelta_eps', default=1e-9, type=float)

    parser.add_argument('--wd',
                        '--weight_decay',
                        default=4e-5,
                        type=float,
                        help='weight decay (default: 4e-5)',
                        dest='weight_decay')

    # training settings

    parser.add_argument('--resume', default=None, type=str, help='path to latest checkpoint (default: none)')
    parser.add_argument('--auto_resume', action='store_true', help='auto resume from latest check point')
    parser.add_argument('--load_parameters_from', default=None, type=str, help='Only load parameters from pth file.')
    parser.add_argument('--strict_load', action='store_true', help='Mixed precision training using apex.')

    parser.add_argument('--start_epoch', default=0, type=int, help='manual epoch number (useful on restarts)')
    parser.add_argument('--epochs', default=90, type=int, metavar='N', help='number of total epochs to run')
    parser.add_argument('--save_dir', default=None, type=str, help='where to save models.')
    parser.add_argument('--save_freq', default=10, type=int, help='How many epochs to save a model.')
    parser.add_argument('--print_freq', default=100, type=int, help='print frequency (default: 100)')

    # training tricks
    parser.add_argument('--label_smoothing', action='store_true')
    parser.add_argument('--weight_init', type=str, default='None', help='How to initialize parameters')
    parser.add_argument('--nesterov', action='store_true')
    parser.add_argument('--grad_clip', type=float, default=None)

    # BN layer
    parser.add_argument('--bn_momentum', type=float, default=None)
    parser.add_argument('--bn_eps', type=float, default=None)

    # data augmentation
    parser.add_argument('--mixup', action='store_true')
    parser.add_argument('--random_erase', action='store_true')
    parser.add_argument('--auto_augment', action='store_true')
    parser.add_argument('--no_data_augment', action='store_true')

    # for loading dataset
    parser.add_argument('--data_dir', type=str, default=None, help='path to dataset')
    parser.add_argument('--dataset', type=str, default=None, help='name of the dataset')
    parser.add_argument('--workers_per_gpu',
                        default=6,
                        type=int,
                        help='number of data loading workers per gpu. default 6.')
    parser.add_argument(
        '--batch_size',
        default=None,
        type=int,
        help='mini-batch size (default: 256), this is the total '
        'batch size of all GPUs on the current node when '
        'using Data Parallel or Distributed Data Parallel',
    )

    parser.add_argument('--batch_size_per_gpu', default=None, type=int, help='batch size per GPU.')
    parser.add_argument('--auto_batch_size', action='store_true', help='allow adjust batch size smartly.')
    parser.add_argument('--num_cv_folds', type=int, default=None, help='Number of cross-validation folds.')
    parser.add_argument('--cv_id', type=int, default=None, help='Current ID of cross-validation fold.')
    parser.add_argument('--input_image_size', type=int, default=224, help='input image size.')
    parser.add_argument('--input_image_crop', type=float, default=0.875, help='crop ratio of input image')

    # for loading model
    parser.add_argument('--arch', default=None, help='model names/module to load')
    parser.add_argument('--pretrained', dest='pretrained', action='store_true', help='use pre-trained model')
    parser.add_argument('--num_classes', type=int, default=None, help='number of classes.')

    # for testing
    parser.add_argument('--dataloader_testing', action='store_true', help='Testing data loader.')

    # for teacher-student distillation
    parser.add_argument('--teacher_input_image_size', type=int, default=None)
    parser.add_argument('--teacher_arch', type=str, default=None)
    parser.add_argument('--teacher_pretrained', action='store_true')
    parser.add_argument('--ts_proj_no_relu', action='store_true')
    parser.add_argument('--ts_proj_no_bn', action='store_true')
    parser.add_argument('--teacher_load_parameters_from', type=str, default=None)
    parser.add_argument('--teacher_feature_weight', type=float, default=None)
    parser.add_argument('--teacher_logit_weight', type=float, default=None)
    parser.add_argument('--ts_clip', type=float, default=None)
    parser.add_argument('--target_downsample_ratio', type=int, default=None)

    opt, _ = parser.parse_known_args(argv)
    return opt

def create_logging(log_filename=None, level=logging.INFO):
    if log_filename is not None:
        mkfilepath(log_filename)
        logging.basicConfig(
            level=level,
            format="%(message)s",
            handlers=[
                logging.FileHandler(log_filename),
                logging.StreamHandler()
            ]
        )
    else:
        logging.basicConfig(
            level=level,
            format="%(message)s",
            handlers=[
                logging.StreamHandler()
            ]
        )

class LearningRateScheduler():
    def __init__(self,
                 mode,
                 lr,
                 target_lr=None,
                 num_training_instances=None,
                 stop_epoch=None,
                 warmup_epoch=None,
                 stage_list=None,
                 stage_decay=None,
                 ):
        self.mode = mode
        self.lr = lr
        self.target_lr = target_lr if target_lr is not None else 0
        self.num_training_instances = num_training_instances if num_training_instances is not None else 1
        self.stop_epoch = stop_epoch if stop_epoch is not None else np.inf
        self.warmup_epoch = warmup_epoch if warmup_epoch is not None else 0
        self.stage_list = stage_list if stage_list is not None else None
        self.stage_decay = stage_decay if stage_decay is not None else 0

        self.num_received_training_instances = 0

        if self.stage_list is not None:
            self.stage_list = [int(x) for x in self.stage_list.split(',')]

    def update_lr(self, batch_size):
        self.num_received_training_instances += batch_size

    def get_lr(self, num_received_training_instances=None):
        if num_received_training_instances is None:
            num_received_training_instances = self.num_received_training_instances

        # start_instances = self.num_training_instances * self.start_epoch
        stop_instances = self.num_training_instances * self.stop_epoch
        warmup_instances = self.num_training_instances * self.warmup_epoch

        assert stop_instances > warmup_instances

        current_epoch = self.num_received_training_instances // self.num_training_instances

        if num_received_training_instances < warmup_instances:
            return float(num_received_training_instances + 1) / float(warmup_instances) * self.lr

        ratio_epoch = float(num_received_training_instances - warmup_instances + 1) / \
                      float(stop_instances - warmup_instances)

        if self.mode == 'cosine':
            factor = (1 - np.math.cos(np.math.pi * ratio_epoch)) / 2.0
            return self.lr + (self.target_lr - self.lr) * factor
        elif self.mode == 'stagedecay':
            stage_lr = self.lr
            for stage_epoch in self.stage_list:
                if current_epoch <= stage_epoch:
                    return stage_lr
                else:
                    stage_lr *= self.stage_decay
                pass  # end if
            pass  # end for
            return stage_lr
        elif self.mode == 'linear':
            factor = ratio_epoch
            return self.lr + (self.target_lr - self.lr) * factor
        else:
            raise RuntimeError('Unknown learning rate mode: ' + self.mode)
        pass  # end if