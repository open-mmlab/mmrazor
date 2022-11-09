# Copyright (c) OpenMMLab. All rights reserved.
import torch
from mmengine.config import Config
from torch.ao.quantization import get_default_qconfig
from torch.ao.quantization.quantize_fx import convert_fx, prepare_fx
from torch.onnx import OperatorExportTypes
from mmrazor.registry import MODELS

# ckpt_path = '/mnt/lustre/humu/experiments/adaround/quantizied.pth'
# state_dict = torch.load(ckpt_path, map_location='cpu')
# for k, v in state_dict['state_dict'].items():
#     print(k)

cfg_path = 'configs/quantization/ptq/demo.py'
onnx_path = '../experiments/export_demo/resnet18_quant.onnx'
cfg = Config.fromfile(cfg_path)
model_fp = MODELS.build(cfg.model)
model_fp.eval()

qconfig = get_default_qconfig('fbgemm')
qconfig_dict = {'': qconfig}
model_pre = prepare_fx(model_fp, qconfig_dict)
model_quant = convert_fx(model_pre)

# import pdb
# pdb.set_trace()

dummy_input = torch.randn([1, 3, 224, 224])
torch.onnx.export(
    model_quant,
    dummy_input,
    onnx_path,
    opset_version=13,
    operator_export_type=OperatorExportTypes.ONNX_ATEN_FALLBACK)
