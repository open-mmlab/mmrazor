from torch.ao.quantization.observer import _ObserverBase
from mmrazor.models.utils import sync_tensor, pot_quantization
from mmrazor.registry import MODELS

@MODELS.register_module()
class BaseObserver(_ObserverBase):

    min_val: torch.Tensor
    max_val: torch.Tensor

    def __init__(
        self,
        dtype=torch.quint8,
        qscheme=torch.per_tensor_affine,
        reduce_range=False,
        quant_min=None,
        quant_max=None,
        ch_axis=-1,
        is_pot_scale=False,
        factory_kwargs=None,
        eps=torch.finfo(torch.float32).eps) -> None:
        super().__init__(dtype, qscheme, reduce_range, quant_min, quant_max, 
            factory_kwargs, eps)
        factory_kwargs = torch.nn.factory_kwargs(factory_kwargs)
        self.ch_axis = ch_axis
        self.register_buffer("min_val", torch.tensor([], **factory_kwargs))
        self.register_buffer("max_val", torch.tensor([], **factory_kwargs))
        self.is_pot_scale = is_pot_scale

    @torch.jit.export
    def calculate_qparams(self) -> Tuple[torch.Tensor, torch.Tensor]:
        r"""Calculates the quantization parameters."""
        scale, zero_point = self._calculate_qparams(self.min_val, self.max_val)
        scale.data = sync_tensor(scale).data
        zero_point.data = sync_tensor(zero_point).data
        if self.is_pot_scale:
            scale = pot_quantization(scale)
        return scale, zero_point


    @torch.jit.export
    def extra_repr(self):
        return "min_val={}, max_val={} ch_axis={} is_pot_scale={}".format(
            self.min_val, self.max_val, self.ch_axis, self.is_pot_scale)

    def _load_from_state_dict(
        self,
        state_dict: Dict[str, Any],
        prefix: str,
        local_metadata: Dict[str, torch.Tensor],
        strict: bool,
        missing_keys: List[str],
        unexpected_keys: List[str],
        error_msgs: List[str],
    ):
        version = local_metadata.get("version", None)
        if version is None or version < 3:
            local_state = ["min_vals", "max_vals"]
            expected_min_name = "min_vals"
            expected_max_name = "max_vals"
        else:
            local_state = ["min_val", "max_val"]
            expected_min_name = "min_val"
            expected_max_name = "max_val"
        for name in local_state:
            key = prefix + name
            if key in state_dict:
                val = state_dict[key]
                # Custom handling to allow loading min_val or max_val
                # of size N into uninitialized buffers of size 0. The
                # buffers are resized here, and the values are copied in
                # the default state_dict loading code of the parent.
                if name == expected_min_name:
                    self.min_val.resize_(val.shape)
                elif name == expected_max_name:
                    self.max_val.resize_(val.shape)
                else:
                    warnings.warn("Observer load_from_state_dict got \
                        unexpected name {}".format(name))
                # For torchscript module we need to update the attributes here 
                # since we do not call the `_load_from_state_dict` function 
                # defined module.py
                if torch.jit.is_scripting():
                    if name == expected_min_name:
                        self.min_val.copy_(val)
                    elif name == expected_max_name:
                        self.max_val.copy_(val)
                    else:
                        warnings.warn("Observer load_from_state_dict got \
                            unexpected name {}".format(name))
            elif strict:
                missing_keys.append(key)

        if not torch.jit.is_scripting():
            super(PerChannelMinMaxObserver, self)._load_from_state_dict(
                state_dict,
                prefix,
                local_metadata,
                False,
                missing_keys,
                unexpected_keys,
                error_msgs,
            )