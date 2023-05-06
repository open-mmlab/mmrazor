import time
import torch
import torch.nn as nn
import transformers
import math
from texttable import Texttable
from mmrazor.implementations.pruning.sparse_gpt import SparseGptMixIn
from mmrazor.implementations.pruning.sparse_gpt.utils import torch_setting

from .quantizer import Quantizer
from .utils import torch_snr_error

class Observer:

    def __init__(self, topk=32):
        self.loss_list = []
        self.topk = topk

    def submit(self, name: str, layerid: int, gptq, error: float):

        item = (name, layerid, {'gptq': gptq, 'error': error})

        if len(self.loss_list) < self.topk:
            self.loss_list.append(item)
            return

        min_error = error
        min_idx = -1
        for idx, data in enumerate(self.loss_list):
            if min_error > data[2]['error']:
                min_idx = idx
                min_error = data[2]['error']

        if min_idx >= 0:
            self.loss_list[min_idx] = item

    def print(self):
        self.loss_list = sorted(self.loss_list, key=lambda s: s[2]['error'], reverse=True)

        table = Texttable()

        table.header(['name', 'error'])
        table.set_cols_dtype(['t', 'f'])

        for item in self.loss_list:
            table.add_row([f"{item[0]}.{item[1]}", item[2]['error']])
        print(table.draw())
        print('\n')

    def items(self):
        return self.loss_list

class GPTQMixIn(SparseGptMixIn):

    def get_input_output_for_obs(m, input, output):
        if self.observe:
            self.input_obs = input
            self.output_obs = output
        else:
            self.input_obs = None
            self.output_obs = None

    def print_loss(self, 
                   name, 
                   q_weight, 
                   weight_error, 
                   timecost):
        table = Texttable()
        name += ' ' * (16 - len(name))

        table.header(['name', 'weight_error', 'fp_inp_SNR', 'q_inp_SNR', 'time'])

        # assign weight
        self.layer.weight.data = q_weight.reshape(self.layer.weight.shape).to(self.layer.weight.data.dtype)

        if self.input_obs is not None:
            # quantize input to int8
            quantizer = Quantizer()
            quantizer.configure(8, perchannel=False, sym=True, mse=False)
            quantizer.find_params(self.input_obs)
            q_in = quantizer.quantize(self.input_obs).type(torch.float16)
            q_out = self.layer(q_in)

            # get kinds of SNR
            q_SNR = torch_snr_error(q_out, self.output_obs).item()
            fp_SNR = torch_snr_error(self.layer(self.input_obs), self.output_obs).item()
        else:
            q_SNR = '-'
            fp_SNR = '-'

        table.add_row([name, weight_error, fp_SNR, q_SNR, timecost])
        print(table.draw().split('\n')[-2])

    @torch.no_grad()
    def quant(self,
              quantizer,
              blocksize=128,
              percdamp=0.01,
              groupsize=-1,
              actorder=False):
        with torch_setting(dtype=torch.float):
            tick = time.time()
            assert self.hessian is not None
            W: torch.Tensor = self.weight_matrix.float()  # out in
            H = self.hessian.float().to(W.device)
            dead = torch.diag(H) == 0
            H[dead, dead] = 1
            W[:, dead] = 0

            if actorder:
                perm = torch.argsort(torch.diag(H), descending=True)
                W = W[:, perm]
                H = H[perm][:, perm]

            Losses = torch.zeros_like(W)
            Q = torch.zeros_like(W)

            damp = percdamp * torch.mean(torch.diag(H))
            diag = torch.arange(self.columns, device=self.dev)
            H[diag, diag] += damp
            H = torch.linalg.cholesky(H)
            H = torch.cholesky_inverse(H)
            H = torch.linalg.cholesky(H, upper=True)
            Hinv = H

            g_idx = []
            scale = []
            zero = []
            now_idx = 1

            for i1 in range(0, self.columns, blocksize):
                i2 = min(i1 + blocksize, self.columns)
                count = i2 - i1

                W1 = W[:, i1:i2].clone()
                Q1 = torch.zeros_like(W1)
                Err1 = torch.zeros_like(W1)
                Losses1 = torch.zeros_like(W1)
                Hinv1 = Hinv[i1:i2, i1:i2]

                for i in range(count):
                    w = W1[:, i]
                    d = Hinv1[i, i]

                    if groupsize != -1:
                        if (i1 + i) % groupsize == 0:
                            quantizer.find_params(W[:, (i1 + i):(i1 + i + groupsize)], weight=True)

                        if ((i1 + i) // groupsize) - now_idx == -1:
                            scale.append(quantizer.scale)
                            zero.append(quantizer.zero)
                            now_idx += 1

                    q = quantizer.quantize(w.unsqueeze(1)).flatten()
                    Q1[:, i] = q
                    Losses1[:, i] = (w - q)**2 / d**2

                    err1 = (w - q) / d
                    W1[:, i:] -= err1.unsqueeze(1).matmul(Hinv1[i, i:].unsqueeze(0))
                    Err1[:, i] = err1

                Q[:, i1:i2] = Q1
                Losses[:, i1:i2] = Losses1 / 2

                W[:, i2:] -= Err1.matmul(Hinv[i1:i2, i2:])

            torch.cuda.synchronize()
            error = torch.sum(Losses).item()

            groupsize = groupsize if groupsize != -1 else self.columns
            g_idx = [i // groupsize for i in range(self.columns)]
            g_idx = torch.tensor(g_idx, dtype=torch.int32, device=Q.device)
            if actorder:
                invperm = torch.argsort(perm)
                Q = Q[:, invperm]
                g_idx = g_idx[invperm]

            if isinstance(self.layer, transformers.Conv1D):
                Q = Q.t()

            if scale == []:
                scale.append(quantizer.scale)
                zero.append(quantizer.zero)
            scale = torch.cat(scale, dim=1)
            zero = torch.cat(zero, dim=1)
            return scale, zero, g_idx, error, Q
    
    def free(self):
        self.input_obs = None
        self.output_obs = None
        self.H = None
        self.Losses = None
        self.Trace = None
        torch.cuda.empty_cache()