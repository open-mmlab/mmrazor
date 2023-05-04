import torch
import torch.nn as nn
import transformers
import math
from texttable import Texttable
from mmrazor.implementations.pruning.sparse_gpt import SparseGptMixIn
from mmrazor.implementations.pruning.sparse_gpt.utils import torch_setting

class GPTQMixIn(SparseGptMixIn):

    @torch.no_grad()
    def quant(self,
              quantizer,
              blocksize=128,
              percdamp=0.01,
              groupsize=-1,
              actorder=False):
        with torch_setting(dtype=torch.float):
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

            # if isinstance(self.layer, transformers.Conv1D):
            #     Q = Q.t()

            # self.print_loss(name=name, q_weight=Q, weight_error=error, timecost=(time.time() - tick))

            if scale == []:
                scale.append(quantizer.scale)
                zero.append(quantizer.zero)
            scale = torch.cat(scale, dim=1)
            zero = torch.cat(zero, dim=1)
            return scale, zero, g_idx, error