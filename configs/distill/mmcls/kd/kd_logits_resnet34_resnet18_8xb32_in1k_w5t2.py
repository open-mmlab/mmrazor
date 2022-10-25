_base_ = './kd_logits_resnet34_resnet18_8xb32_in1k_w5t1.py'

model = dict(distiller=dict(distill_losses=dict(loss_kl=dict(tau=2))))
