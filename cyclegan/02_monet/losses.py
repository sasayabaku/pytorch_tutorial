import torch
import torch.nn as nn
from torch.autograd import Variable

__all__ = ['GANLoss']


class GANLoss(nn.Module):
    def __init__(self, cuda=False):
        super(GANLoss, self).__init__()
        self.real_label_var = None
        self.fake_label_var = None
        self.loss = nn.MSELoss()
        self.cuda = cuda

    def get_target_tensor(self, input, target_is_real):
        target_tensor = None
        if target_is_real:
            create_label = ((self.real_label_var is None) or (self.real_label_var.numel() != input.numel()))
            if create_label:
                real_tensor = torch.ones(input.size())
                if self.cuda:
                    real_tensor = real_tensor.cuda()
                self.real_label_var = Variable(real_tensor, requires_grad=False)

            target_tensor = self.real_label_var

        else:
            create_label = ((self.fake_label_var is None) or (self.fake_label_var.numel() != input.numel()))
            if create_label:
                fake_tensor = torch.zeros(input.size())
                if self.cuda:
                    fake_tensor = fake_tensor.cuda()
                self.fake_label_var = Variable(fake_tensor, requires_grad=False)

            target_tensor = self.fake_label_var

        return target_tensor

    def __call__(self, input, target_is_real):
        target_tensor = self.get_target_tensor(input, target_is_real)
        return self.loss(input, target_tensor)