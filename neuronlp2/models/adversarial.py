import torch
import numpy as np
import torch.nn as nn
from neuronlp2.nn.utils import generate_optimizer
from torch.optim import RMSprop


class GradientReversal(torch.autograd.Function):
    def __init__(self, scale_):
        super(GradientReversal, self).__init__()
        self.scale = scale_

    def forward(self, inp):
        return inp.view_as(inp)

    def backward(self, grad_out):
        return -self.scale * grad_out


def grad_reverse(x, scale):
    return GradientReversal(scale)(x)


class ResBlock(nn.Module):
    def __init__(self, nhid):
        super(ResBlock, self).__init__()

        self.relu = nn.ReLU(True)
        self.conv1 = nn.Conv1d(nhid, nhid, 5, padding=2)
        self.conv2 = nn.Conv1d(nhid, nhid, 5, padding=2)

    def forward(self, inputs):
        """"Defines the forward computation of the discriminator."""
        assert inputs.dim() == 3
        output = inputs
        output = self.relu(output)
        output = self.conv1(output)
        output = self.relu(output)
        output = self.conv2(output)
        return inputs + (0.3 * output)


class Discriminator(nn.Module):
    def __init__(self,
                 level,
                 grad_reverse=False,
                 scale=1.0):
        super(Discriminator, self).__init__()

        self.level = level
        self.grad_reverse = grad_reverse
        self.scale = scale

    def count_parameters(self):
        param_dict = {}
        for name, param in self.named_parameters():
            if param.requires_grad:
                param_dict[name] = np.prod(param.size())
        return param_dict

    def forward(self, input):
        assert input.dim() == 3
        if self.grad_reverse:
            input = grad_reverse(input, self.scale)

        output = self._run_forward_pass(input)
        if self.level == 'sent':
            output = torch.mean(output, dim=1)
        return self.model(output)


# ref: https://github.com/igul222/improved_wgan_training/blob/master/gan_language.py#L75
class ConvDiscriminator(Discriminator):
    def __init__(self,
                 nhid,
                 mhid,
                 level,
                 nclass=1,
                 grad_reverse=False,
                 scale=1.0):
        super(ConvDiscriminator, self).__init__(level, grad_reverse, scale)

        self.convolve = nn.Sequential(
            nn.Conv1d(nhid, mhid, 1),
            ResBlock(mhid),
            ResBlock(mhid),
            ResBlock(mhid)
        )
        self.model = nn.Linear(mhid, nclass)

    def _run_forward_pass(self, input):
        """"Defines the forward computation of the discriminator."""
        output = input.transpose(1, 2)
        output = self.convolve(output)
        output = output.transpose(1, 2)
        return output


class MLP(Discriminator):
    """A simple discriminator for adversarial training."""

    def __init__(self,
                 nhid,
                 level,
                 nclass=1,
                 grad_reverse=False,
                 scale=1.0):
        super(MLP, self).__init__(level, grad_reverse, scale)

        self.model = nn.Sequential(
            nn.Linear(nhid, 256),
            nn.Tanh(),
            nn.Linear(256, 128),
            nn.Tanh(),
            nn.Linear(128, nclass)
        )

    def _run_forward_pass(self, input):
        return input


class FFNN(Discriminator):
    """A simple discriminator for adversarial training."""

    def __init__(self,
                 nhid,
                 level,
                 nclass=1,
                 grad_reverse=False,
                 scale=1.0,
                 use_activation=False):
        super(FFNN, self).__init__(level, grad_reverse, scale)

        if use_activation:
            self.model = nn.Sequential(
                nn.Linear(nhid, 512),
                nn.Tanh(),
                nn.Linear(512, nclass)
            )
        else:
            self.model = nn.Sequential(
                nn.Linear(nhid, 512),
                nn.Linear(512, nclass)
            )

    def _run_forward_pass(self, input):
        return input


class LR(Discriminator):
    """A simple discriminator for adversarial training."""

    def __init__(self,
                 nhid,
                 level,
                 nclass=1,
                 grad_reverse=False,
                 scale=1.0):
        super(LR, self).__init__(level, grad_reverse, scale)

        self.model = nn.Linear(nhid, nclass)

    def _run_forward_pass(self, input):
        return input


class Motivator(nn.Module):
    def __init__(self,
                 input_size,
                 disc_type,
                 train_level,
                 nclass,
                 scale,
                 use_gpu,
                 opt,
                 lr,
                 betas,
                 gamma,
                 eps,
                 momentum,
                 clip_disc):
        super(Motivator, self).__init__()

        self.clip_disc = clip_disc
        self.scale = scale

        if disc_type == 'weak':
            self.discriminator = LR(input_size, level=train_level, nclass=nclass)
        elif disc_type == 'not-so-weak':
            self.discriminator = MLP(input_size, level=train_level, nclass=nclass)
        elif disc_type == 'strong':
            self.discriminator = ConvDiscriminator(input_size, mhid=128, level=train_level, nclass=nclass)
        else:
            raise NotImplementedError()

        param_dict = self.discriminator.count_parameters()
        print("Motivator (type, #params) : (%s, %d)" % (disc_type, np.sum(list(param_dict.values()))))
        if use_gpu:
            self.discriminator.cuda()

        self.optim = generate_optimizer(opt, lr, self.discriminator.parameters(),
                                        betas, gamma, eps, momentum)
        self.criterion = nn.CrossEntropyLoss()

    def loss(self, input, label):
        output = self.discriminator(input)
        if output.dim() == 3:
            output = output.contiguous().view(-1, output.size(2))

        labels = torch.empty(output.size(0)).fill_(label).type_as(output).long()
        loss = self.criterion(output, labels)

        return output, loss

    def accuracy(self, output, label):
        preds = output.max(1)[1].cpu()

        labels = torch.LongTensor([label])
        labels = labels.expand(*preds.size())
        n_correct = preds.eq(labels).sum().item()
        acc = 1.0 * n_correct / output.size(0)

        return acc

    def update(self, real_in, fake_in, real_id, fake_id):
        self.optim.zero_grad()

        real_out, real_loss = self.loss(real_in, real_id)
        fake_out, fake_loss = self.loss(fake_in, fake_id)

        loss = 0.5 * (real_loss + fake_loss)
        loss.backward()
        # clip_grad_norm_(self.discriminator.parameters(), self.clip_disc)

        self.optim.step()

        real_acc = self.accuracy(real_out, real_id)
        fake_acc = self.accuracy(fake_out, fake_id)

        return real_loss.item(), fake_loss.item(), real_acc, fake_acc

    def gen_loss(self, real_in, fake_in, real_id, fake_id):
        """Function to calculate loss to update the Generator in motivational training."""

        _, real_loss = self.loss(real_in, real_id)
        _, fake_loss = self.loss(fake_in, fake_id)
        loss = self.scale * 0.5 * (real_loss + fake_loss)

        return loss

    def forward(self, input):
        raise NotImplementedError()


class Adversarial(nn.Module):
    def __init__(self,
                 input_size,
                 disc_type,
                 train_level,
                 train_type,
                 reverse_grad,
                 soft_label,
                 nclass,
                 scale,
                 use_gpu,
                 opt,
                 lr,
                 betas,
                 gamma,
                 eps,
                 momentum,
                 clip_disc):
        super(Adversarial, self).__init__()

        self.train_type = train_type
        self.clip_disc = clip_disc
        self.reverse_grad = reverse_grad
        self.scale = scale

        if disc_type == 'weak':
            self.discriminator = LR(input_size, level=train_level, nclass=nclass,
                                    grad_reverse=reverse_grad, scale=scale)
        elif disc_type == 'not-so-weak':
            self.discriminator = MLP(input_size, level=train_level, nclass=nclass,
                                     grad_reverse=reverse_grad, scale=scale)
        elif disc_type == 'strong':
            self.discriminator = ConvDiscriminator(input_size, mhid=128, level=train_level,
                                                   nclass=nclass, grad_reverse=reverse_grad, scale=scale)
        else:
            raise NotImplementedError()

        param_dict = self.discriminator.count_parameters()
        print("Discriminator (type, #params) : (%s, %d)" % (disc_type, np.sum(list(param_dict.values()))))
        if use_gpu:
            self.discriminator.cuda()

        if self.train_type == 'WGAN':
            self.optim = RMSprop(self.discriminator.parameters(), lr=5e-5)
            # self.optim = generate_optimizer(opt, lr, self.discriminator.parameters(),
            #                                 betas, gamma, eps, momentum)
        else:
            self.optim = generate_optimizer(opt, lr, self.discriminator.parameters(),
                                            betas, gamma, eps, momentum)
            self.criterion = nn.CrossEntropyLoss() if self.train_type == 'GR' else nn.BCEWithLogitsLoss()

        self.real = 1
        self.fake = 0
        if soft_label and self.train_type == 'GAN':
            # Tip: (https://github.com/soumith/ganhacks#6-use-soft-and-noisy-labels)
            # Let's use soft labels instead of original ones
            self.real = np.random.uniform(0.7, 1.0)  # 1
            self.fake = np.random.uniform(0.0, 0.3)  # 0

    def loss(self, input, label):
        output = self.discriminator(input)
        if output.dim() == 3:
            output = output.contiguous().view(-1, output.size(2))

        if self.train_type == 'WGAN':
            loss = torch.mean(output)

        else:
            if self.train_type == 'GAN':
                labels = torch.empty(*output.size()).fill_(label).type_as(output)
            elif self.train_type == 'GR':
                labels = torch.empty(output.size(0)).fill_(label).type_as(output).long()
            loss = self.criterion(output, labels)

        return output, loss

    def accuracy(self, output, label, sample_type):
        if self.train_type == 'GR':
            preds = output.max(1)[1].cpu()

        elif self.train_type == 'GAN':
            label = 1
            # if samples are from source distribution, discriminator should predict prob >= 0.5, else prob < 0.5
            if sample_type == 'real':
                preds = (torch.sigmoid(output) >= 0.5).long().cpu()
            else:
                preds = (torch.sigmoid(output) < 0.5).long().cpu()

        else:
            raise NotImplementedError()

        labels = torch.LongTensor([label])
        labels = labels.expand(*preds.size())
        n_correct = preds.eq(labels).sum().item()
        acc = 1.0 * n_correct / output.size(0)

        return acc

    def update(self, real_in, fake_in, real_id, fake_id):
        self.optim.zero_grad()

        # train Discriminator to output closer to 1.0 given examples from source dist.
        # train Discriminator to output closer to 0.0 given examples from target dist.
        # L_D = SUM_(i=1)^B [log D(G(s_{tgt}^i)) + log (1-D(G(s_{src}^i)))]

        if self.train_type == 'GAN':
            real_id, fake_id = self.real, self.fake

        real_out, real_loss = self.loss(real_in, real_id)
        fake_out, fake_loss = self.loss(fake_in, fake_id)

        if self.train_type in ['GR', 'GAN']:
            loss = 0.5 * (real_loss + fake_loss)
        else:
            loss = fake_loss - real_loss

        # Note: usually gradient clipping is not required
        # clip_grad_norm_(self.discriminator.parameters(), self.clip_disc)
        loss.backward()
        self.optim.step()

        real_acc, fake_acc = 0, 0
        if self.train_type in ['GR', 'GAN']:
            real_acc = self.accuracy(real_out, real_id, 'real')
            fake_acc = self.accuracy(fake_out, fake_id, 'fake')

        return real_loss.item(), fake_loss.item(), real_acc, fake_acc

    def gen_loss(self, real_in, fake_in, real_id, fake_id):
        """Function to calculate loss to update the Generator in adversarial training."""

        if self.train_type == 'GR':
            _, real_loss = self.loss(real_in, real_id)
            _, fake_loss = self.loss(fake_in, fake_id)
            loss = 0.5 * (real_loss + fake_loss)
            if not self.reverse_grad:
                loss = -self.scale * loss

        elif self.train_type == 'GAN':
            _, loss = self.loss(fake_in, self.real)
            loss = self.scale * loss
            # _, loss = self.loss(real_in, self.real)
            # loss = -self.scale * loss

        # ref: https://github.com/eriklindernoren/PyTorch-GAN/blob/master/implementations/wgan/wgan.py
        elif self.train_type == 'WGAN':
            # clamp parameters to a cube
            for p in self.discriminator.parameters():
                p.data.clamp_(-0.01, 0.01)

            _, loss = self.loss(fake_in, self.real)
            loss = -self.scale * loss

        else:
            raise NotImplementedError()

        return loss

    def forward(self, input):
        raise NotImplementedError()
