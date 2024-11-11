import torch


class GaussianNormalizer(object):
    def __init__(self, x, eps=0.0000001):
        super(GaussianNormalizer, self).__init__()

        self.mean = torch.mean(x)
        self.std = torch.std(x)
        self.eps = eps

    def encode(self, x):
        x = (x - self.mean) / (self.std + self.eps)
        return x

    def decode(self, x):
        x = (x * (self.std + self.eps)) + self.mean
        return x

    def cuda(self):
        self.mean = self.mean.cuda()
        self.std = self.std.cuda()

    def cpu(self):
        self.mean = self.mean.cpu()
        self.std = self.std.cpu()


class GaussianNormalizer2(object):
    def __init__(self, mean, std, eps=0.0000001):
        super(GaussianNormalizer2, self).__init__()

        self.mean = mean
        self.std = std
        self.eps = eps

    def encode(self, x):
        x = (x - self.mean) / (self.std + self.eps)
        return x

    def decode(self, x):
        x = (x * (self.std + self.eps)) + self.mean
        return x

    def cuda(self):
        self.mean = self.mean.cuda()
        self.std = self.std.cuda()

    def cpu(self):
        self.mean = self.mean.cpu()
        self.std = self.std.cpu()

class RangeNormalizer(object):
    def __init__(self, x, include_batch_chnl=True, low=0.0, high=1.0, eps=0.00001):
        super(RangeNormalizer, self).__init__()
        self.include_batch_chnl = include_batch_chnl
        self.eps = eps
        if not include_batch_chnl:
            self.bs = bs = -1
            mymin = torch.min(x, 0)[0].view(-1)
            mymax = torch.max(x, 0)[0].view(-1)
        else:
            self.bs = bs = x.size(0)
            mymin = torch.min(x, 1)[0].view(bs, -1)
            mymax = torch.max(x, 1)[0].view(bs, -1)

        self.a = (high - low)/(mymax - mymin)
        self.b = -self.a*mymax + high

    def encode(self, x):
        s = x.size()
        if not self.include_batch_chnl:
            x = x.view(s[0], -1)
        else:
            assert s[0] == self.bs
            x = x.view(self.bs, s[1], -1)

        x = (self.a.unsqueeze(-2) + self.eps)*x + self.b.unsqueeze(-2)
        x = x.view(s)
        return x

    def decode(self, x):
        s = x.size()
        if not self.include_batch_chnl:
            x = x.view(s[0], -1)
        else:
            assert s[0] == self.bs
            x = x.view(self.bs, s[1], -1)

        x = (x - self.b.unsqueeze(-2))/(self.a.unsqueeze(-2) + self.eps)
        x = x.view(s)
        return x

    def cuda(self):
        self.a = self.a.cuda()
        self.b = self.b.cuda()

    def cpu(self):
        self.a = self.a.cpu()
        self.b = self.b.cpu()
