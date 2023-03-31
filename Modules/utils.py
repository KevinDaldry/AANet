import torch
from torch import nn
from torch.nn import functional as F


class StdConv(nn.Module):
    def __init__(self, in_channel, out_channel, kernel=1, padding=0, activation='relu'):
        super(StdConv, self).__init__()
        self.conv = nn.Conv2d(in_channel, out_channel, kernel, padding=padding)
        self.bn = nn.BatchNorm2d(num_features=out_channel)
        if activation == 'relu':
            self.act = nn.ReLU(inplace=True)
        elif activation == 'leaky_relu':
            self.act = nn.LeakyReLU(inplace=True)
        elif activation == 'gelu':
            self.act = nn.GELU()
        else:
            self.act = nn.Identity()

    def forward(self, x):
        return self.bn(self.act(self.conv(x)))


class InceptionConv(nn.Module):
    def __init__(self, in_channel=64, out_channel=1, branch=2):
        super(InceptionConv, self).__init__()
        self.residual = StdConv(in_channel, out_channel, kernel=1)
        self.branch_list = nn.ModuleList()
        for i in range(branch):
            self.branch_list.append(StdConv(out_channel, out_channel // branch, kernel=2*(i+1)+1, padding=i+1))
        self.fusion = StdConv(out_channel, out_channel, kernel=1)

    def forward(self, x):
        tensor_list = []
        res = self.residual(x)
        for item in self.branch_list:
            tensor_list.append(item(res))
        out = tensor_list[0]
        for item in tensor_list[1: ]:
            out = torch.cat([out, item], dim=1)
        out = self.fusion(out)
        return out + res


class SAG(nn.Module):
    def __init__(self, in_channel, out_channel, upsize):
        super(SAG, self).__init__()
        self.c = out_channel
        self.up = upsize
        self.cos = nn.CosineSimilarity(dim=-1)
        self.t1 = nn.Conv2d(in_channel, out_channel, kernel_size=1)
        self.t2 = nn.Conv2d(in_channel, out_channel, kernel_size=1)

        self.sim_pool = nn.AvgPool2d(kernel_size=2)

        self.dif_fusion = InceptionConv(out_channel * 2, out_channel)
        self.dif_branch1 = nn.Conv2d(out_channel, out_channel, kernel_size=3, padding=1)
        self.dif_branch2 = nn.Conv2d(out_channel, out_channel, kernel_size=5, padding=2)

        self.fusion = StdConv(out_channel * 2, out_channel, kernel=1)

    def forward(self, t1, t2):
        assert t1.shape == t2.shape, f'tensor shape cannot match'
        B, _, H, W = t1.shape

        res1 = self.t1(t1)
        t1 = res1.contiguous().view(B, -1, self.c)
        res2 = self.t2(t2)
        t2 = res2.contiguous().view(B, -1, self.c)

        similarity = self.cos(t1, t2).unsqueeze(dim=-1)
        difference = torch.full(similarity.shape, 1.).to('cuda') - similarity

        sim1 = torch.mul(t1, similarity).permute(0, 2, 1).contiguous().view(B, -1, H, W) + res1
        sim2 = torch.mul(t2, similarity).permute(0, 2, 1).contiguous().view(B, -1, H, W) + res2
        dif1 = torch.mul(t1, difference).permute(0, 2, 1).contiguous().view(B, -1, H, W) + res1
        dif2 = torch.mul(t2, difference).permute(0, 2, 1).contiguous().view(B, -1, H, W) + res2

        sim = self.sim_pool(torch.abs(sim1 - sim2))
        dif = self.dif_fusion(torch.cat([dif1, dif2], dim=1))

        b, c, h, w = dif.shape
        attn = sim.contiguous().view(b, -1, c)

        inception1 = self.dif_branch1(dif)
        branch1 = inception1.contiguous().view(b, c, -1)
        attn1 = nn.Softmax(dim=-1)(torch.bmm(attn, branch1))
        attn1 = torch.bmm(attn.permute(0, 2, 1), attn1).view(b, c, h, w) + inception1

        inception2 = self.dif_branch2(dif)
        branch2 = inception2.contiguous().view(b, c, -1)
        attn2 = nn.Softmax(dim=-1)(torch.bmm(attn, branch2))
        attn2 = torch.bmm(attn.permute(0, 2, 1), attn2).view(b, c, h, w) + inception2

        out = F.interpolate(self.fusion(torch.cat([attn1, attn2], dim=1)) + dif, size=self.up, mode='bilinear')

        return out


class WeightRearrangementModule(nn.Module):
    def __init__(self, in_channel, stage, upsize, inner_channel=None):
        super(WeightRearrangementModule, self).__init__()
        self.c = inner_channel if inner_channel is not None else in_channel // stage
        self.num = stage
        self.up = upsize
        self.reduce = nn.Conv2d(in_channel, self.c, kernel_size=1)
        self.weight = StdConv(self.c, stage, kernel=1)
        self.fusion = nn.ModuleList()
        self.conv = nn.ModuleList()
        for i in range(stage):
            self.fusion.append(StdConv(self.c, self.c, kernel=3, padding=1))
            self.conv.append(nn.Conv2d(self.c, self.c, kernel_size=1))

    def forward(self, x):
        result = []
        base = F.adaptive_avg_pool2d(x[0], (1, 1))
        for item in x[1: ]:
            base = torch.cat([base, F.adaptive_avg_pool2d(item, (1, 1))], dim=1)
        w = nn.Softmax(dim=1)(self.weight(self.reduce(base)))

        for i in range(self.num):
            B, C, H, W = x[i].shape
            hidden = F.interpolate(torch.mul(x[i], w[:, i, :, :].unsqueeze(dim=1).repeat(1, C, H, W)), size=self.up, mode='bilinear')
            result.append(self.fusion[i](nn.AvgPool2d(kernel_size=self.up // H)(hidden)))

        fpn = self.conv[-1](result[-1])
        for i in range(1, self.num):
            fpn = self.conv[-(i + 1)](F.interpolate(fpn, scale_factor=2, mode='bilinear') + result[-(i + 1)])
        return fpn
