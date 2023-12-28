import torch
from torch import nn
import torch.nn.functional as F


class ABF(nn.Module):
    def __init__(self, in_channel, mid_channel, out_channel, fuse):
        super(ABF, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channel, mid_channel, kernel_size=1, bias=False),
            nn.BatchNorm2d(mid_channel),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(mid_channel, out_channel, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channel),
        )
        if fuse:
            self.att_conv = nn.Sequential(
                    nn.Conv2d(mid_channel*2, 2, kernel_size=1),
                    nn.Sigmoid(),
                )
        else:
            self.att_conv = None
        nn.init.kaiming_uniform_(self.conv1[0].weight, a=1)  # pyre-ignore
        nn.init.kaiming_uniform_(self.conv2[0].weight, a=1)  # pyre-ignore

    def forward(self, x, y=None, shape=None):
        n, _, h, w = x.shape
        # transform student features
        x = self.conv1(x)
        if self.att_conv is not None:
            # upsample residual features
            y = F.interpolate(y, shape, mode="nearest")
            # fusion
            z = torch.cat([x, y], dim=1)
            z = self.att_conv(z)
            x = (x * z[:, 0].view(n, 1, h, w) + y * z[:, 1].view(n, 1, h, w))
        # output
        y = self.conv2(x)
        return y, x


class ReviewKD(nn.Module):
    def __init__(
        self, student, in_channels, out_channels, mid_channel
    ):
        super(ReviewKD, self).__init__()
        self.shapes = [1, (24, 8), (48, 16)]
        self.student = student

        abfs = nn.ModuleList()

        for idx, in_channel in enumerate(in_channels):
            abfs.append(ABF(in_channel, mid_channel, out_channels[idx], idx < len(in_channels)-1))

        self.abfs = abfs[::-1]

    def forward(self, x, label=None, cam_label=None, view_label=None):
        if self.training:
            score, feat, fea, feats = self.student(x, is_feat=True)
            x = [feats[0], feats[2], feats[5]][::-1]  # feats[::-1]
            results = []
            out_features, res_features = self.abfs[0](x[0])
            results.append(out_features)
            for features, abf, shape in zip(x[1:], self.abfs[1:], self.shapes[1:]):
                out_features, res_features = abf(features, res_features, shape)
                results.insert(0, out_features)
            return score, feat, fea, results
        else:
            return self.student(x, is_feat=False)


def build_review_kd(
    student, in_channels=[24, 64, 320], out_channels=[192, 384, 768], mid_channel=256
):
    # Swin-Tiny: [64, 96, 96, 32], [64, 192, 48, 16], [64, 384, 24, 8], [64, 768, 12, 4]

    # MobileNetV2:
    # [64, 24, 48, 16], [64, 32, 24, 8], [64, 64, 24, 8], [64, 96, 24, 8], [64, 160, 12, 4], [64, 320, 12, 4]

    model = ReviewKD(student, in_channels, out_channels, mid_channel)
    return model


def hcl(fstudent, fteacher):
    loss_all = 0.0
    for fs, ft in zip(fstudent, fteacher):
        n, c, h, w = fs.shape
        loss = F.mse_loss(fs, ft, reduction='mean')
        cnt = 1.0
        tot = 1.0
        for _l in [4, 2, 1]:
            if _l >= h:
                continue
            tmpfs = F.adaptive_avg_pool2d(fs, (_l, _l))
            tmpft = F.adaptive_avg_pool2d(ft, (_l, _l))
            cnt /= 2.0
            loss += F.mse_loss(tmpfs, tmpft, reduction='mean') * cnt
            tot += cnt
        loss = loss / tot
        loss_all = loss_all + loss
    return loss_all