import torch
import torch.nn as nn
import torch.nn.functional as F


class VegIndexLayer(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        if x.shape[1] > 3:
            x = x[:, :3, :, :]

        # x shape: (B, 3, H, W)  RGB
        r = x[:, 0:1, :, :]
        g = x[:, 1:2, :, :]
        b = x[:, 2:3, :, :]

        # GLI: (2G - R - B) / (2G + R + B)
        gli = (2 * g - r - b) / (2 * g + r + b + 1e-6)

        # Concatena como 4 canal
        return torch.cat([x, gli], dim=1)


class Bottleneck(nn.Module):
    def __init__(self, c, hidden=None, shortcut=True):
        super().__init__()
        hidden = hidden or c
        self.conv1 = nn.Conv2d(c, hidden, 1, 1, 0)
        self.bn1 = nn.BatchNorm2d(hidden)
        self.act = nn.SiLU(inplace=True)
        self.conv2 = nn.Conv2d(hidden, c, 3, 1, 1)
        self.bn2 = nn.BatchNorm2d(c)
        self.use_shortcut = shortcut and c == c

    def forward(self, x):
        y = self.act(self.bn1(self.conv1(x)))
        y = self.bn2(self.conv2(y))
        return self.act(y + x)


class PEC(nn.Module):
    # divide map em 4 patches (top-left, top-right, bottom-left, bottom-right)
    # concatena em canais (reduz H,W pela metade, multiplica canais por 4)
    # aplica squeeze 1x1 conv
    # passa por bottleneck(s)
    # upsamples de volta para o tamanho original
    def __init__(self, bottleneck_repeats=2, squeeze_factor=0.5):
        super().__init__()
        self.bottleneck_repeats = bottleneck_repeats
        self.squeeze_factor = squeeze_factor
        self.initialized = False

    def _init(self, c_in):
        c_expanded = c_in * 4
        c_squeezed = max(1, int(c_in * self.squeeze_factor))
        self.pre_conv = nn.Conv2d(c_expanded, c_squeezed, 1, 1, 0)
        self.pre_bn = nn.BatchNorm2d(c_squeezed)
        self.bottlenecks = nn.Sequential(
            *[Bottleneck(c_squeezed, hidden=max(1, c_squeezed // 2)) for _ in range(self.bottleneck_repeats)]
        )
        self.post_conv = nn.Conv2d(c_squeezed, c_in, 1, 1, 0)
        self.post_bn = nn.BatchNorm2d(c_in)
        self.act = nn.SiLU(inplace=True)
        self.initialized = True

    def forward(self, x):
        # x: B,C,H,W
        B, C, H, W = x.shape

        if not self.initialized:
            self._init(C)
        # slice into 4 submaps
        # if H or W odd, pad 1
        pad_h = H % 2
        pad_w = W % 2

        if pad_h or pad_w:
            x = F.pad(x, (0, pad_w, 0, pad_h))
            _B, C, H, W = x.shape

        x00 = x[..., 0:H:2, 0:W:2]
        x01 = x[..., 0:H:2, 1:W:2]
        x10 = x[..., 1:H:2, 0:W:2]
        x11 = x[..., 1:H:2, 1:W:2]
        y = torch.cat([x00, x01, x10, x11], dim=1)  # B x 4C x H/2 x W/2
        y = self.act(self.pre_bn(self.pre_conv(y)))
        y = self.bottlenecks(y)
        y = self.act(self.post_bn(self.post_conv(y)))
        # upsample back to original H,W
        y = F.interpolate(y, size=(H, W), mode="nearest")
        # residual connection
        return self.act(y + x)


class ESC2f(nn.Module):
    # expande canais via 1x1
    # aplica conv3 e conv5 em paralelo
    # concat + squeeze + residual

    def __init__(self, expansion=1.5):
        super().__init__()
        self.expansion = expansion
        self.initialized = False

    def _init(self, c_in):
        c_mid = max(1, int(c_in * self.expansion))
        self.conv1 = nn.Conv2d(c_in, c_mid, 1, 1, 0)
        self.bn1 = nn.BatchNorm2d(c_mid)
        self.conv3 = nn.Conv2d(c_mid, c_mid, 3, 1, 1, groups=1)
        self.conv5 = nn.Conv2d(c_mid, c_mid, 5, 1, 2, groups=1)
        self.bn_out = nn.BatchNorm2d(c_mid * 2)
        self.squeeze = nn.Conv2d(c_mid * 2, c_in, 1, 1, 0)
        self.bn_s = nn.BatchNorm2d(c_in)
        self.act = nn.SiLU(inplace=True)
        self.initialized = True

    def forward(self, x):
        if not self.initialized:
            self._init(x.shape[1])

        y = self.act(self.bn1(self.conv1(x)))
        y3 = self.conv3(y)
        y5 = self.conv5(y)
        ycat = torch.cat([y3, y5], dim=1)
        ycat = self.act(self.bn_out(ycat))
        y = self.act(self.bn_s(self.squeeze(ycat)))
        return self.act(y + x)


class MFI(nn.Module):
    # divide canais em 4 grupos
    # processa cada grupo com pequenas convs/bottlenecks
    # combina e concatena
    def __init__(self, groups=4, reduction=0.5):
        super().__init__()
        self.groups = groups
        self.reduction = reduction
        self.initialized = False

    def _init(self, c_in):
        assert c_in % self.groups == 0 or True
        self.c_in = c_in
        cg = max(1, c_in // self.groups)
        self.squeeze = nn.Conv2d(cg, cg, 1, 1, 0)
        self.bn = nn.BatchNorm2d(cg)
        # small bottlenecks for each group
        self.bns = nn.ModuleList([Bottleneck(cg, hidden=max(1, cg // 2)) for _ in range(self.groups)])
        self.post = nn.Conv2d(cg * self.groups, c_in, 1, 1, 0)
        self.post_bn = nn.BatchNorm2d(c_in)
        self.act = nn.SiLU(inplace=True)
        self.initialized = True

    def forward(self, x):
        _B, C, _H, _W = x.shape

        if not self.initialized:
            self._init(C)

        # split channels into groups as evenly as possible
        splits = torch.chunk(x, self.groups, dim=1)
        outs = []
        prev = None

        for i, part in enumerate(splits):
            y = self.act(self.bn(self.squeeze(part)))
            y = self.bns[i](y)
            # cross-group fusion for groups 1..n

            if prev is not None:
                y = y + prev

            prev = y
            outs.append(y)
        out = torch.cat(outs, dim=1)
        out = self.act(self.post_bn(self.post(out)))
        return self.act(out + x)


# Teste
if __name__ == "__main__":
    x = torch.randn(2, 3, 640, 640)
    #  vi = VegIndexLayer()
    #  y = vi(x)
    #  print('VegIndex:', y.shape)
    pec = PEC()
    out = pec(y[:, :3])
    print("PEC:", out.shape)
    esc = ESC2f()
    mfi = MFI()
    z = esc(out)
    print("ESC2f:", z.shape)
    w = mfi(z)
    print("MFI:", w.shape)
