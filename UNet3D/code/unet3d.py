import torch 
import torch.nn as nn
import torch.nn.functional as F

class ConvBlock3D(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.seq = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm3d(out_ch),
            nn.LeakyReLU(inplace=True),
            nn.Conv3d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm3d(out_ch),
            nn.LeakyReLU(inplace=True),
        )
    def forward(self, x): return self.seq(x)

class Down3D(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.pool = nn.MaxPool3d(kernel_size=(1,2,2), stride=(1,2,2))  # no z pool
        self.conv = ConvBlock3D(in_ch, out_ch)
    def forward(self, x): return self.conv(self.pool(x))

class Up3D(nn.Module):
    def __init__(self, in_dec, in_skip, out_ch):
        super().__init__()
        self.up     = nn.Upsample(scale_factor=(1,2,2), mode='trilinear', align_corners=True)
        self.reduce = nn.Conv3d(in_dec, in_skip, kernel_size=1, bias=False)
        self.conv   = ConvBlock3D(in_skip*2, out_ch)
    def forward(self, x_dec, x_skip):
        x = self.up(x_dec)      
        x = self.reduce(x)
        dh = x_skip.size(3) - x.size(3)
        dw = x_skip.size(4) - x.size(4)
        if dh or dw:
            x = F.pad(x, [0, max(0,dw), 0, max(0,dh), 0, 0])
        x = torch.cat([x_skip, x], dim=1)
        return self.conv(x)

class UNet3D(nn.Module):
    def __init__(self, in_ch=1, n_class=4, base=26):
        super().__init__()
        f1,f2,f3,f4,f5 = base, base*2, base*4, base*8, base*16
        self.in_conv = ConvBlock3D(in_ch, f1)
        self.d1 = Down3D(f1, f2)
        self.d2 = Down3D(f2, f3)
        self.d3 = Down3D(f3, f4)
        self.d4 = Down3D(f4, f5)
        self.u1 = Up3D(f5, f4, f4)
        self.u2 = Up3D(f4, f3, f3)
        self.u3 = Up3D(f3, f2, f2)
        self.u4 = Up3D(f2, f1, f1)
        self.head = nn.Conv3d(f1, n_class, kernel_size=1)
        self.apply(self._init)

    @staticmethod
    def _init(m):
        if isinstance(m, nn.Conv3d):
            nn.init.kaiming_normal_(m.weight, nonlinearity='leaky_relu')
            if m.bias is not None: nn.init.zeros_(m.bias)
        elif isinstance(m, (nn.BatchNorm3d, nn.GroupNorm)):
            nn.init.ones_(m.weight); nn.init.zeros_(m.bias)

    def forward(self, x):
        x0 = self.in_conv(x)   
        x1 = self.d1(x0)   
        x2 = self.d2(x1)      
        x3 = self.d3(x2)     
        x4 = self.d4(x3)      
        y  = self.u1(x4, x3)
        y  = self.u2(y, x2)
        y  = self.u3(y, x1)
        y  = self.u4(y, x0)
        return self.head(y)   
