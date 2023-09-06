from torch import nn as nn
from torch.nn import functional as F

from basicsr.utils.registry import ARCH_REGISTRY
from basicsr.archs.arch_util import ResidualBlockNoBN, default_init_weights, make_layer


@ARCH_REGISTRY.register()
class FSRCNN(nn.Module):
    """Modified FSRCNN.

    A compacted version modified from FSRCNN in
    "Accelerating the super-resolution convolutional neural network"

    Args:
        num_in_ch (int): Channel number of inputs. Default: 3.
        num_out_ch (int): Channel number of outputs. Default: 3.
        num_feat (int): Channel number of intermediate features. Default: 64.
        num_block (int): Block number in the body network. Default: 16.
        upscale (int): Upsampling factor. Support x2, x3 and x4. Default: 4.
    """

    def __init__(self, num_in_ch=3, num_out_ch=3, num_feat=64, upscale=4):
        super(FSRCNN, self).__init__()
        self.upscale = upscale

        self.conv_first = nn.Conv2d(num_in_ch, num_feat, 3, 1, 1)
        # self.body = make_layer(ResidualBlockNoBN, num_block, num_feat=num_feat)
        # self.body = nn.Sequential()
        self.body = nn.Sequential(
                  nn.Conv2d(num_feat,num_feat, 3, 1, 1),
                  nn.ReLU(),
                  nn.Conv2d(num_feat,num_feat, 3, 1, 1),
                  nn.ReLU(),
                  nn.Conv2d(num_feat,num_feat, 3, 1, 1),
                  nn.ReLU()
                )
        # upsampling
        if self.upscale in [2, 3]:
            self.upconv1 = nn.Conv2d(num_feat, num_feat * self.upscale * self.upscale, 3, 1, 1)
            self.pixel_shuffle = nn.PixelShuffle(self.upscale)
        elif self.upscale == 4:
            self.upconv1 = nn.Conv2d(num_feat, num_feat * 4, 3, 1, 1)
            self.upconv2 = nn.Conv2d(num_feat, num_feat * 4, 3, 1, 1)
            self.pixel_shuffle = nn.PixelShuffle(2)

        self.conv_hr = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_last = nn.Conv2d(num_feat, num_out_ch, 3, 1, 1)

        # activation function
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

        # initialization
        default_init_weights([self.conv_first,self.body, self.upconv1, self.conv_hr, self.conv_last], 0.1)
        if self.upscale == 4:
            default_init_weights(self.upconv2, 0.1)

    def forward(self, x):
        feat = self.lrelu(self.conv_first(x))
        out = self.body(feat)

        if self.upscale == 4:
            out = self.lrelu(self.pixel_shuffle(self.upconv1(out)))
            out = self.lrelu(self.pixel_shuffle(self.upconv2(out)))
        elif self.upscale in [2, 3]:
            out = self.lrelu(self.pixel_shuffle(self.upconv1(out)))

        out = self.conv_last(self.lrelu(self.conv_hr(out)))
        # base = F.interpolate(x, scale_factor=self.upscale, mode='bilinear', align_corners=False)
        # out += base
        return out