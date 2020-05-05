import torch
from torch import nn
import torch.nn.functional as F

class UNet3D_ACDC(nn.Module):
    def __init__(
        self,
        in_channels=1,
        n_classes=2,
        depth=5,
        base_n_filters = 26,
        padding=True,
        batch_norm=True,
        dropout=0,
        up_mode='upsample',
    ):
        super(UNet3D_ACDC, self).__init__()
        assert up_mode in ('upconv', 'upsample')
        self.padding = padding
        self.depth = depth
        self.dropout= dropout
        prev_channels = in_channels
        self.down_path = nn.ModuleList()
        for i in range(depth):
            self.down_path.append(
                UNetConvBlock(prev_channels, base_n_filters * (2**i), padding, batch_norm)
            )
            prev_channels = base_n_filters * (2**i)
            if i < depth-1 and i >0 and self.dropout:
                self.down_path.append(
                    nn.Dropout(self.dropout)
                )
        self.up_path = nn.ModuleList()
        for i in reversed(range(depth - 1)):
            self.up_path.append(
                UNetUpBlock(prev_channels, base_n_filters * (2**i), up_mode, padding, batch_norm, dropout)
            )
            prev_channels = base_n_filters * (2**i)
        
        self.ds2_conv = nn.Sequential(
            nn.Conv3d(base_n_filters * (2**(depth-3)), n_classes, kernel_size=1),
            nn.ReLU(),
        )
        self.ds3_conv = nn.Sequential(
            nn.Conv3d(base_n_filters * (2**(depth-4)), n_classes, kernel_size=1),
            nn.ReLU(),
        )
        self.last = nn.Conv3d(prev_channels, n_classes, kernel_size=1)

    def forward(self, x):
        blocks = []
        blocks_up = []
        for i, down in enumerate(self.down_path):
            x = down(x)
            if i != len(self.down_path) - 1:
                blocks.append(x)
                x = F.max_pool3d(x, (2,2,1))

        for i, up in enumerate(self.up_path):
            x = up(x, blocks[-i - 1])
            blocks_up.append(x)
        
        ds_2 = blocks_up[-3]
        ds_2_seg = self.ds2_conv(ds_2)
        ds_2_seg_up = F.interpolate(ds_2_seg, scale_factor=(2,2,1))
        # print("ds_2_seg_up shape: ", ds_2_seg_up.shape)

        ds_3 = blocks_up[-2]
        ds_3_seg = self.ds3_conv(ds_3)
        # print("ds_3_seg_up shape: ", ds_3_seg.shape)

        ds_2_seg_up_ds3_seg_sum = ds_2_seg_up + ds_3_seg
        ds_2_seg_up_ds3_seg_sum_up = F.interpolate(ds_2_seg_up_ds3_seg_sum, scale_factor=(2,2,1))

        seg = self.last(x)
        outputs = seg + ds_2_seg_up_ds3_seg_sum_up
        probability = F.softmax(outputs, dim=1)
        predictions = torch.argmax(probability, dim=1)
        return outputs, predictions

class UNetConvBlock(nn.Module):
    def __init__(self, in_size, out_size, padding, batch_norm):
        super(UNetConvBlock, self).__init__()
        block = []

        block.append(nn.Conv3d(in_size, out_size, kernel_size=3, padding=int(padding)))
        block.append(nn.ReLU())
        if batch_norm:
            block.append(nn.BatchNorm3d(out_size))

        block.append(nn.Conv3d(out_size, out_size, kernel_size=3, padding=int(padding)))
        block.append(nn.ReLU())
        if batch_norm:
            block.append(nn.BatchNorm3d(out_size))

        self.block = nn.Sequential(*block)

    def forward(self, x):
        out = self.block(x)
        return out

class Interpolate(nn.Module):
    def __init__(self, scale_factor, mode):
        super(Interpolate, self).__init__()
        self.interp = nn.functional.interpolate
        self.scale_factor = scale_factor
        self.mode = mode
    def forward(self, x):
        x = self.interp(x, scale_factor=self.scale_factor, mode=self.mode)
        return x

class UNetUpBlock(nn.Module):
    def __init__(self, in_size, out_size, up_mode, padding, batch_norm, dropout):
        super(UNetUpBlock, self).__init__()
        self.dropout= dropout
        self.up_mode = up_mode
        if up_mode == 'upconv':
            self.up = nn.ConvTranspose3d(in_size, out_size, kernel_size=2, stride=2)
        elif up_mode == 'upsample':
            pass
            self.up = nn.Sequential(
                Interpolate(scale_factor=(2,2,1), mode='trilinear'),
                nn.Conv3d(in_size, out_size, kernel_size=1),
            )

        self.conv_block = UNetConvBlock(in_size, out_size, padding, batch_norm)

    def center_crop(self, layer, target_size):
        _, _, layer_height, layer_width, layer_depth = layer.size()
        diff_y = (layer_height - target_size[0]) // 2
        diff_x = (layer_width - target_size[1]) // 2
        return layer[
            :, :, diff_y : (diff_y + target_size[0]), diff_x : (diff_x + target_size[1])
        ]

    def forward(self, x, bridge):
        # if self.up_mode == 'upconv':
        #     up = self.up(x)
        # elif self.up_mode == 'upsample':
        #     up = F.interpolate(x, scale_factor=(1,2,2))
        up = self.up(x)
        crop1 = self.center_crop(bridge, up.shape[2:])
        out = torch.cat([up, crop1], 1)
        out = F.dropout(out, self.dropout)
        out = self.conv_block(out)
        return out

if __name__ == "__main__":
    unet = UNet3D_ACDC(padding=True, batch_norm=True)
    print(unet)