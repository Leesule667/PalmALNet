
import torch, torchvision
import itertools
from torch import nn
from timm.models.vision_transformer import trunc_normal_
from timm.models.layers import SqueezeExcite


class Conv2d_BN(torch.nn.Sequential):
    def __init__(self, a, b, ks=1, stride=1, pad=0, dilation=1,
                 groups=1, bn_weight_init=1, resolution=-10000):
        super().__init__()

        self.add_module('c', torch.nn.Conv2d(
            a, b, ks, stride, pad, dilation, groups, bias=False))
        self.add_module('bn', torch.nn.BatchNorm2d(b))
        torch.nn.init.constant_(self.bn.weight, bn_weight_init)
        torch.nn.init.constant_(self.bn.bias, 0)

    @torch.no_grad()
    def fuse(self):
        c, bn = self._modules.values()
        w = bn.weight / (bn.running_var + bn.eps) ** 0.5
        w = c.weight * w[:, None, None, None]
        b = bn.bias - bn.running_mean * bn.weight / \
            (bn.running_var + bn.eps) ** 0.5
        m = torch.nn.Conv2d(w.size(1) * self.c.groups, w.size(
            0), w.shape[2:], stride=self.c.stride, padding=self.c.padding, dilation=self.c.dilation, groups=self.c.groups)
        m.weight.data.copy_(w)
        m.bias.data.copy_(b)
        return m


class BN_Linear(torch.nn.Sequential):
    def __init__(self, a, b, bias=True, std=0.02):
        super().__init__()
        self.add_module('bn', torch.nn.BatchNorm1d(a))
        self.add_module('l', torch.nn.Linear(a, b, bias=bias))
        trunc_normal_(self.l.weight, std=std)
        if bias:
            torch.nn.init.constant_(self.l.bias, 0)

    @torch.no_grad()
    def fuse(self):
        bn, l = self._modules.values()
        w = bn.weight / (bn.running_var + bn.eps) ** 0.5
        b = bn.bias - self.bn.running_mean * \
            self.bn.weight / (bn.running_var + bn.eps) ** 0.5
        w = l.weight * w[None, :]
        if l.bias is None:
            b = b @ self.l.weight.T
        else:
            b = (l.weight @ b[:, None]).view(-1) + self.l.bias
        m = torch.nn.Linear(w.size(1), w.size(0))
        m.weight.data.copy_(w)
        m.bias.data.copy_(b)
        return m


class PatchMerging(torch.nn.Module):
    def __init__(self, dim, out_dim, input_resolution):
        super().__init__()
        self.norm = nn.BatchNorm2d(dim)
        self.relu = nn.ReLU6(inplace=True)
        self.conv = nn.Conv2d(dim, out_dim, kernel_size=1, stride=1, bias=False)
        self.pool = nn.AvgPool2d(kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        x = self.pool(self.conv(self.relu(self.norm(x))))
        return x


class DepthwiseSeparableConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super(DepthwiseSeparableConv2d, self).__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size, stride, padding, groups=in_channels, bias=False)
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1, 1, 0, bias=False)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x


class Residual(torch.nn.Module):
    def __init__(self, m, drop=0.):
        super().__init__()
        self.m = m
        self.drop = drop

    def forward(self, x):
        if self.training and self.drop > 0:
            return x + self.m(x) * torch.rand(x.size(0), 1, 1, 1,
                                              device=x.device).ge_(self.drop).div(1 - self.drop).detach()
        else:
            return x + self.m(x)


class FFN(torch.nn.Module):
    def __init__(self, ed, h, resolution=0):
        super().__init__()
        self.pw1 = nn.Conv2d(ed, h, 1, 1, 0)
        self.act = torch.nn.ReLU6()
        self.pw2 = nn.Conv2d(h, ed, 1, 1, 0)

    def forward(self, x):
        x = self.pw1(x)
        x = self.act(x)
        x = self.pw2(x)
        return x


class FFU(nn.Module):
    def __init__(self, n_feats, out):
        super(FFU, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(n_feats, n_feats // 4, 1, padding=0, bias=False), nn.ReLU(),
            nn.Conv2d(n_feats // 4, n_feats, 1, padding=0, bias=False))
        self.conv2 = nn.Conv2d(n_feats, out, kernel_size=1, padding=0, stride=1)

    def forward(self, x):
        return self.conv2(self.conv1(x))

class MultiheadSelfAttention(nn.Module):
    def __init__(self, input_dim, num_heads, embed_dim):
        super(MultiheadSelfAttention, self).__init__()
        self.embed_layer = nn.Conv2d(input_dim, embed_dim, 1, 1, 0)
        self.attention = nn.MultiheadAttention(embed_dim, num_heads)
        self.unembed_layer = nn.Conv2d(embed_dim, input_dim, 1, 1, 0)

    def forward(self, input_tensor):
        # input_tensor: 输入张量 [B, C, H, W]

        # 嵌入 (embedding)
        embedded_input = self.embed_layer(input_tensor)  # [B, H * W, embed_dim]

        # 将输入张量转换为 [B, H * W, C]
        B, C, H, W = embedded_input.size()
        embedded_input = embedded_input.view(B, H * W, C)

        # 多头自注意力 (multi-head self-attention)
        attended_output, _ = self.attention(embedded_input.transpose(0, 1), embedded_input.transpose(0, 1), embedded_input.transpose(0, 1))
        # attended_output: [H * W, B, embed_dim]

        # 将输出张量恢复为原来的形状 [B, C, H, W]
        output_tensor = attended_output.transpose(0, 1).view(B, H, W, C).permute(0, 3, 1, 2)

        # 反嵌入 (unembedding)
        output_tensor = self.unembed_layer(output_tensor)  # [B, H * W, C]

        return output_tensor


class SpatialAttention(nn.Module):
    def __init__(self, in_channels):
        super(SpatialAttention, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 1, kernel_size=1, stride=1, padding=0, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Compute spatial attention map
        weights = self.conv1(x)
        weights = self.sigmoid(weights)
        return x * weights


class A3Conv(torch.nn.Module):

    def __init__(self, type, in_can,
                 ed, kd, nh=8,
                 ar=4,
                 resolution=14,
                 window_resolution=7,
                 kernels=[5, 5, 5, 5], ):
        super().__init__()
        self.dw0 = Residual(Conv2d_BN(ed, ed, 3, 1, 1, groups=ed, bn_weight_init=0., resolution=resolution))
        self.ffn0 = Residual(FFN(ed, int(ed * 2), resolution))
        self.se = SqueezeExcite(ed, 1)
        self.act = MultiheadSelfAttention(ed, nh, ed * 2)
        self.dw1 = Residual(Conv2d_BN(ed, ed, 3, 1, 1, groups=ed, bn_weight_init=0., resolution=resolution))
        self.ffn1 = Residual(nn.Sequential(nn.BatchNorm2d(ed), FFN(ed, int(ed * 2), resolution)))
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(ed, ed * 2, ),
            nn.ReLU6(inplace=True),
            nn.Linear(ed * 2, ed)
        )
        self.bn = nn.BatchNorm2d(ed)

        conv_dim = ed * 2
        # self.conv1 = nn.Sequential(nn.BatchNorm2d(in_can), torch.nn.Conv2d(in_channels=in_can, out_channels=conv_dim, kernel_size=1, stride=1, padding=0, bias=False))

        self.conv = nn.Sequential(nn.BatchNorm2d(in_can), torch.nn.Conv2d(in_channels=in_can, out_channels=conv_dim, kernel_size=1, stride=1, padding=0, bias=False))
        # self.conv1 = nn.Sequential(torch.nn.Conv2d(in_channels=ed, out_channels=conv_dim, kernel_size=1, stride=1, padding=0, bias=False))
        self.conv2 = nn.Sequential(DepthwiseSeparableConv2d(in_channels=conv_dim, out_channels=conv_dim, kernel_size=3, stride=1, padding=1))
        self.conv3 = nn.Sequential(nn.BatchNorm2d(conv_dim), torch.nn.Conv2d(in_channels=conv_dim, out_channels=ed, kernel_size=1, stride=1, padding=0, bias=False))
        self.sa = nn.Sequential(nn.Conv2d(conv_dim, 1, kernel_size=1, stride=1, padding=0, bias=False), nn.Conv2d(1, 1, kernel_size=1, stride=1, padding=0, bias=False))
        self.sigmoid = nn.Sigmoid()
        self.ffn0_conv = Residual(FFN(conv_dim, int(conv_dim * 2)))
        self.ffn1_conv = Residual(FFN(ed, int(conv_dim)))

    def forward(self, x):
        x2 = self.conv(x)
        x_ = self.conv2(x2)
        weights = self.sigmoid(self.sa(x2))
        x2 = x_ * weights + x2
        x1 = self.conv3(x2)
        x = x1

        x = self.ffn0(self.dw0(x))
        xk = x
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        x2 = self.act(x)
        x = x2 * self.sigmoid(y) + xk
        x = self.ffn1(self.dw1(x))
        return x


## dense
class DPFL(torch.nn.Module):
    def __init__(self, stg,
                 ed, kd, nh=8,
                 ar=4,
                 resolution=14,
                 wd=7,
                 kernels=[5, 5, 5, 5], deth=1, ga=64):
        super().__init__()
        for i in range(deth):
            layer = A3Conv(stg, ed + ga * i, ga, kd, nh, ar, resolution, wd, kernels)
            self.add_module("denselayer%d" % (i + 1), layer)

    def forward(self, x):
        features = x
        for name, layer in self._modules.items():
            new_features = layer(features)
            features = torch.cat((features, new_features), dim=1)
        return features


class SMFA(torch.nn.Module):
    def __init__(self, in_dim, ):
        super().__init__()
        hid_dim = in_dim
        self.def_c = in_dim
        self.def_dim = self.def_c // 2
        self.conv = torch.nn.Sequential(Conv2d_BN(self.def_c, self.def_dim, ks=1, stride=1, pad=0))
        self.conv1 = torch.nn.Sequential(Conv2d_BN(self.def_dim, hid_dim, ks=1, stride=1, pad=0),  DepthwiseSeparableConv2d(in_channels=hid_dim, out_channels=hid_dim, kernel_size=3, stride=1, padding=1), nn.ReLU6(inplace=True), Conv2d_BN(hid_dim, self.def_dim, ks=1, stride=1, pad=0))
        self.conv2 = torch.nn.Sequential(Conv2d_BN(self.def_dim, hid_dim, ks=1, stride=1, pad=0),  DepthwiseSeparableConv2d(in_channels=hid_dim, out_channels=hid_dim, kernel_size=5, stride=1, padding=2), nn.ReLU6(inplace=True), Conv2d_BN(hid_dim, self.def_dim, ks=1, stride=1, pad=0))
        self.conv3 = torch.nn.Sequential(Conv2d_BN(self.def_dim, hid_dim, ks=1, stride=1, pad=0),  DepthwiseSeparableConv2d(in_channels=hid_dim, out_channels=hid_dim, kernel_size=7, stride=1, padding=3), nn.ReLU6(inplace=True), Conv2d_BN(hid_dim, self.def_dim, ks=1, stride=1, pad=0))
        self.conv0 = torch.nn.Sequential(Conv2d_BN(self.def_dim, hid_dim, ks=1, stride=1, pad=0),  DepthwiseSeparableConv2d(in_channels=hid_dim, out_channels=hid_dim, kernel_size=1, stride=1, padding=0), nn.ReLU6(inplace=True), Conv2d_BN(hid_dim, self.def_dim, ks=1, stride=1, pad=0))

        self.defcn0 = Residual(torch.nn.Sequential(DeformConv2D(self.def_dim, self.def_dim, kernel_size=1, padding=0)))
        self.defcn1 = torch.nn.Sequential(DeformConv2D(self.def_dim, self.def_dim, kernel_size=3, padding=1))
        self.defcn2 = Residual(torch.nn.Sequential(DeformConv2D(self.def_dim, self.def_dim, kernel_size=5, padding=1)))
        self.defcn3 = Residual(torch.nn.Sequential(DeformConv2D(self.def_dim, self.def_dim, kernel_size=7, padding=1)))
        self.ffu = nn.Sequential(FFN(self.def_dim * 5, self.def_dim * 5 * 2), Conv2d_BN(self.def_dim * 5, self.def_dim,3,1,1, resolution=0), torch.nn.ReLU6(inplace=True))
        self.cfu = nn.Sequential(Conv2d_BN(self.def_dim, self.def_c, 3, 1, 1, resolution=0), torch.nn.ReLU6(inplace=True))

    def forward(self, x):
        xk = x
        x = self.conv(x)
        x0 = self.conv0(x)
        # x0 = self.defcn0(x0)
        x1 = self.conv1(x)
        # x1  = self.defcn1(x1)
        x2 = self.conv2(x)
        # x2 = self.defcn2(x2)
        x3 = self.conv3(x)
        # x3 = self.defcn3(x3)
        x_cat = torch.cat((x0, x1, x2, x3, x), dim=1)
        x = self.ffu(x_cat)
        x = self.defcn1(x)
        x = self.cfu(x) + xk

        return x


class Net(torch.nn.Module):
    def __init__(self, img_size=224,
                 patch_size=4,
                 in_chans=3,
                 num_classes=1000,
                 stages=['s', 's', 's', 's'],
                 embed_dim=[128, 192, 256],
                 key_dim=[8, 8, 8, 8],
                 depth=[1, 2, 3],
                 num_heads=[4, 4, 4],
                 window_size=[7, 7, 7],
                 kernels=[5, 5, 5, 5],
                 dpth_conv=[1, 1, 1, 1],
                 dpth_vit=[1, 1, 1, 1],
                 down_ops=[['subsample', 2], ['subsample', 2], ['subsample', 2], ['', 2]],
                 distillation=False,
                 ga=32):
        super().__init__()
        resolution = img_size
        # Patch embedding
        self.patch_embed1 = torch.nn.Sequential(nn.BatchNorm2d(in_chans),  nn.Conv2d(in_chans, embed_dim[0] // 4, 3, 2, 1), torch.nn.ReLU6(inplace=True), )
        self.patch_embed2 = torch.nn.Sequential(nn.BatchNorm2d(embed_dim[0] // 8),  nn.Conv2d(embed_dim[0] // 8, embed_dim[0] // 4, 3, 2, 1), torch.nn.ReLU6(inplace=True), )
        self.patch_embed3 = torch.nn.Sequential(nn.BatchNorm2d(embed_dim[0] // 4),  nn.Conv2d(embed_dim[0] // 4, embed_dim[0] // 2, 3, 2, 1), torch.nn.ReLU6(inplace=True), )
        self.patch_embed4 = torch.nn.Sequential(nn.BatchNorm2d(embed_dim[0] // 2),  nn.Conv2d(embed_dim[0] // 2, embed_dim[0], 3, 2, 1))

        resolution = img_size // patch_size
        attn_ratio = [ga / (key_dim[i] * num_heads[i]) for i in range(len(embed_dim))]
        self.blocks1 = []
        self.blocks2 = []
        self.blocks3 = []
        self.blocks4 = []
        self.subsample1 = []
        self.subsample2 = []
        self.subsample3 = []
        dim_ = embed_dim[0]
        # Build DPFL
        for i, (stg, ed, kd, dpth, nh, ar, wd, do, dc, dv) in enumerate(zip(stages, embed_dim, key_dim, depth, num_heads, attn_ratio, window_size, down_ops, dpth_conv, dpth_vit)):
            eval('self.blocks' + str(i + 1)).append(DPFL(stg, dim_, kd, nh, ar, resolution, wd, kernels, dpth, ga))
            dim_ = dim_ + ga * dpth
            if do[0] == 'subsample':
                blk = eval('self.blocks' + str(i + 1))
                blk.append(PatchMerging(dim_, dim_, resolution))
                resolution = (resolution - 1) // do[1] + 1

        self.blocks1 = torch.nn.Sequential(*self.blocks1)
        self.blocks2 = torch.nn.Sequential(*self.blocks2)
        self.blocks3 = torch.nn.Sequential(*self.blocks3)
        self.blocks4 = torch.nn.Sequential(*self.blocks4)
        self.bn = nn.Sequential(Residual(FFN(dim_, dim_ * 2)), Conv2d_BN(dim_, dim_, 3, 1, 1), torch.nn.ReLU6(inplace=True))

        self.SMFA = SMFA(embed_dim[0] // 1)

        self.head = BN_Linear(dim_, num_classes) if num_classes > 0 else torch.nn.Identity()
        self.distillation = distillation
        if distillation:
            self.head_dist = BN_Linear(embed_dim[-1], num_classes) if num_classes > 0 else torch.nn.Identity()

    @torch.jit.ignore
    def no_weight_decay(self):
        return {x for x in self.state_dict().keys() if 'attention_biases' in x}

    def forward(self, x):
        x = self.patch_embed1(x)
        x = self.patch_embed3(x)
        x = self.patch_embed4(x)

        x = self.SMFA(x)
        xg = torch.flatten(x, 1)

        x = self.blocks1(x)
        x = self.blocks2(x)
        x = self.blocks3(x)
        x = self.blocks4(x)

        x = self.bn(x)
        x = torch.nn.functional.adaptive_avg_pool2d(x, 1).flatten(1)
        if self.distillation:
            x = self.head(x), self.head_dist(x)
            if not self.training:
                x = (x[0] + x[1]) / 2
        else:
            x = self.head(x)
        return x, xg


class DeformConv2D(nn.Module):
    def __init__(self, inc, outc, kernel_size=3, padding=1, stride=1, bias=None, lr_ratio=1.0):
        super(DeformConv2D, self).__init__()
        self.kernel_size = kernel_size
        self.padding = padding
        self.stride = stride
        self.zero_padding = nn.ZeroPad2d(padding)

        self.offset_conv = nn.Conv2d(inc, 2 * kernel_size * kernel_size, kernel_size=3, padding=1, stride=stride)
        nn.init.constant_(self.offset_conv.weight, 0)  # the offset learning are initialized with zero weights
        self.offset_conv.register_full_backward_hook(self._set_lr)

        self.conv = nn.Conv2d(inc, outc, kernel_size=kernel_size, stride=kernel_size, bias=bias)

        self.lr_ratio = lr_ratio

    def _set_lr(self, module, grad_input, grad_output):
        # print('grad input:', grad_input)
        new_grad_input = []

        for i in range(len(grad_input)):
            if grad_input[i] is not None:
                new_grad_input.append(grad_input[i] * self.lr_ratio)
            else:
                new_grad_input.append(grad_input[i])

        new_grad_input = tuple(new_grad_input)
        # print('new grad input:', new_grad_input)
        return new_grad_input

    def forward(self, x):
        offset = self.offset_conv(x)
        dtype = offset.data.type()
        ks = self.kernel_size
        N = offset.size(1) // 2

        # Change offset's order from [x1, x2, ..., y1, y2, ...] to [x1, y1, x2, y2, ...]
        # Codes below are written to make sure same results of MXNet implementation.
        # You can remove them, and it won't influence the module's performance.
        offsets_index = torch.cat([torch.arange(0, 2 * N, 2), torch.arange(1, 2 * N + 1, 2)]).type_as(x).long()
        offsets_index.requires_grad = False
        offsets_index = offsets_index.unsqueeze(dim=0).unsqueeze(dim=-1).unsqueeze(dim=-1).expand(*offset.size())
        offset = torch.gather(offset, dim=1, index=offsets_index)
        # ------------------------------------------------------------------------

        if self.padding:
            x = self.zero_padding(x)

        # (b, 2N, h, w)
        p = self._get_p(offset, dtype)

        # (b, h, w, 2N)
        p = p.contiguous().permute(0, 2, 3, 1)

        """
            if q is float, using bilinear interpolate, it has four integer corresponding position.
            The four position is left top, right top, left bottom, right bottom, defined as q_lt, q_rb, q_lb, q_rt
        """
        # (b, h, w, 2N)
        q_lt = p.detach().floor()

        """
            Because the shape of x is N, b, h, w, the pixel position is (y, x)
            *┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄→y
            ┊  .(y, x)   .(y+1, x)
            ┊   
            ┊  .(y, x+1) .(y+1, x+1)
            ┊
            ↓
            x

            For right bottom point, it'x = left top'y + 1, it'y = left top'y + 1
        """
        q_rb = q_lt + 1

        """
            x.size(2) is h, x.size(3) is w, make 0 <= p_y <= h - 1, 0 <= p_x <= w-1
        """
        q_lt = torch.cat([torch.clamp(q_lt[..., :N], 0, x.size(2) - 1), torch.clamp(q_lt[..., N:], 0, x.size(3) - 1)],
                         dim=-1).long()

        """
            x.size(2) is h, x.size(3) is w, make 0 <= p_y <= h - 1, 0 <= p_x <= w-1
        """
        q_rb = torch.cat([torch.clamp(q_rb[..., :N], 0, x.size(2) - 1), torch.clamp(q_rb[..., N:], 0, x.size(3) - 1)],
                         dim=-1).long()

        """
            For the left bottom point, it'x is equal to right bottom, it'y is equal to left top
            Therefore, it's y is from q_lt, it's x is from q_rb
        """
        q_lb = torch.cat([q_lt[..., :N], q_rb[..., N:]], -1)

        """
            y from q_rb, x from q_lt
            For right top point, it's x is equal t to left top, it's y is equal to right bottom 
        """
        q_rt = torch.cat([q_rb[..., :N], q_lt[..., N:]], -1)

        """
            find p_y <= padding or p_y >= h - 1 - padding, find p_x <= padding or p_x >= x - 1 - padding
            This is to find the points in the area where the pixel value is meaningful.
        """
        # (b, h, w, N)
        mask = torch.cat([p[..., :N].lt(self.padding) + p[..., :N].gt(x.size(2) - 1 - self.padding),
                          p[..., N:].lt(self.padding) + p[..., N:].gt(x.size(3) - 1 - self.padding)], dim=-1).type_as(p)
        mask = mask.detach()
        # print('mask:', mask)

        floor_p = torch.floor(p)
        # print('floor_p = ', floor_p)

        """
           when mask is 1, take floor_p;
           when mask is 0, take original p.
           When thr point in the padding area, interpolation is not meaningful and we can take the nearest
           point which is the most possible to have meaningful value.
        """
        p = p * (1 - mask) + floor_p * mask
        p = torch.cat([torch.clamp(p[..., :N], 0, x.size(2) - 1), torch.clamp(p[..., N:], 0, x.size(3) - 1)], dim=-1)

        """
            In the paper, G(q, p) = g(q_x, p_x) * g(q_y, p_y)
            g(a, b) = max(0, 1-|a-b|)
        """
        # bilinear kernel (b, h, w, N)
        g_lt = (1 + (q_lt[..., :N].type_as(p) - p[..., :N])) * (1 + (q_lt[..., N:].type_as(p) - p[..., N:]))
        g_rb = (1 - (q_rb[..., :N].type_as(p) - p[..., :N])) * (1 - (q_rb[..., N:].type_as(p) - p[..., N:]))
        g_lb = (1 + (q_lb[..., :N].type_as(p) - p[..., :N])) * (1 - (q_lb[..., N:].type_as(p) - p[..., N:]))
        g_rt = (1 - (q_rt[..., :N].type_as(p) - p[..., :N])) * (1 + (q_rt[..., N:].type_as(p) - p[..., N:]))

        # print('g_lt size is ', g_lt.size())
        # print('g_lt unsqueeze size:', g_lt.unsqueeze(dim=1).size())

        # (b, c, h, w, N)
        x_q_lt = self._get_x_q(x, q_lt, N)
        x_q_rb = self._get_x_q(x, q_rb, N)
        x_q_lb = self._get_x_q(x, q_lb, N)
        x_q_rt = self._get_x_q(x, q_rt, N)

        """
            In the paper, x(p) = ΣG(p, q) * x(q), G is bilinear kernal
        """
        # (b, c, h, w, N)
        x_offset = g_lt.unsqueeze(dim=1) * x_q_lt + \
                   g_rb.unsqueeze(dim=1) * x_q_rb + \
                   g_lb.unsqueeze(dim=1) * x_q_lb + \
                   g_rt.unsqueeze(dim=1) * x_q_rt

        """
            x_offset is kernel_size * kernel_size(N) times x. 
        """
        x_offset = self._reshape_x_offset(x_offset, ks)

        out = self.conv(x_offset)
        return out

    def _get_p_n(self, N, dtype):
        """
            In torch 0.4.1 grid_x, grid_y = torch.meshgrid([x, y])
            In torch 1.0   grid_x, grid_y = torch.meshgrid(x, y)
        """
        p_n_x, p_n_y = torch.meshgrid(
            [torch.arange(-(self.kernel_size - 1) // 2, (self.kernel_size - 1) // 2 + 1),
             torch.arange(-(self.kernel_size - 1) // 2, (self.kernel_size - 1) // 2 + 1)])
        # (2N, 1)
        p_n = torch.cat([torch.flatten(p_n_x), torch.flatten(p_n_y)], 0)
        p_n = p_n.view(1, 2 * N, 1, 1).type(dtype)
        p_n.requires_grad = False
        # print('requires_grad:', p_n.requires_grad)

        return p_n

    def _get_p_0(self, h, w, N, dtype):
        p_0_x, p_0_y = torch.meshgrid([
            torch.arange(1, h * self.stride + 1, self.stride),
            torch.arange(1, w * self.stride + 1, self.stride)])
        p_0_x = torch.flatten(p_0_x).view(1, 1, h, w).repeat(1, N, 1, 1)
        p_0_y = torch.flatten(p_0_y).view(1, 1, h, w).repeat(1, N, 1, 1)
        p_0 = torch.cat([p_0_x, p_0_y], 1).type(dtype)
        p_0.requires_grad = False

        return p_0

    def _get_p(self, offset, dtype):
        N, h, w = offset.size(1) // 2, offset.size(2), offset.size(3)

        # (1, 2N, 1, 1)
        p_n = self._get_p_n(N, dtype)

        # (1, 2N, h, w)
        p_0 = self._get_p_0(h, w, N, dtype)

        p = p_0 + p_n + offset
        return p

    def _get_x_q(self, x, q, N):
        b, h, w, _ = q.size()
        padded_w = x.size(3)
        c = x.size(1)
        # (b, c, h*w)
        x = x.contiguous().view(b, c, -1)

        # (b, h, w, N)
        index = q[..., :N] * padded_w + q[..., N:]  # offset_x*w + offset_y
        # (b, c, h*w*N)
        index = index.contiguous().unsqueeze(dim=1).expand(-1, c, -1, -1, -1).contiguous().view(b, c, -1)

        x_offset = x.gather(dim=-1, index=index).contiguous().view(b, c, h, w, N)

        return x_offset

    @staticmethod
    def _reshape_x_offset(x_offset, ks):
        b, c, h, w, N = x_offset.size()
        x_offset = torch.cat([x_offset[..., s:s + ks].contiguous().view(b, c, h, w * ks) for s in range(0, N, ks)],
                             dim=-1)
        x_offset = x_offset.contiguous().view(b, c, h * ks, w * ks)

        return x_offset