import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule, xavier_init

from mmdet.core import auto_fp16
from ..builder import NECKS
import torch

from mmdet.models.backbones.effnet.utils import MemoryEfficientSwish, Swish, SeparableConvBlock
from mmdet.models.backbones.effnet.utils_extra import Conv2dStaticSamePadding, MaxPool2dStaticSamePadding


@NECKS.register_module
class BiFPN(nn.Module):
    def __init__(self,
                in_channels,
                out_channels,
                num_outs,
                stack=1,
                onnx_export=False, 
                attention=True):
        super(BiFPN, self).__init__()
        assert isinstance(in_channels, list)
        self.in_channels = in_channels
        self.out_channels = out_channels #[b0:64,b1:88,b2:112,b3:160,b4:224,b5:288,b6:384,b7:384]
        self.num_ins = len(in_channels)
        self.num_outs = num_outs
        self.stack = stack #[b0:3,b1:4,b2:5,b3:6,b4:7,b5:7,b6:8,b7:8]

        self.stack_bifpn_convs = nn.ModuleList()
        for ii in range(stack):
            self.stack_bifpn_convs.append(BiFPNModule(
                                                in_channels = in_channels,
                                                out_channels=out_channels,
                                                num_outs=self.num_outs,
                                                first_time=True if ii == 0 else False, 
                                                onnx_export=False, 
                                                attention=True) )

        # self.init_weights()


    # default init_weights for conv(msra) and norm in ConvModule
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                xavier_init(m, distribution='uniform')
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, inputs):
        assert len(inputs) == len(self.in_channels)
        # for i,input in enumerate(inputs):
        #     print('the P{} input size is {}'.format(i+3,input.size()))

        laterals = inputs

        # build top-down and down-top path with stack
        for bifpn_module in self.stack_bifpn_convs:
            laterals = bifpn_module(laterals)
        outs = laterals

        return tuple(outs)


class BiFPNModule(nn.Module):
    def __init__(self,
                in_channels,
                out_channels,
                num_outs,
                first_time=False,
                epsilon=0.0001,
                onnx_export=False, 
                attention=True):
        super(BiFPNModule, self).__init__()
        self.epsilon = epsilon
        self.num_outs = num_outs
        self.attention = attention
        self.in_channels = in_channels
        self.first_time = first_time
        if self.first_time:
            self.p5_to_p6 = nn.Sequential(
                Conv2dStaticSamePadding(self.in_channels[-1], out_channels, 1),
                nn.BatchNorm2d(out_channels, momentum=0.01, eps=1e-3),
                MaxPool2dStaticSamePadding(3, 2)
            )
            self.p6_to_p7 = nn.Sequential(
                MaxPool2dStaticSamePadding(3, 2)
            )

            self.p3_down_channel = nn.Sequential(
                Conv2dStaticSamePadding(self.in_channels[-3], out_channels, 1),
                nn.BatchNorm2d(out_channels, momentum=0.01, eps=1e-3),
            )
            self.p4_down_channel = nn.Sequential(
                Conv2dStaticSamePadding(self.in_channels[-2], out_channels, 1),
                nn.BatchNorm2d(out_channels, momentum=0.01, eps=1e-3),
            )
            self.p5_down_channel = nn.Sequential(
                Conv2dStaticSamePadding(self.in_channels[-1], out_channels, 1),
                nn.BatchNorm2d(out_channels, momentum=0.01, eps=1e-3),
            )

            self.p4_down_channel_2 = nn.Sequential(
                Conv2dStaticSamePadding(self.in_channels[-2], out_channels, 1),
                nn.BatchNorm2d(out_channels, momentum=0.01, eps=1e-3),
            )
            self.p5_down_channel_2 = nn.Sequential(
                Conv2dStaticSamePadding(self.in_channels[-1], out_channels, 1),
                nn.BatchNorm2d(out_channels, momentum=0.01, eps=1e-3),
            )

        # Conv layers
        self.conv6_up = SeparableConvBlock(out_channels, onnx_export=onnx_export)
        self.conv5_up = SeparableConvBlock(out_channels, onnx_export=onnx_export)
        self.conv4_up = SeparableConvBlock(out_channels, onnx_export=onnx_export)
        self.conv3_up = SeparableConvBlock(out_channels, onnx_export=onnx_export)
        self.conv4_down = SeparableConvBlock(out_channels, onnx_export=onnx_export)
        self.conv5_down = SeparableConvBlock(out_channels, onnx_export=onnx_export)
        self.conv6_down = SeparableConvBlock(out_channels, onnx_export=onnx_export)
        self.conv7_down = SeparableConvBlock(out_channels, onnx_export=onnx_export)

        # Feature scaling layers
        self.p6_upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.p5_upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.p4_upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.p3_upsample = nn.Upsample(scale_factor=2, mode='nearest')

        self.p4_downsample = MaxPool2dStaticSamePadding(3, 2)
        self.p5_downsample = MaxPool2dStaticSamePadding(3, 2)
        self.p6_downsample = MaxPool2dStaticSamePadding(3, 2)
        self.p7_downsample = MaxPool2dStaticSamePadding(3, 2)

        self.swish = MemoryEfficientSwish() if not onnx_export else Swish()
        
        # Weight
        self.p6_w1 = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
        self.p6_w1_relu = nn.ReLU()
        self.p5_w1 = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
        self.p5_w1_relu = nn.ReLU()
        self.p4_w1 = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
        self.p4_w1_relu = nn.ReLU()
        self.p3_w1 = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
        self.p3_w1_relu = nn.ReLU()

        self.p4_w2 = nn.Parameter(torch.ones(3, dtype=torch.float32), requires_grad=True)
        self.p4_w2_relu = nn.ReLU()
        self.p5_w2 = nn.Parameter(torch.ones(3, dtype=torch.float32), requires_grad=True)
        self.p5_w2_relu = nn.ReLU()
        self.p6_w2 = nn.Parameter(torch.ones(3, dtype=torch.float32), requires_grad=True)
        self.p6_w2_relu = nn.ReLU()
        self.p7_w2 = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
        self.p7_w2_relu = nn.ReLU()

        self.attention = attention

    def forward(self, inputs):
        #assert len(inputs) == self.num_outs
        if self.attention:
            return self._forward_fast_attention(inputs)
        else:
            return self._forward(inputs)

    def _forward_fast_attention(self, inputs):
        if self.first_time:
            _,p3,p4,p5 = inputs
            p3_in = self.p3_down_channel(p3)
            p4_in = self.p4_down_channel(p4)
            p5_in = self.p5_down_channel(p5)


            p6_in = self.p5_to_p6(p5)
            p7_in = self.p6_to_p7(p6_in)
        else:
            p3_in, p4_in, p5_in, p6_in, p7_in = inputs

        print("p3_in size is ",p3_in.size())
        print("p4_in size is ",p4_in.size())
        print("p5_in size is ",p5_in.size())
        print("p6_in size is ",p6_in.size())
        print("p7_in size is ",p7_in.size())


        # P7_0 to P7_2
        
        # top-down
        # Weights for P6_0 and P7_0 to P6_1
        p6_w1 = self.p6_w1_relu(self.p6_w1)
        weight = p6_w1 / (torch.sum(p6_w1, dim=0) + self.epsilon)
        # Connections for P6_0 and P7_0 to P6_1 respectively
        p6_up = self.conv6_up(self.swish(weight[0] * p6_in + weight[1] * self.p6_upsample(p7_in))) #p6_up = p6_in + p7_in

        # Weights for P5_0 and P6_0 to P5_1
        p5_w1 = self.p5_w1_relu(self.p5_w1)
        weight = p5_w1 / (torch.sum(p5_w1, dim=0) + self.epsilon)
        # Connections for P5_0 and P6_0 to P5_1 respectively
        p5_up = self.conv5_up(self.swish(weight[0] * p5_in + weight[1] * self.p5_upsample(p6_up))) #p5_up = p5_in + p6_up

        # Weights for P4_0 and P5_0 to P4_1
        p4_w1 = self.p4_w1_relu(self.p4_w1)
        weight = p4_w1 / (torch.sum(p4_w1, dim=0) + self.epsilon)
        # Connections for P4_0 and P5_0 to P4_1 respectively
        p4_up = self.conv4_up(self.swish(weight[0] * p4_in + weight[1] * self.p4_upsample(p5_up))) #p4_up = p4_in + p5_up

        # Weights for P3_0 and P4_1 to P3_2
        p3_w1 = self.p3_w1_relu(self.p3_w1)
        weight = p3_w1 / (torch.sum(p3_w1, dim=0) + self.epsilon)
        # Connections for P3_0 and P4_1 to P3_2 respectively
        p3_out = self.conv3_up(self.swish(weight[0] * p3_in + weight[1] * self.p3_upsample(p4_up))) #p3_out = p3_in + p4_up

        if self.first_time:
            p4_in = self.p4_down_channel_2(p4)
            p5_in = self.p5_down_channel_2(p5)

        #down-top
        # Weights for P4_0, P4_1 and P3_2 to P4_2
        p4_w2 = self.p4_w2_relu(self.p4_w2)
        weight = p4_w2 / (torch.sum(p4_w2, dim=0) + self.epsilon)
        # Connections for P4_0, P4_1 and P3_2 to P4_2 respectively
        p4_out = self.conv4_down(
            self.swish(weight[0] * p4_in + weight[1] * p4_up + weight[2] * self.p4_downsample(p3_out))) #p4_out = p4_in + p4_up + p3_out

        # Weights for P5_0, P5_1 and P4_2 to P5_2
        p5_w2 = self.p5_w2_relu(self.p5_w2)
        weight = p5_w2 / (torch.sum(p5_w2, dim=0) + self.epsilon)
        # Connections for P5_0, P5_1 and P4_2 to P5_2 respectively
        p5_out = self.conv5_down(
            self.swish(weight[0] * p5_in + weight[1] * p5_up + weight[2] * self.p5_downsample(p4_out))) #p5_out = p5_in + p5_up + p4_out

        # Weights for P6_0, P6_1 and P5_2 to P6_2
        p6_w2 = self.p6_w2_relu(self.p6_w2)
        weight = p6_w2 / (torch.sum(p6_w2, dim=0) + self.epsilon)
        # Connections for P6_0, P6_1 and P5_2 to P6_2 respectively
        p6_out = self.conv6_down(
            self.swish(weight[0] * p6_in + weight[1] * p6_up + weight[2] * self.p6_downsample(p5_out))) #p6_out = p6_in + p6_up + p5_out

        # Weights for P7_0 and P6_2 to P7_2
        p7_w2 = self.p7_w2_relu(self.p7_w2)
        weight = p7_w2 / (torch.sum(p7_w2, dim=0) + self.epsilon)
        # Connections for P7_0 and P6_2 to P7_2
        p7_out = self.conv7_down(self.swish(weight[0] * p7_in + weight[1] * self.p7_downsample(p6_out))) #p7_out = p7_in + p6_out

        return p3_out, p4_out, p5_out, p6_out, p7_out

    def _forward(self, inputs):
        if self.first_time:
            _,p3,p4,p5 = inputs
            p3_in = self.p3_down_channel(p3)
            p5_in = self.p5_down_channel(p5)
            p4_in = self.p4_down_channel(p4)

            p6_in = self.p5_to_p6(p5)
            p7_in = self.p6_to_p7(p6_in)
        else:
            p3_in, p4_in, p5_in, p6_in, p7_in = inputs

        # P7_0 to P7_2

        # Connections for P6_0 and P7_0 to P6_1 respectively
        p6_up = self.conv6_up(self.swish(p6_in + self.p6_upsample(p7_in)))

        # Connections for P5_0 and P6_0 to P5_1 respectively
        p5_up = self.conv5_up(self.swish(p5_in + self.p5_upsample(p6_up)))

        # Connections for P4_0 and P5_0 to P4_1 respectively
        p4_up = self.conv4_up(self.swish(p4_in + self.p4_upsample(p5_up)))

        # Connections for P3_0 and P4_1 to P3_2 respectively
        p3_out = self.conv3_up(self.swish(p3_in + self.p3_upsample(p4_up)))

        if self.first_time:
            p4_in = self.p4_down_channel_2(p4)
            p5_in = self.p5_down_channel_2(p5)

        # Connections for P4_0, P4_1 and P3_2 to P4_2 respectively
        p4_out = self.conv4_down(
            self.swish(p4_in + p4_up + self.p4_downsample(p3_out)))

        # Connections for P5_0, P5_1 and P4_2 to P5_2 respectively
        p5_out = self.conv5_down(
            self.swish(p5_in + p5_up + self.p5_downsample(p4_out)))

        # Connections for P6_0, P6_1 and P5_2 to P6_2 respectively
        p6_out = self.conv6_down(
            self.swish(p6_in + p6_up + self.p6_downsample(p5_out)))

        # Connections for P7_0 and P6_2 to P7_2
        p7_out = self.conv7_down(self.swish(p7_in + self.p7_downsample(p6_out)))

        return p3_out, p4_out, p5_out, p6_out, p7_out

# class ConvModuleSSP(nn.Module):
#     def __init__(self, 
#                  in_channels, 
#                  out_channels, 
#                  kernel_size, 
#                  stride=1, 
#                  bias=True, 
#                  groups=1, 
#                  dilation=1,
#                  **kwargs):
#         super(ConvModuleSSP,self).__init__()
#         self.conv = Conv2dStaticSamePadding(in_channels, out_channels, kernel_size)
#         self.norm = nn.BatchNorm2d(out_channels, momentum=0.01, eps=1e-3)
#         self.act = MemoryEfficientSwish()

#     def forward(self,x,norm=True,act=False):
#         x = self.conv(x)
#         if norm:
#             x = self.norm(x)
#         if act:
#             x = self.act(x)
#         return x



# @NECKS.register_module()
# class BiFPN(nn.Module):
#     """BiFPN.

#     BiFPN: EfficientDet: Scalable and Efficient Object Detection. 
#     (https://arxiv.org/abs/1911.09070)
#     """

#     def __init__(self,
#                  in_channels,
#                  out_channels,
#                  num_outs,
#                  stack_times,
#                  start_level=0,
#                  end_level=-1,
#                  add_extra_convs=False,
#                  norm_cfg=None):
#         super(BiFPN, self).__init__()
#         assert isinstance(in_channels, list)
#         self.in_channels = in_channels
#         self.out_channels = out_channels
#         self.num_ins = len(in_channels)  # num of input feature levels
#         self.num_outs = num_outs  # num of output feature levels
#         self.stack_times = stack_times
#         self.norm_cfg = norm_cfg

#         if end_level == -1:
#             self.backbone_end_level = self.num_ins
#             assert num_outs >= self.num_ins - start_level
#         else:
#             # if end_level < inputs, no extra level is allowed
#             self.backbone_end_level = end_level
#             assert end_level <= len(in_channels)
#             assert num_outs == end_level - start_level
#         self.start_level = start_level
#         self.end_level = end_level
#         self.add_extra_convs = add_extra_convs

#         print('bifpn backbone_end_level is ',self.backbone_end_level)

#         # add lateral connections
#         self.lateral_convs = nn.ModuleList()
#         for i in range(self.start_level, self.backbone_end_level):
#             l_conv = ConvModuleSSP(
#                 in_channels[i],
#                 out_channels,
#                 1)
#             self.lateral_convs.append(l_conv)

        
#         # add extra downsample layers (stride-2 pooling or conv)
#         extra_levels = num_outs - self.backbone_end_level + self.start_level
#         self.extra_downsamples = nn.ModuleList()
#         for i in range(extra_levels):
#             extra_conv = ConvModuleSSP(
#                 out_channels, out_channels, 1)
#             self.extra_downsamples.append(
#                 nn.Sequential(extra_conv, MaxPool2dStaticSamePadding(3, 2)))
        
#         # add BiFPN connections
#         '''
#         illustration of a minimal bifpn unit
#         P7_0 -------------------------> P7_2 -------->
#             |-------------|                ¡ü
#                         ¡ý                |
#         P6_0 ---------> P6_1 ---------> P6_2 -------->
#             |-------------|--------------¡ü ¡ü
#                         ¡ý                |
#         P5_0 ---------> P5_1 ---------> P5_2 -------->
#             |-------------|--------------¡ü ¡ü
#                         ¡ý                |
#         P4_0 ---------> P4_1 ---------> P4_2 -------->
#             |-------------|--------------¡ü ¡ü
#                         |--------------¡ý |
#         P3_0 -------------------------> P3_2 -------->
#         '''
#         self.bifpn_stages = nn.ModuleList()
#         for _ in range(self.stack_times):
#             stage = nn.ModuleDict()

#     # default init_weights for conv(msra) and norm in ConvModule
#     def init_weights(self):
#         for m in self.modules():
#             if isinstance(m, nn.Conv2d):
#                 xavier_init(m, distribution='uniform')
#             elif isinstance(m, nn.BatchNorm2d):
#                 m.weight.data.fill_(1)
#                 m.bias.data.zero_()
        
#     @auto_fp16()
#     def forward(self, inputs):
#         # build P3-P5
#         feats = [
#             lateral_conv(inputs[i + self.start_level])
#             for i, lateral_conv in enumerate(self.lateral_convs)
#         ]

#         # build P6-P7 on top of P5
#         for downsample in self.extra_downsamples:
#             feats.append(downsample(feats[-1]))
        
#         for i,feat in enumerate(feats):
#             print("the P{} feat size is {}".format(i+3,feat.size()))

#         p3, p4, p5, p6, p7 = feats