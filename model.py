import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
import utils
#前置网络
class VGG13_qz(nn.Module):
    def __init__(self, pertrain):
        super(VGG13_qz, self).__init__()
        # 感受器细胞
        self.stage1_g1 = self.make_layers_1(3, [64])
        self.stage1_g2 = self.make_layers_1(64, [64])
        # 池化
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        # 双极细胞
        self.stage2_sj1 = self.make_layers_1(64, [256])
        self.stage2_sj2 = self.make_layers_1(256, [256])
        # 水平细胞
        self.stage2_sp1 = self.make_layers_1(64, [128])
        self.stage2_sp2 = self.make_layers_1(128, [128])
        # 双极-水平融合
        self.stage2_spj1 = self.make_layers_2(256, [128])
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        # 无长突细胞
        self.stage_sj1 = self.make_layers_1(128, [256])
        self.stage_sj2 = self.make_layers_1(256, [256])
        # 神经节细胞
        self.stage_sp1 = self.make_layers_1(128, [512])
        self.stage_sp2 = self.make_layers_1(512, [512])
        # 无长突-神经节融合
        self.stage_spj1 = self.make_layers_2(512, [256])

        if pertrain: 
            vgg_state_dict = torch.load(pertrain)
            vgg_name = []
            for name, par in vgg_state_dict.items():
                vgg_name.append(name)
            own_state_dict = self.state_dict()
            for name, param in own_state_dict.items():
                if name in vgg_name:
                    param.copy_(vgg_state_dict[name])
                    print('Successfully copied parameters: ', name)
                else:
                    if 'bias' in name:
                        param.zero_()
                    else:
                        param.normal_(0, 1e-2)
        else:
            self._initialize_weights()
            print('Failed to copy the existing network parameters!')

    def forward(self, x):
        stage1_g1 = self.stage1_g1(x) 
        stage1_g2 = self.stage1_g2(stage1_g1) 
        pool1 = self.pool1(stage1_g2)
        stage2_sj1 = self.stage2_sj1(pool1)  
        stage2_sj2 = self.stage2_sj2(stage2_sj1)  
        stage2_sp1 = self.stage2_sp1(pool1)  
        stage2_sp2 = self.stage2_sp2(stage2_sp1)  
        stage2_sp = stage2_sp1 + stage2_sp2  
        stage2_spj1 = self.stage2_spj1(stage2_sj2)  
        stage2_spj1_1 = stage2_sp + stage2_spj1 
        pool2 = self.pool2(stage2_spj1_1)
        stage_sj1 = self.stage_sj1(pool2)  
        stage_sj2 = self.stage_sj2(stage_sj1)  
        stage_sp1 = self.stage_sp1(pool2) 
        stage_sp2 = self.stage_sp2(stage_sp1)  
        stage_sj = stage_sj1 + stage_sj2  
        stage_spj1 = self.stage_spj1(stage_sp2)  
        stage_spj1_1 = stage_spj1 + stage_sj  

        return stage1_g2, stage2_spj1_1, stage_spj1_1
    @staticmethod
    # 定义的实现卷积层的模块化函数
    def make_layers(in_channels, cfg, stride=1, rate=1):
        layers = []
        for v in cfg:
            if v == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                conv2d = nn.Conv2d(in_channels, v, kernel_size=5, padding=2, stride=stride, dilation=rate)
                layers += [conv2d, nn.ReLU(inplace=True)]
                in_channels = v
        return nn.Sequential(*layers)

    @staticmethod
    # 定义的实现卷积层的模块化函数
    def make_layers_1(in_channels, cfg, stride=1, rate=1):
        layers = []
        for v in cfg:
            if v == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1, stride=stride, dilation=rate)
                layers += [conv2d, nn.ReLU(inplace=True)]
                in_channels = v
        return nn.Sequential(*layers)

    @staticmethod
    # 定义的实现卷积层的模块化函数
    def make_layers_2(in_channels, cfg, stride=1, rate=1):
        layers = []
        for v in cfg:
            if v == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                conv2d = nn.Conv2d(in_channels, v, kernel_size=1, padding=0, stride=stride, dilation=rate)
                layers += [conv2d, nn.ReLU(inplace=True)]
                in_channels = v
        return nn.Sequential(*layers)
class VGG13(nn.Module):
    def __init__(self, pertrain):
        super(VGG13, self).__init__()
        self.stage1 = self.make_layers(3, [64])
        self.stage2 = self.make_layers(64, [64])

        self.stage3 = self.make_layers(64, ['M', 128])
        self.stage4 = self.make_layers(128, [128])

        self.stage5 = self.make_layers(128, ['M', 256])
        self.stage6 = self.make_layers(256, [256])
        self.stage7 = self.make_layers(256, [256])

        self.stage8 = self.make_layers(256, ['M', 512])
        self.stage9 = self.make_layers(512, [512])
        self.stage10 = self.make_layers(512, [512])

        self.stage11 = self.make_layers(512, ['M', 512])
        self.stage12 = self.make_layers(512, [512])
        self.stage13 = self.make_layers(512, [512])

        self._initialize_weights(pertrain)
        self.weight = nn.Parameter(torch.Tensor([0.]))

    def forward(self, x):
        stage1 = self.stage1(x[0])
        stage2 = self.stage2(stage1)
        stage3 = self.stage3(stage2)
        stage4 = self.stage4(stage3)
        stage5 = self.stage5(stage4+x[1]* self.weight.sigmoid())
        stage6 = self.stage6(stage5)
        stage7 = self.stage7(stage6)
        stage8 = self.stage8(stage7+x[2]*(1-self.weight.sigmoid()))
        stage9 = self.stage9(stage8)
        stage10 = self.stage10(stage9)
        stage11 = self.stage11(stage10)
        stage12 = self.stage12(stage11)
        stage13 = self.stage13(stage12)

        return stage1,stage2,stage3,stage4,stage5,stage6,stage7,stage8,stage9,stage10,stage11,stage12,stage13
    @staticmethod
    #定义的实现卷积层的模块化函数
    def make_layers(in_channels, cfg, stride=1, rate=1):
        layers = []
        for v in cfg:
            if v == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1, stride=stride, dilation=rate)
                layers += [conv2d, nn.ReLU(inplace=True)]
                in_channels = v
        return nn.Sequential(*layers)
    #权重初始化函数
    def _initialize_weights(self, dict_path):
        model_paramters = torch.load(dict_path)#读取的参数
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data = (model_paramters.popitem(last=False)[-1])
                m.bias.data = model_paramters.popitem(last=False)[-1]

#前置网络
class VGG13_qz1(nn.Module):
    def __init__(self,cfgs):
        super(VGG13_qz1, self).__init__()
        self.encodes = VGG13_qz(cfgs)
        self.level1 = Refine_block2_1((64, 256), 64, 4, 32, 16)  # 2上采样1不变
        self.level2 = Refine_block2_1((128, 256), 128, 2, 32, 16)  # 2上采样1不变
        self.level3 = nn.Conv2d(64, 3, kernel_size=1, padding=0)
        self.encode = VGG13(cfgs)
    def forward(self, x):
        stage_out = self.encodes(x)
        level1 = self.level1(stage_out[0], stage_out[2])
        level2 = self.level2(stage_out[1], stage_out[2])
        stage_outs1 = self.level3(level1)
        stage_outs = [stage_outs1, level2, stage_out[2]]
        stage = self.encode(stage_outs) 

        return  stage[1] , stage[3],  stage[6],  stage[9],   stage[12]
    @staticmethod
    # 定义的实现卷积层的模块化函数
    def make_layers_2(in_channels, cfg, stride=1, rate=1):
        layers = []
        for v in cfg:
            if v == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                conv2d = nn.Conv2d(in_channels, v, kernel_size=1, padding=0, stride=stride, dilation=rate)
                layers += [conv2d, nn.ReLU(inplace=True)]
                in_channels = v
        return nn.Sequential(*layers)
class adap_conv(nn.Module):
    def __init__(self, in_channels, out_channels, D, groups):
        super(adap_conv, self).__init__()
        self.conv = nn.Sequential(*[nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                                    nn.BatchNorm2d(out_channels),
                                    nn.ReLU(inplace=True)])
        self.weight = nn.Parameter(torch.Tensor([0.]))
    def forward(self, x):
        x = self.conv(x) * self.weight.sigmoid()
        # x = self.conv(x)
        return x
class Refine_block2_1(nn.Module):
    def __init__(self, in_channel, out_channel, factor, D, groups, require_grad=False):
        super(Refine_block2_1, self).__init__()
        self.pre_conv1 = adap_conv(in_channel[0], out_channel, D, groups)
        self.pre_conv2 = adap_conv(in_channel[1], out_channel, D, groups)
        self.deconv_weight = nn.Parameter(utils.bilinear_upsample_weights(factor, out_channel), requires_grad=require_grad)
        self.factor = factor
    def forward(self, *input):
        x1 = self.pre_conv1(input[0])
        x2 = self.pre_conv2(input[1])
        x2 = F.conv_transpose2d(x2, self.deconv_weight, stride=self.factor, padding=int(self.factor/2),
                                output_padding=(x1.size(2) - x2.size(2)*self.factor, x1.size(3) - x2.size(3)*self.factor))
        return x1 + x2

class super_pixels(nn.Module):
    def __init__(self, inplanes, factor):
        super(super_pixels, self).__init__()
        self.superpixels = nn.PixelShuffle(factor) #伸缩
        planes = int(inplanes/(factor*2))
        self.down_sample = nn.Conv2d(planes, 1, kernel_size=2, stride=2, padding=0)
    def forward(self, x):
        x = self.superpixels(x)
        x = self.down_sample(x)
        return x

class Pool_Conv_no_change(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(Pool_Conv_no_change, self).__init__()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.stage_pool = self.make_layers(in_channel, [out_channel])
        self.stage_no_change = self.make_layers_1(in_channel, [in_channel])
    def forward(self, input):
        pool = self.pool(input)
        stage_pool = self.stage_pool(pool)           
        stage_no_change = self.stage_no_change(input) 
        stage_no_with_1 = input                      
        stage_no_with_2 = input                       
        return stage_no_with_1, stage_no_with_2, stage_pool, stage_no_change

class decode1(nn.Module):
    def __init__(self):
        super(decode1, self).__init__()       
        self.j_w = nn.Conv2d(64, 32, kernel_size=1, padding=0)
        self.j_w1 = nn.Conv2d(128, 64, kernel_size=1, padding=0)
        self.j_w2 = nn.Conv2d(256, 128, kernel_size=1, padding=0)
        self.j_w3 = nn.Conv2d(512, 256, kernel_size=1, padding=0)

        self.level1 = Refine_block2_1((32, 64), 32, 2, 32, 16)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.stage1_pool1 = self.make_layers(64, [64])
        self.stage2_22 = self.make_layers_1(128, [64])

        self.level2 = Refine_block2_1((64, 128), 64, 2, 32, 16)

        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)  
        self.stage2_pool2 = self.make_layers(128, [128])
        self.stage3_33 = self.make_layers_1(256, [128])

        self.level3 = Refine_block2_1((128, 256), 128, 2, 32, 16)

        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.stage3_pool3 = self.make_layers(256, [256])
        self.stage4_44 = self.make_layers_1(512, [256])

        self.level4 = Refine_block2_1((256, 512), 256, 2, 32, 16)

        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2) 
        self.stage4_pool4 = self.make_layers(512, [512])
        self.stage5_55 = self.make_layers_1(512, [512])
   
        self.level2_11 = Refine_block2_1((32, 64), 32, 2, 32, 16)
        self.level2_22 = Refine_block2_1((64, 128), 64, 2, 32, 16)
        self.level2_33 = Refine_block2_1((128, 256), 128, 2, 32, 16)
        self.level2_44 = Refine_block2_1((256, 512), 256, 2, 32, 16)

        self.level2_2 = Refine_block2_1((32, 64), 32, 2, 32, 16)
        self.pool21 = nn.MaxPool2d(kernel_size=2, stride=2)  
        self.stage21_pool21 = self.make_layers(32, [64])
        self.stage21_22 = self.make_layers_1(64, [64])

        self.level2_3 = Refine_block2_1((64, 128), 64, 2, 32, 16)
        self.pool22 = nn.MaxPool2d(kernel_size=2, stride=2)  
        self.stage22_pool22 = self.make_layers(64, [128])
        self.stage22_23 = self.make_layers_1(128, [128])

        self.level2_4 = Refine_block2_1((128, 256), 128, 2, 32, 16) 
        self.pool23 = nn.MaxPool2d(kernel_size=2, stride=2) 
        self.stage23_pool23 = self.make_layers(128, [256])
        self.stage23_24 = self.make_layers_1(256, [256])

        self.level3_11 = Refine_block2_1((32, 64), 32, 2, 32, 16)
        self.level3_22 = Refine_block2_1((64, 128), 64, 2, 32, 16)
        self.level3_33 = Refine_block2_1((128, 256), 128, 2, 32, 16)

        self.level3_2 = Refine_block2_1((32, 64), 32, 2, 32, 16)
        self.pool31 = nn.MaxPool2d(kernel_size=2, stride=2) 
        self.stage31_pool31 = self.make_layers(32, [64])
        self.stage31_32 = self.make_layers_1(64, [64])

        self.level3_3 = Refine_block2_1((64, 128), 64, 2, 32, 16)
        self.pool32 = nn.MaxPool2d(kernel_size=2, stride=2)  
        self.stage32_pool32 = self.make_layers(64, [128])
        self.stage32_33 = self.make_layers_1(128, [128])

        self.level4_1 = Refine_block2_1((32, 64), 32, 2, 32, 16)
        self.level4_2 = Refine_block2_1((64, 128), 64, 2, 32, 16)

        self.level5_2 = Refine_block2_1((32, 64), 32, 2, 32, 16)
        self.pool41 = nn.MaxPool2d(kernel_size=2, stride=2) 
        self.stage41_pool41 = self.make_layers(32, [64])
        self.stage41_42 = self.make_layers_1(64, [64])

        self.level5_1 = Refine_block2_1((32, 64), 32, 2, 32, 16)

        self.level7 = nn.Conv2d(32, 1, kernel_size=1, padding=0)


        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.normal_(m.weight, 0, 1e-2)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, *input):
        j_w = self.j_w(input[0])
        j_w1 = self.j_w1(input[1])
        j_w2 = self.j_w2(input[2])
        j_w3 = self.j_w3(input[3])

        level1 = self.level1(j_w, j_w1)  
        pool1 = self.pool1(input[0])                
        stage1_pool1 = self.stage1_pool1(pool1)      
        stage2_22 = self.stage2_22(input[1])         
        ss1_2 = stage1_pool1 + stage2_22


        level2 = self.level2(j_w1, j_w2)  
        pool2 = self.pool2(input[1])                 
        stage2_pool2 = self.stage2_pool2(pool2)      
        stage3_33 = self.stage3_33(input[2])         

        ss2_3 = stage2_pool2 + stage3_33

        level3 = self.level3(j_w2, j_w3)
        pool3 = self.pool3(input[2])
        stage3_pool3 = self.stage3_pool3(pool3)
        stage4_44 = self.stage4_44(input[3])

        ss3_4 = stage3_pool3 + stage4_44

        level4 = self.level4(j_w3, input[4])
        pool4 = self.pool4(input[3])
        stage4_pool4 = self.stage4_pool4(pool4)
        stage5_55 = self.stage5_55(input[4])

        ss4_5 = stage4_pool4 + stage5_55

        ss1_2_level2 = ss1_2 + level2   
        ss2_3_level3 = ss2_3 + level3   
        ss3_4_level4 = ss3_4 + level4   

        level22 = []
        level22 += [self.level2_11(level1, ss1_2_level2)]
        level22 += [self.level2_22(ss1_2_level2, ss2_3_level3)]
        level22 += [self.level2_33(ss2_3_level3, ss3_4_level4)]
        level22 += [self.level2_44(ss3_4_level4, ss4_5)]

        level2_2 = self.level2_2(level22[0], level22[1])
        pool21 = self.pool21(level22[0])
        stage21_pool21 = self.stage21_pool21(pool21)
        stage21_22 = self.stage21_22(level22[1])
        stage21_22_stage21_pool21 = stage21_pool21 + stage21_22

        level2_3 = self.level2_3(level22[1], level22[2])
        pool22 = self.pool22(level22[1])
        stage22_pool22 = self.stage22_pool22(pool22)
        stage22_23 = self.stage22_23(level22[2])
        stage22_23_stage22_pool22 = stage22_pool22 + stage22_23

        level2_4 = self.level2_4(level22[2], level22[3])
        pool23 = self.pool23(level22[2])
        stage23_pool23 = self.stage23_pool23(pool23)
        stage23_24 = self.stage23_24(level22[3])
        stage23_24_stage23_pool23 = stage23_pool23 + stage23_24

        stage21_22_stage21_pool21_level2_3 = stage21_22_stage21_pool21 + level2_3
        stage22_23_stage22_pool22_level2_4 = stage22_23_stage22_pool22 + level2_4

        level33 = []
        level33 += [self.level3_11(level2_2, stage21_22_stage21_pool21_level2_3)]
        level33 += [self.level3_22(stage21_22_stage21_pool21_level2_3, stage22_23_stage22_pool22_level2_4)]
        level33 += [self.level3_33(stage22_23_stage22_pool22_level2_4, stage23_24_stage23_pool23)]

        level3_2 = self.level3_2(level33[0], level33[1])
        pool31 = self.pool21(level33[0])
        stage31_pool31 = self.stage31_pool31(pool31)
        stage31_32 = self.stage31_32(level33[1])
        stage31_32_stage31_pool31 = stage31_pool31 + stage31_32

        level3_3 = self.level3_3(level33[1], level33[2])
        pool32 = self.pool32(level33[1])
        stage32_pool32 = self.stage32_pool32(pool32)
        stage32_33 = self.stage32_33(level33[2])
        stage32_33_stage32_pool32 = stage32_pool32 + stage32_33

        stage31_32_stage31_pool31_level3_3 = stage31_32_stage31_pool31 + level3_3
        level44 = []
        level44 += [self.level4_1(level3_2, stage31_32_stage31_pool31_level3_3)]
        level44 += [self.level4_2(stage31_32_stage31_pool31_level3_3, stage32_33_stage32_pool32)]

        level5_2 = self.level5_2(level44[0], level44[1])
        pool41 = self.pool41(level44[0])
        stage41_pool41 = self.stage41_pool41(pool41)
        stage41_42 = self.stage41_42(level44[1])
        stage41_42_stage41_pool41 = stage41_pool41 + stage41_42

        level5 = self.level5_1(level5_2, stage41_42_stage41_pool41)


        return self.level7(level5)

    @staticmethod
    # 定义的实现卷积层的模块化函数
    def make_layers(in_channels, cfg, stride=1, rate=1):
        layers = []
        for v in cfg:
            if v == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                conv2d = nn.Conv2d(in_channels, v, kernel_size=1, padding=0, stride=stride)
                layers += [conv2d, nn.ReLU(inplace=True)]
                in_channels = v
        return nn.Sequential(*layers)

    @staticmethod
    # 定义的实现卷积层的模块化函数
    def make_layers_1(in_channels, cfg, stride=1, rate=1):
        layers = []
        for v in cfg:
            if v == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1, stride=stride, dilation=rate)
                layers += [conv2d, nn.ReLU(inplace=True)]
                in_channels = v
        return nn.Sequential(*layers)
            # 解码网络可修改的地方，弄懂参数，D，groups是残差网络模块中的参数对应intermediate,cardinality。
            # def __init__(self, in_channel, out_channel, factor, D, groups, require_grad=False):
#奇异层1
        self.level1 = Refine_block2_1((64, 128), 64, 2, 32, 16)#2上采样1不变

        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)#1下采样，卷积变为64通道，2卷积变为64通道
        self.stage1_pool1 = self.make_layers(64, [64])
        self.stage2_22 = self.make_layers_1(128, [64])

        self.level2 = Refine_block2_1((128, 256), 128, 2, 32, 16)#3上采样2不变

        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)  #2下采样，卷积变为128通道，3卷积变为128通道
        self.stage2_pool2 = self.make_layers(128, [128])
        self.stage3_33 = self.make_layers_1(256, [128])

        self.level3 = Refine_block2_1((256, 512), 256, 2, 32, 16)#4上采样3不变

        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)  # 3下采样，卷积变为256通道，4卷积变为256通道
        self.stage3_pool3 = self.make_layers(256, [256])
        self.stage4_44 = self.make_layers_1(512, [256])

        self.level4 = Refine_block2_1((512, 512), 512, 2, 32, 16)#5上采样4不变

        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)  # 4下采样，卷积变为512通道，5卷积变为512通道
        self.stage4_pool4 = self.make_layers(512, [512])
        self.stage5_55 = self.make_layers_1(512, [512])
        #融合
        self.level2_11 = Refine_block2_1((64, 64), 64, 2, 32, 16)
        self.level2_22 = Refine_block2_1((128, 128), 128, 2, 32, 16)
        self.level2_33 = Refine_block2_1((256, 256), 256, 2, 32, 16)
        self.level2_44 = Refine_block2_1((512, 512), 512, 2, 32, 16)
#奇异层2
        self.level2_2 = Refine_block2_1((64, 128), 64, 2, 32, 16)#2层2上采样1不变
        self.pool21 = nn.MaxPool2d(kernel_size=2, stride=2)  # 1下采样，卷积变为64通道，2卷积变为64通道
        self.stage21_pool21 = self.make_layers(64, [64])
        self.stage21_22 = self.make_layers_1(128, [64])

        self.level2_3 = Refine_block2_1((128, 256), 128, 2, 32, 16)#2层3上采样2不变
        self.pool22 = nn.MaxPool2d(kernel_size=2, stride=2)  # 2下采样，卷积变为128通道，3卷积变为128通道
        self.stage22_pool22 = self.make_layers(128, [128])
        self.stage22_23 = self.make_layers_1(256, [128])

        self.level2_4 = Refine_block2_1((256, 512), 256, 2, 32, 16) #2层4上采样3不变
        self.pool23 = nn.MaxPool2d(kernel_size=2, stride=2)  # 3下采样，卷积变为256通道，4卷积变为256通道
        self.stage23_pool23 = self.make_layers(256, [256])
        self.stage23_24 = self.make_layers_1(512, [256])
        #融合
        self.level3_11 = Refine_block2_1((64, 64), 64, 2, 32, 16)
        self.level3_22 = Refine_block2_1((128, 128), 128, 2, 32, 16)
        self.level3_33 = Refine_block2_1((256, 256), 256, 2, 32, 16)

        # self.level3_11 = Refine_block2_1((64, 128), 64, 2, 32, 16)
        # self.level3_22 = Refine_block2_1((128, 256), 128, 2, 32, 16)
        # self.level3_33 = Refine_block2_1((256, 512), 256, 2, 32, 16)
#奇异层3
        self.level3_2 = Refine_block2_1((64, 128), 64, 2, 32, 16)#3层2上采样1不变
        self.pool31 = nn.MaxPool2d(kernel_size=2, stride=2)  # 1下采样，卷积变为64通道，2卷积变为64通道
        self.stage31_pool31 = self.make_layers(64, [64])
        self.stage31_32 = self.make_layers_1(128, [64])

        self.level3_3 = Refine_block2_1((128, 256), 128, 2, 32, 16)#3层3上采样2不变
        self.pool32 = nn.MaxPool2d(kernel_size=2, stride=2)  # 2下采样，卷积变为128通道，3卷积变为128通道
        self.stage32_pool32 = self.make_layers(128, [128])
        self.stage32_33 = self.make_layers_1(256, [128])
#融合
        self.level4_1 = Refine_block2_1((64, 64), 64, 2, 32, 16)
        self.level4_2 = Refine_block2_1((128, 128), 128, 2, 32, 16)
#奇异层4
        self.level5_2 = Refine_block2_1((64, 128), 64, 2, 32, 16)#4层2上采样1不变
        self.pool41 = nn.MaxPool2d(kernel_size=2, stride=2)  # 1下采样，卷积变为64通道，2卷积变为64通道
        self.stage41_pool41 = self.make_layers(64, [64])
        self.stage41_42 = self.make_layers_1(128, [64])

        self.level5_1 = Refine_block2_1((64, 64), 64, 2, 32, 16)
        self.level7 = nn.Conv2d(64, 1, kernel_size=1, padding=0)


        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.normal_(m.weight, 0, 1e-2)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, *input):

        level1 = self.level1(*[input[0], input[1]])
        pool1 = self.pool1(input[0])
        stage1_pool1 = self.stage1_pool1(pool1)
        stage2_22 = self.stage2_22(input[1])

        ss1_2 = stage1_pool1 + stage2_22

        level2 = self.level2(*[input[1], input[2]])
        pool2 = self.pool2(input[1])
        stage2_pool2 = self.stage2_pool2(pool2)
        stage3_33 = self.stage3_33(input[2])

        ss2_3 = stage2_pool2 + stage3_33

        level3 = self.level3(*[input[2], input[3]])
        pool3 = self.pool3(input[2])
        stage3_pool3 = self.stage3_pool3(pool3)
        stage4_44 = self.stage4_44(input[3])

        ss3_4 = stage3_pool3 + stage4_44

        level4 = self.level4(*[input[3], input[4]])
        pool4 = self.pool4(input[3])
        stage4_pool4 = self.stage4_pool4(pool4)
        stage5_55 = self.stage5_55(input[4])

        ss4_5 = stage4_pool4 + stage5_55

        level22 = []
        level22 += [self.level2_11(level1, ss1_2)]
        level22 += [self.level2_22(level2, ss2_3)]
        level22 += [self.level2_33(level3, ss3_4)]
        level22 += [self.level2_44(level4, ss4_5)]

        level2_2 = self.level2_2(level22[0], level22[1])
        pool21 = self.pool21(level22[0])
        stage21_pool21 = self.stage21_pool21(pool21)
        stage21_22 = self.stage21_22(level22[1])
        stage21_22_stage21_pool21 = stage21_pool21 + stage21_22

        level2_3 = self.level2_3(level22[1], level22[2])
        pool22 = self.pool22(level22[1])
        stage22_pool22 = self.stage22_pool22(pool22)
        stage22_23 = self.stage22_23(level22[2])
        stage22_23_stage22_pool22 = stage22_pool22 + stage22_23

        level2_4 = self.level2_4(level22[2], level22[3])
        pool23 = self.pool23(level22[2])
        stage23_pool23 = self.stage23_pool23(pool23)
        stage23_24 = self.stage23_24(level22[3])
        stage23_24_stage23_pool23 = stage23_pool23 + stage23_24

        level33 = []
        level33 += [self.level3_11(level2_2, stage21_22_stage21_pool21)]
        level33 += [self.level3_22(level2_3, stage22_23_stage22_pool22)]
        level33 += [self.level3_33(level2_4, stage23_24_stage23_pool23)]

#奇异层3
        level3_2 = self.level3_2(level33[0], level33[1])
        pool31 = self.pool21(level33[0])
        stage31_pool31 = self.stage31_pool31(pool31)
        stage31_32 = self.stage31_32(level33[1])
        stage31_32_stage31_pool31 = stage31_pool31 + stage31_32

        level3_3 = self.level3_3(level33[1], level33[2])
        pool32 = self.pool32(level33[1])
        stage32_pool32 = self.stage32_pool32(pool32)
        stage32_33 = self.stage32_33(level33[2])
        stage32_33_stage32_pool32 = stage32_pool32 + stage32_33

        level44 = []
        level44 += [self.level4_1(level3_2, stage31_32_stage31_pool31)]
        level44 += [self.level4_2(level3_3, stage32_33_stage32_pool32)]
#奇异层4
        level5_2 = self.level5_2(level44[0], level44[1])
        pool41 = self.pool41(level44[0])
        stage41_pool41 = self.stage41_pool41(pool41)
        stage41_42 = self.stage41_42(level44[1])
        stage41_42_stage41_pool41 = stage41_pool41 + stage41_42

        level5 = self.level5_1(level5_2, stage41_42_stage41_pool41)


        return self.level7(level5)

    @staticmethod
    # 定义的实现卷积层的模块化函数
    def make_layers(in_channels, cfg, stride=1, rate=1):
        layers = []
        for v in cfg:
            if v == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1, stride=stride, dilation=rate)
                layers += [conv2d, nn.ReLU(inplace=True)]
                in_channels = v
        return nn.Sequential(*layers)

    @staticmethod
    # 定义的实现卷积层的模块化函数
    def make_layers_1(in_channels, cfg, stride=1, rate=1):
        layers = []
        for v in cfg:
            if v == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1, stride=stride, dilation=rate)
                layers += [conv2d, nn.ReLU(inplace=True)]
                in_channels = v
        return nn.Sequential(*layers)
#VGG的5个输出，除了1,5之外2,3,4每个输出三个，分别为不变，下采样，上采样3*3降维 1*1不变

    def __init__(self):
        super(decode8, self).__init__()
            # 解码网络可修改的地方，弄懂参数，D，groups是残差网络模块中的参数对应intermediate,cardinality。
            # def __init__(self, in_channel, out_channel, factor, D, groups, require_grad=False):
#奇异层1
        self.level1 = Refine_block2_1((64, 128), 64, 2, 32, 16)#2上采样1不变

        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)#1下采样，卷积变为64通道，2卷积变为64通道
        self.stage1_pool1 = self.make_layers(64, [64])
        self.stage2_22 = self.make_layers_1(128, [64])

        self.level2 = Refine_block2_1((128, 256), 128, 2, 32, 16)#3上采样2不变

        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)  #2下采样，卷积变为128通道，3卷积变为128通道
        self.stage2_pool2 = self.make_layers(128, [128])
        self.stage3_33 = self.make_layers_1(256, [128])

        self.level3 = Refine_block2_1((256, 512), 256, 2, 32, 16)#4上采样3不变

        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)  # 3下采样，卷积变为256通道，4卷积变为256通道
        self.stage3_pool3 = self.make_layers(256, [256])
        self.stage4_44 = self.make_layers_1(512, [256])

        self.level4 = Refine_block2_1((512, 512), 512, 2, 32, 16)#5上采样4不变

        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)  # 4下采样，卷积变为512通道，5卷积变为512通道
        self.stage4_pool4 = self.make_layers(512, [512])
        self.stage5_55 = self.make_layers_1(512, [512])
        #融合
        self.level2_11 = Refine_block2_1((64, 64), 64, 2, 32, 16)
        self.level2_22 = Refine_block2_1((128, 128), 128, 2, 32, 16)
        self.level2_33 = Refine_block2_1((256, 256), 256, 2, 32, 16)
        self.level2_44 = Refine_block2_1((512, 512), 512, 2, 32, 16)
#奇异层2
        self.level2_2 = Refine_block2_1((64, 128), 64, 2, 32, 16)#2层2上采样1不变
        self.pool21 = nn.MaxPool2d(kernel_size=2, stride=2)  # 1下采样，卷积变为64通道，2卷积变为64通道
        self.stage21_pool21 = self.make_layers(64, [64])
        self.stage21_22 = self.make_layers_1(128, [64])

        self.level2_3 = Refine_block2_1((128, 256), 128, 2, 32, 16)#2层3上采样2不变
        self.pool22 = nn.MaxPool2d(kernel_size=2, stride=2)  # 2下采样，卷积变为128通道，3卷积变为128通道
        self.stage22_pool22 = self.make_layers(128, [128])
        self.stage22_23 = self.make_layers_1(256, [128])

        self.level2_4 = Refine_block2_1((256, 512), 256, 2, 32, 16) #2层4上采样3不变
        self.pool23 = nn.MaxPool2d(kernel_size=2, stride=2)  # 3下采样，卷积变为256通道，4卷积变为256通道
        self.stage23_pool23 = self.make_layers(256, [256])
        self.stage23_24 = self.make_layers_1(512, [256])
        #融合
        self.level3_11 = Refine_block2_1((64, 64), 64, 2, 32, 16)
        self.level3_22 = Refine_block2_1((128, 128), 128, 2, 32, 16)
        self.level3_33 = Refine_block2_1((256, 256), 256, 2, 32, 16)

        # self.level3_11 = Refine_block2_1((64, 128), 64, 2, 32, 16)
        # self.level3_22 = Refine_block2_1((128, 256), 128, 2, 32, 16)
        # self.level3_33 = Refine_block2_1((256, 512), 256, 2, 32, 16)
#奇异层3
        self.level3_2 = Refine_block2_1((64, 128), 64, 2, 32, 16)#3层2上采样1不变
        self.pool31 = nn.MaxPool2d(kernel_size=2, stride=2)  # 1下采样，卷积变为64通道，2卷积变为64通道
        self.stage31_pool31 = self.make_layers(64, [64])
        self.stage31_32 = self.make_layers_1(128, [64])

        self.level3_3 = Refine_block2_1((128, 256), 128, 2, 32, 16)#3层3上采样2不变
        self.pool32 = nn.MaxPool2d(kernel_size=2, stride=2)  # 2下采样，卷积变为128通道，3卷积变为128通道
        self.stage32_pool32 = self.make_layers(128, [128])
        self.stage32_33 = self.make_layers_1(256, [128])
#融合
        self.level4_1 = Refine_block2_1((64, 64), 64, 2, 32, 16)
        self.level4_2 = Refine_block2_1((128, 128), 128, 2, 32, 16)
#奇异层4
        self.level5_2 = Refine_block2_1((64, 128), 64, 2, 32, 16)#4层2上采样1不变
        self.pool41 = nn.MaxPool2d(kernel_size=2, stride=2)  # 1下采样，卷积变为64通道，2卷积变为64通道
        self.stage41_pool41 = self.make_layers(64, [64])
        self.stage41_42 = self.make_layers_1(128, [64])

        self.level5_1 = Refine_block2_1((64, 64), 64, 2, 32, 16)
        self.level7 = nn.Conv2d(64, 1, kernel_size=1, padding=0)


        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.normal_(m.weight, 0, 1e-2)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, *input):

        level1 = self.level1(*[input[0], input[1]])
        pool1 = self.pool1(input[0])
        stage1_pool1 = self.stage1_pool1(pool1)
        stage2_22 = self.stage2_22(input[1])

        ss1_2 = stage1_pool1 + stage2_22

        level2 = self.level2(*[input[1], input[2]])
        pool2 = self.pool2(input[1])
        stage2_pool2 = self.stage2_pool2(pool2)
        stage3_33 = self.stage3_33(input[2])

        ss2_3 = stage2_pool2 + stage3_33

        level3 = self.level3(*[input[2], input[3]])
        pool3 = self.pool3(input[2])
        stage3_pool3 = self.stage3_pool3(pool3)
        stage4_44 = self.stage4_44(input[3])

        ss3_4 = stage3_pool3 + stage4_44

        level4 = self.level4(*[input[3], input[4]])
        pool4 = self.pool4(input[3])
        stage4_pool4 = self.stage4_pool4(pool4)
        stage5_55 = self.stage5_55(input[4])

        ss4_5 = stage4_pool4 + stage5_55

        level22 = []
        level22 += [self.level2_11(level1, ss1_2)]
        level22 += [self.level2_22(level2, ss2_3)]
        level22 += [self.level2_33(level3, ss3_4)]
        level22 += [self.level2_44(level4, ss4_5)]

        level2_2 = self.level2_2(level22[0], level22[1])
        pool21 = self.pool21(level22[0])
        stage21_pool21 = self.stage21_pool21(pool21)
        stage21_22 = self.stage21_22(level22[1])
        stage21_22_stage21_pool21 = stage21_pool21 + stage21_22

        level2_3 = self.level2_3(level22[1], level22[2])
        pool22 = self.pool22(level22[1])
        stage22_pool22 = self.stage22_pool22(pool22)
        stage22_23 = self.stage22_23(level22[2])
        stage22_23_stage22_pool22 = stage22_pool22 + stage22_23

        level2_4 = self.level2_4(level22[2], level22[3])
        pool23 = self.pool23(level22[2])
        stage23_pool23 = self.stage23_pool23(pool23)
        stage23_24 = self.stage23_24(level22[3])
        stage23_24_stage23_pool23 = stage23_pool23 + stage23_24

        level33 = []
        level33 += [self.level3_11(level2_2, stage21_22_stage21_pool21)]
        level33 += [self.level3_22(level2_3, stage22_23_stage22_pool22)]
        level33 += [self.level3_33(level2_4, stage23_24_stage23_pool23)]

#奇异层3
        level3_2 = self.level3_2(level33[0], level33[1])
        pool31 = self.pool21(level33[0])
        stage31_pool31 = self.stage31_pool31(pool31)
        stage31_32 = self.stage31_32(level33[1])
        stage31_32_stage31_pool31 = stage31_pool31 + stage31_32

        level3_3 = self.level3_3(level33[1], level33[2])
        pool32 = self.pool32(level33[1])
        stage32_pool32 = self.stage32_pool32(pool32)
        stage32_33 = self.stage32_33(level33[2])
        stage32_33_stage32_pool32 = stage32_pool32 + stage32_33

        level44 = []
        level44 += [self.level4_1(level3_2, stage31_32_stage31_pool31)]
        level44 += [self.level4_2(level3_3, stage32_33_stage32_pool32)]
#奇异层4
        level5_2 = self.level5_2(level44[0], level44[1])
        pool41 = self.pool41(level44[0])
        stage41_pool41 = self.stage41_pool41(pool41)
        stage41_42 = self.stage41_42(level44[1])
        stage41_42_stage41_pool41 = stage41_pool41 + stage41_42

        level5 = self.level5_1(level5_2, stage41_42_stage41_pool41)


        return self.level7(level5)

    @staticmethod
    # 定义的实现卷积层的模块化函数
    def make_layers(in_channels, cfg, stride=1, rate=1):
        layers = []
        for v in cfg:
            if v == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                conv2d = nn.Conv2d(in_channels, v, kernel_size=1, padding=0, stride=stride, dilation=rate)
                layers += [conv2d, nn.ReLU(inplace=True)]
                in_channels = v
        return nn.Sequential(*layers)

    @staticmethod
    # 定义的实现卷积层的模块化函数
    def make_layers_1(in_channels, cfg, stride=1, rate=1):
        layers = []
        for v in cfg:
            if v == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1, stride=stride, dilation=rate)
                layers += [conv2d, nn.ReLU(inplace=True)]
                in_channels = v
        return nn.Sequential(*layers)
class BFENNet(nn.Module):
    def __init__(self, cfgs):
        super(BFENNet, self).__init__()
        self.encode = VGG13_qz1(cfgs)
        self.decode1 = decode1()
    def forward(self, x):
        end_points = self.encode(x)  
        x = self.decode1(*end_points).sigmoid() 
        return x

class Cross_Entropy(nn.Module):
    def __init__(self):
        super(Cross_Entropy, self).__init__()
        self.weight1 = nn.Parameter(torch.Tensor([1.]))
        self.weight2 = nn.Parameter(torch.Tensor([1.]))
    def forward(self, pred, labels):
        pred_flat = pred.view(-1)
        labels_flat = labels.view(-1)
        pred_pos = pred_flat[labels_flat > 0]
        pred_neg = pred_flat[labels_flat == 0]
        total_loss = 1.00 * cross_entropy_per_image(pred, labels)
 
        return total_loss, (1-pred_pos).abs(), pred_neg

def dice(logits, labels):
    logits = logits.view(-1)
    labels = labels.view(-1)
    eps = 1e-6
    dice = ((logits * labels).sum() * 2 + eps) / (logits.sum() + labels.sum() + eps)
    dice_loss = dice.pow(-1)
    return dice_loss

def dice_loss_per_image(logits, labels):
    total_loss = 0
    for i, (_logit, _label) in enumerate(zip(logits, labels)):
        total_loss += dice(_logit, _label)
    return total_loss / len(logits)

def cross_entropy_per_image(logits, labels):
    total_loss = 0
    for i, (_logit, _label) in enumerate(zip(logits, labels)):
        total_loss += cross_entropy_with_weight(_logit, _label)
    return total_loss / len(logits)

def cross_entropy_orignal(logits, labels):
    logits = logits.view(-1)
    labels = labels.view(-1)
    eps = 1e-6
    pred_pos = logits[labels >= 0.5].clamp(eps, 1.0 - eps)
    pred_neg = logits[labels == 0].clamp(eps, 1.0 - eps)

    weight_pos, weight_neg = get_weight(labels, labels, 0.5, 1.5)

    cross_entropy = (-pred_pos.log() * weight_pos).sum() + \
                            (-(1.0 - pred_neg).log() * weight_neg).sum()
    return cross_entropy

def cross_entropy_with_weight(logits, labels):
    logits = logits.view(-1)
    labels = labels.view(-1)
    eps = 1e-6
    pred_pos = logits[labels > 0].clamp(eps,1.0-eps)
    pred_neg = logits[labels == 0].clamp(eps,1.0-eps)
    w_anotation = labels[labels > 0]
    # weight_pos, weight_neg = get_weight(labels, labels, 0.5, 1.5)
    cross_entropy = (-pred_pos.log() * w_anotation).mean() + \
                    (-(1.0 - pred_neg).log()).mean()
    # cross_entropy = (-pred_pos.log() * weight_pos).sum() + \
    #                     (-(1.0 - pred_neg).log() * weight_neg).sum()
    return cross_entropy

def get_weight(src, mask, threshold, weight):
    count_pos = src[mask >= threshold].size()[0]
    count_neg = src[mask == 0.0].size()[0]
    total = count_neg + count_pos
    weight_pos = count_neg / total
    weight_neg = (count_pos / total) * weight
    return weight_pos, weight_neg

def learning_rate_decay(optimizer, epoch, decay_rate=0.1, decay_steps=10):
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr'] * (decay_rate ** (epoch // decay_steps))