# coding=utf-8
# Copyright 2021 The Ravens Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# TF vs Pytorch
# BHWC  vs   BCHW


"""Resnet module."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class Bottleneck(nn.Module):
    expansion = 4
    def __init__(self, in_channels, out_channels, i_downsample=None, stride=1, do_upsample = False, activation=True, include_batchnorm=True):
        super(Bottleneck, self).__init__()

        self.activation=activation
        self.include_batchnorm=include_batchnorm
        self.do_upsample = do_upsample
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
        self.batch_norm1 = nn.BatchNorm2d(out_channels)
        
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.batch_norm2 = nn.BatchNorm2d(out_channels)
        
        self.conv3 = nn.Conv2d(out_channels, out_channels*self.expansion, kernel_size=1, stride=1, padding=0)
        self.batch_norm3 = nn.BatchNorm2d(out_channels*self.expansion)
        
        self.i_downsample = i_downsample
        self.stride = stride
        self.relu = nn.ReLU()
        self.uscale = nn.UpsamplingNearest2d(scale_factor=2)
        
    def forward(self, x):
        identity = x.clone()
        if self.include_batchnorm:
            x = self.relu(self.batch_norm1(self.conv1(x)))
            x = self.relu(self.batch_norm2(self.conv2(x)))
            x = self.relu(self.batch_norm3(self.conv3(x)))
        else:
            x = self.relu(self.conv1(x))
            x = self.relu(self.conv2(x))
            x = self.relu(self.conv3(x))
        
        #downsample if needed
        if self.i_downsample is not None:
            identity = self.i_downsample(identity)
        #add identity
        x+=identity
        if self.activation:
            x=self.relu(x)

        if self.do_upsample:
            x = self.uscale(x)
        
        return x

class Block(nn.Module):
    expansion = 1
    def __init__(self, in_channels, out_channels, i_downsample=None, stride=1):
        super(Block, self).__init__()
       

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=stride, bias=False)
        self.batch_norm1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, stride=stride, bias=False)
        self.batch_norm2 = nn.BatchNorm2d(out_channels)

        self.i_downsample = i_downsample
        self.stride = stride
        self.relu = nn.ReLU()

    def forward(self, x):
      identity = x.clone()

      x = self.relu(self.batch_norm2(self.conv1(x)))
      x = self.batch_norm2(self.conv2(x))

      if self.i_downsample is not None:
          identity = self.i_downsample(identity)
      print(x.shape)
      print(identity.shape)
      x += identity
      x = self.relu(x)
      return x


        
        
class ResNet(nn.Module):
    def __init__(self, ResBlock, layer_list, num_classes, num_channels=3):
        super(ResNet, self).__init__()
        self.in_channels = 64
        
        self.conv1 = nn.Conv2d(num_channels, 64, kernel_size=7, stride=2, padding=4, bias=False)
        self.batch_norm1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.max_pool = nn.MaxPool2d(kernel_size = 3, stride=2, padding=1)
        
        self.layer1 = self._make_layer(ResBlock, layer_list[0], planes=64)
        self.layer2 = self._make_layer(ResBlock, layer_list[1], planes=128, stride=2)
        self.layer3 = self._make_layer(ResBlock, layer_list[2], planes=256, stride=2)
        self.layer4 = self._make_layer(ResBlock, layer_list[3], planes=512, stride=2)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(512*ResBlock.expansion, num_classes)
        
    def forward(self, x):
        print('x:{}'.format(x.shape))
        x = self.conv1(x)
        x = self.batch_norm1(x)
        x = self.relu(x)
        x = self.max_pool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        #x = self.avgpool(x)
        #x = x.reshape(x.shape[0], -1)
        #x = self.fc(x)
        
        return x
        
    def _make_layer(self, ResBlock, blocks, planes, stride=1):
        ii_downsample = None
        layers = []
        
        if stride != 1 or self.in_channels != planes*ResBlock.expansion:
            ii_downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, planes*ResBlock.expansion, kernel_size=1, stride=stride),
                nn.BatchNorm2d(planes*ResBlock.expansion)
            )
            
        layers.append(ResBlock(self.in_channels, planes, i_downsample=ii_downsample, stride=stride))
        self.in_channels = planes*ResBlock.expansion
        
        for i in range(blocks-1):
            layers.append(ResBlock(self.in_channels, planes))
            
        return nn.Sequential(*layers)

class ResNet2(nn.Module):
    def __init__(self, ResBlock, layer_list, num_classes, num_channels=3):
        super(ResNet2, self).__init__()
        self.in_channels = 64
        
        self.conv1 = nn.Conv2d(num_channels, 64, kernel_size=3, stride=1, padding='same', bias=False)
        self.batch_norm1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.max_pool = nn.MaxPool2d(kernel_size = 3, stride=2, padding=1)
        
        self.layer1 = self._make_layer(ResBlock, layer_list[0], planes=64)
        self.layer2 = self._make_layer(ResBlock, layer_list[1], planes=128, stride=2)
        self.layer3 = self._make_layer(ResBlock, layer_list[2], planes=256, stride=2)
        self.layer4 = self._make_layer(ResBlock, layer_list[3], planes=512, stride=2)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(512*ResBlock.expansion, num_classes)
        
    def forward(self, x):
        x = self.relu(self.batch_norm1(self.conv1(x)))
        x = self.max_pool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        #x = self.avgpool(x)
        #x = x.reshape(x.shape[0], -1)
        #x = self.fc(x)
        
        return x
        
    def _make_layer(self, ResBlock, blocks, planes, stride=1):
        ii_downsample = None
        layers = []
        
        if stride != 1 or self.in_channels != planes*ResBlock.expansion:
            ii_downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, planes*ResBlock.expansion, kernel_size=1, stride=stride),
                nn.BatchNorm2d(planes*ResBlock.expansion)
            )
            
        layers.append(ResBlock(self.in_channels, planes, i_downsample=ii_downsample, stride=stride))
        self.in_channels = planes*ResBlock.expansion
        
        for i in range(blocks-1):
            layers.append(ResBlock(self.in_channels, planes, do_upsample = True))
            
        return nn.Sequential(*layers)

def xavier_uniform_(tensor, gain = 1.):
    r"""Fills the input `Tensor` with values according to the method
    described in `Understanding the difficulty of training deep feedforward
    neural networks` - Glorot, X. & Bengio, Y. (2010), using a uniform
    distribution. The resulting tensor will have values sampled from
    :math:`\mathcal{U}(-a, a)` where

    .. math::
        a = \text{gain} \times \sqrt{\frac{6}{\text{fan\_in} + \text{fan\_out}}}

    Also known as Glorot initialization.

    Args:
        tensor: an n-dimensional `torch.Tensor`
        gain: an optional scaling factor

    Examples:
        >>> w = torch.empty(3, 5)
        >>> nn.init.xavier_uniform_(w, gain=nn.init.calculate_gain('relu'))
    """
    fan_in, fan_out = torch.nn.init._calculate_fan_in_and_fan_out(tensor)
    #print('f_inout:{}'.format((fan_in, fan_out)))
    std = math.sqrt(6.0 / float(fan_in + fan_out))
    #print('std:{}'.format(std))
    #a = math.sqrt(3.0) * std  # Calculate uniform bounds from standard deviation
    a = std
    #print('a:{}'.format(a))

    return torch.nn.init._no_grad_uniform_(tensor, -a, a)

def init_weights(m):
    if isinstance(m, nn.Conv2d):
        #torch.nn.init.constant_(m.weight, 1.)
        xavier_uniform_(m.weight)
        if m.bias is not None:
            m.bias.data.fill_(0.0)

def print_info(model):
    for layer in model.children():
        if isinstance(layer, nn.Conv2d):
            print('w:{} | {}'.format(layer.state_dict()['weight'], layer.state_dict()['weight'].shape))
            #print('b:{}'.format(layer.state_dict()['bias']))

      
def ResNet36_4s(num_classes, channels=6):
    return ResNet(Bottleneck, [6,0,0,0], num_classes, channels)

#def ResNet43_8s(num_classes, channels=6):
#    return ResNet(Bottleneck, [6,6,6,6], num_classes, channels)

def ResNet50(num_classes, channels=3):
    return ResNet(Bottleneck, [3,4,6,3], num_classes, channels)
    
def ResNet101(num_classes, channels=3):
    return ResNet(Bottleneck, [3,4,23,3], num_classes, channels)

def ResNet152(num_classes, channels=3):
    return ResNet(Bottleneck, [3,8,36,3], num_classes, channels)




class IdentityBlock0(nn.Module):
    def __init__(self, filters, kernel_size, stride=1, include_batchnorm=False):
        super(IdentityBlock, self).__init__()
       
        self.include_batchnorm = include_batchnorm
        filters0, filters1, filters2, filters3 = filters

        self.conv1 = nn.Conv2d(filters0, filters1, kernel_size=1, padding=0, stride=1, bias=False)
        self.batch_norm1 = nn.BatchNorm2d(filters1)
        self.conv2 = nn.Conv2d(filters1, filters2, kernel_size=kernel_size, padding=1, stride=stride, bias=False)
        self.batch_norm2 = nn.BatchNorm2d(filters2)
        self.conv3 = nn.Conv2d(filters2, filters3, kernel_size=1, padding=0, stride=1, bias=False)
        self.batch_norm3 = nn.BatchNorm2d(filters3)

        self.relu = nn.ReLU()

        print('AAAAAAAAAAAAAAAAAAAAAAA')


    def forward(self, x):
      identity = x.clone()

      if self.include_batchnorm:
        x = self.relu(self.batch_norm1(self.conv1(x)))
        x = self.relu(self.batch_norm2(self.conv2(x)))
        x = self.relu(self.batch_norm3(self.conv3(x)))
      else:
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.conv3(x)

      #print('Ixi:{}'.format((x.shape, identity.shape)))

      x = x + identity
      x = self.relu(x)
      return x

class ConvBlock0(nn.Module):
    def __init__(self, filters, kernel_size=3, stride=1, include_batchnorm=False):
        super(ConvBlock, self).__init__()

        self.include_batchnorm = include_batchnorm
        filters0, filters1, filters2, filters3 = filters

        self.conv1 = nn.Conv2d(filters0, filters1, kernel_size=1, padding=0, stride=1, bias=False)
        self.batch_norm1 = nn.BatchNorm2d(filters1)
        self.conv2 = nn.Conv2d(filters1, filters2, kernel_size=kernel_size, padding=1, stride=stride, bias=False)
        self.batch_norm2 = nn.BatchNorm2d(filters2)
        self.conv3 = nn.Conv2d(filters2, filters3, kernel_size=1, padding=0, stride=1, bias=False)
        self.batch_norm3 = nn.BatchNorm2d(filters3)

        self.shortcut = nn.Conv2d(filters0, filters3, kernel_size=1, padding=0, stride=stride, bias=False)
        self.batch_shortcut = nn.BatchNorm2d(filters3)
        self.relu = nn.ReLU()

        print('BBBBBBBBBBBBBBBBBBBBB')

    def forward(self, x):
      identity = x.clone()
      identity = self.shortcut(identity)
      return self.relu(identity)

      if self.include_batchnorm:
        x = self.relu(self.batch_norm1(self.conv1(x)))
        x = self.relu(self.batch_norm2(self.conv2(x)))
        x = self.relu(self.batch_norm3(self.conv3(x)))
        identity = self.batch_shortcut(self.shortcut(identity))
      else:
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        #x = self.relu(self.conv3(x))
        x = self.conv3(x)
        identity = self.shortcut(identity)

      #print('Cxi:{}'.format((x.shape, identity.shape)))
      x = x + identity
      x = self.relu(x)
      return x



class IdentityBlock(nn.Module):
    def __init__(self, filters, kernel_size, include_batchnorm=False):
        super(IdentityBlock, self).__init__()

        self.include_batchnorm=include_batchnorm
       
        filters0, filters1, filters2, filters3 = filters

        self.conv1 = nn.Conv2d(filters0, filters1, kernel_size=1, dilation=1, bias=True)
        torch.nn.init.xavier_uniform_(self.conv1.weight)
        self.batch_norm1 = nn.BatchNorm2d(filters1)
        self.conv2 = nn.Conv2d(filters1, filters2, kernel_size=kernel_size, dilation=1, padding=1, bias=True)#'same'
        torch.nn.init.xavier_uniform_(self.conv2.weight)
        self.batch_norm2 = nn.BatchNorm2d(filters2)
        self.conv3 = nn.Conv2d(filters2, filters3, kernel_size=1, dilation=1, bias=True)
        torch.nn.init.xavier_uniform_(self.conv3.weight)
        self.batch_norm3 = nn.BatchNorm2d(filters3)

        self.relu = nn.ReLU()

    def forward(self, x):
      identity = x.clone()

      if self.include_batchnorm:
        x = self.relu(self.batch_norm1(self.conv1(x)))
        x = self.relu(self.batch_norm2(self.conv2(x)))
        x = self.relu(self.batch_norm3(self.conv3(x)))
      else:
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.conv3(x)

      #print('Ixi:{}'.format((x.shape, identity.shape)))

      #print('AAAAAAAAAAAAAAAAAAAAAAAAA')
      x = x + identity
      x = self.relu(x)
      return x

class ConvBlock(nn.Module):
    def __init__(self, filters, kernel_size=3, stride=1, include_batchnorm=False):
        super(ConvBlock, self).__init__()

        self.include_batchnorm=include_batchnorm

        filters0, filters1, filters2, filters3 = filters

        self.conv1 = nn.Conv2d(filters0, filters1, kernel_size=1, dilation=1, stride=stride, bias=True)
        torch.nn.init.xavier_uniform_(self.conv1.weight)
        self.batch_norm1 = nn.BatchNorm2d(filters1)
        self.conv2 = nn.Conv2d(filters1, filters2, kernel_size=kernel_size, padding=1, bias=True)#'same'
        torch.nn.init.xavier_uniform_(self.conv2.weight)
        self.batch_norm2 = nn.BatchNorm2d(filters2)
        self.conv3 = nn.Conv2d(filters2, filters3, kernel_size=1, dilation=1, bias=True)
        torch.nn.init.xavier_uniform_(self.conv3.weight)
        self.batch_norm3 = nn.BatchNorm2d(filters3)

        self.shortcut = nn.Conv2d(filters0, filters3, kernel_size=1, dilation=1, stride=stride, bias=True)
        torch.nn.init.xavier_uniform_(self.shortcut.weight)
        self.batch_shortcut = nn.BatchNorm2d(filters3)
        self.relu = nn.ReLU()

    def forward(self, x):
        identity = x.clone()

        if self.include_batchnorm:
            x = self.relu(self.batch_norm1(self.conv1(x)))
            x = self.relu(self.batch_norm2(self.conv2(x)))
            x = self.relu(self.batch_norm3(self.conv3(x)))
            identity = self.batch_shortcut(self.shortcut(identity))
        else:
            x = self.relu(self.conv1(x))
            x = self.relu(self.conv2(x))
            x = self.conv3(x)
            identity = self.shortcut(identity)

        #print('BBBBBBBBBBBBBBBBBBBBBBBBBB')
        x = x + identity
        x = self.relu(x)
        return x


class ResNet43_8s(nn.Module):
    def __init__(self, input_shape, output_dim, include_batchnorm=False):
        super(ResNet43_8s, self).__init__()
        print('ResNet43_8s:{} | {}'.format(input_shape, output_dim))
        self.in_channels = 64
        self.include_batchnorm = include_batchnorm
        
        self.conv1 = nn.Conv2d(input_shape[2], 64, kernel_size=3, stride=1, padding=1, bias=True)#padding='same' different in pytorch
        torch.nn.init.xavier_uniform_(self.conv1.weight)
        self.batch_norm1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()

        self.c1 = ConvBlock([64, 64, 64, 64], kernel_size=3, stride=1, include_batchnorm=self.include_batchnorm)
        self.i1 = IdentityBlock([64, 64, 64, 64], kernel_size=3, include_batchnorm=self.include_batchnorm)

        self.c2 = ConvBlock([64, 128, 128, 128], kernel_size=3, stride=2, include_batchnorm=self.include_batchnorm)
        self.i2 = IdentityBlock([128, 128, 128, 128], kernel_size=3, include_batchnorm=self.include_batchnorm)

        self.c3 = ConvBlock([128, 256, 256, 256], kernel_size=3, stride=2, include_batchnorm=self.include_batchnorm)
        self.i3 = IdentityBlock([256, 256, 256, 256], kernel_size=3, include_batchnorm=self.include_batchnorm)

        self.c4 = ConvBlock([256, 512, 512, 512], kernel_size=3, stride=2, include_batchnorm=self.include_batchnorm)
        self.i4 = IdentityBlock([512, 512, 512, 512], kernel_size=3, include_batchnorm=self.include_batchnorm)

        self.c5 = ConvBlock([512, 256, 256, 256], kernel_size=3, stride=1, include_batchnorm=self.include_batchnorm)
        self.i5 = IdentityBlock([256, 256, 256, 256], kernel_size=3, include_batchnorm=self.include_batchnorm)
        # upsampling
        self.u1 = torch.nn.UpsamplingNearest2d(scale_factor=2)

        self.c6 = ConvBlock([256, 128, 128, 128], kernel_size=3, stride=1, include_batchnorm=self.include_batchnorm)
        self.i6 = IdentityBlock([128, 128, 128, 128], kernel_size=3, include_batchnorm=self.include_batchnorm)

        # upsampling
        self.u2 = torch.nn.UpsamplingNearest2d(scale_factor=2)

        self.c7 = ConvBlock([128, 64, 64, 64], kernel_size=3, stride=1, include_batchnorm=self.include_batchnorm)
        self.i7 = IdentityBlock([64, 64, 64, 64], kernel_size=3, include_batchnorm=self.include_batchnorm)

        # upsampling
        self.u3 = torch.nn.UpsamplingNearest2d(scale_factor=2)

        self.c8 = ConvBlock([64, 16, 16, output_dim], kernel_size=3, stride=1, include_batchnorm=self.include_batchnorm)
        self.i8 = IdentityBlock([output_dim, 16, 16, output_dim], kernel_size=3, include_batchnorm=self.include_batchnorm)

        self.ccut = ConvBlock([64, 64, 64, output_dim], kernel_size=3, stride=1, include_batchnorm=self.include_batchnorm)
        self.icut = IdentityBlock([output_dim, 64, 64, output_dim], kernel_size=3, include_batchnorm=self.include_batchnorm)

        #init_weights(self)

    def forward(self, x):

        #print('XXXXXXXXXXXXXXXXXXXXXXXXXX')
        x = self.conv1(x)
        if self.include_batchnorm:
            x = self.batch_norm1(x)
        x = self.relu(x)
        #x = self.ccut(x)
        #x = self.icut(x)
        #return x

        x = self.c1(x)
        x = self.i1(x)
        x = self.c2(x)
        x = self.i2(x)
        x = self.c3(x)
        x = self.i3(x)
        x = self.c4(x)
        x = self.i4(x)
        x = self.c5(x)
        x = self.i5(x)
        x = self.u1(x)
        x = self.c6(x)
        x = self.i6(x)
        x = self.u2(x)
        x = self.c7(x)
        x = self.i7(x)
        x = self.u3(x)
        x = self.c8(x)
        x = self.i8(x)
        
        return x



# https://stackoverflow.com/questions/49433936/how-do-i-initialize-weights-in-pytorch
def main():
    #in_shape = [64,64,6]
    #d_in, d_out = ResNet36_4s(in_shape, 1, include_batchnorm = True, cutoff_early = True)
    #print('d_in:{} d_out:{}'.format(d_in, d_out))
    #d_in, d_out = ResNet43_8s(in_shape, 1)
    #print('d_in:{} d_out:{}'.format(d_in, d_out))

    torch.manual_seed(0)
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    torch.autograd.set_detect_anomaly(True)

    #net = ResNet43_8s((320,284,6), 1).to(device)
    net = ResNet43_8sTest((320,284,6), 1, include_batchnorm=False).to(device)
    net.apply(init_weights)
    print_info(net)
    print('net:{}'.format(net))
    t1 = torch.randn([1,6,320,284]).to(device) # as channels first in pytorch
    t1.fill_(1.)
    res = net(t1)
    res = res.permute(0,2,3,1)
    print('res:{}'.format(res))
    print('res:{}'.format(res.shape))

if __name__ == "__main__":
    main()
