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

"""Conv MLP module."""

import sys
sys.path.append('.')

import numpy as np
import torch
import torch.nn as nn
from ravens.models.gt_state import MlpModel
from ravens.models.resnet import ResNet43_8s
#import tensorflow_hub as hub

#layers = tf.keras.layers


def compute_spatial_soft_argmax(x, batch_size, H=149, W=69, C=64):  # pylint: disable=invalid-name
  """Parameter-less, extract coordinates for each channel.

  Args:
    x: shape: (batch_size, ~H, ~W, C)
    batch_size: int
    H: size related to original image H size
    W: size related to original image W size
    C: channels

  Returns:
    shape: (batch_size, C, 2)
  """
  # see: https://github.com/tensorflow/tensorflow/issues/6271
  x = torch.transpose(x, [0, 3, 1, 2])
  x = torch.reshape(x, [batch_size * C, H * W])
  softmax = torch.nn.softmax(x)
  softmax = torch.reshape(softmax, [batch_size, C, H, W])
  softmax = torch.transpose(softmax, [0, 2, 3, 1])

  posx, posy = torch.meshgrid(
      torch.linspace(-1., 1., num=H), torch.linspace(-1., 1., num=W), indexing="ij")

  image_coords = torch.stack((posx, posy), axis=2)  # (H, W, 2)
  # Convert softmax to shape [B, H, W, C, 1]
  softmax = torch.unsqueeze(softmax, -1) #tf.expand_dims
  # Convert image coords to shape [H, W, 1, 2]
  image_coords = torch.unsqueeze(image_coords, 2) #tf.expand_dims
  # Multiply (with broadcasting) and reduce over image dimensions to get the
  # result of shape [B, C, 2].
  spatial_soft_argmax = torch.sum(softmax * image_coords, axis=[1, 2])
  return spatial_soft_argmax


class ConvMLP(nn.Module):
  """Conv MLP module."""

  def __init__(self, d_action, use_mdn, pretrained=True):
    super(ConvMLP, self).__init__()

    if pretrained:
      inception = torch.hub.KerasLayer(
          "https://tfhub.dev/google/imagenet/inception_v1/feature_vector/4",
          trainable=True)
      for i in inception.weights:
        if "Conv2d_1a_7x7/weights" in i.name:
          conv1weights = i
          break

    self.d_action = d_action

    input_shape = (None, 320, 160, 3)

    # filters = [64, 32, 16]
    filters = [64, 64, 64]

    if pretrained:
      self.conv1 = torch.nn.Conv2D(
          filters=filters[0],
          kernel_size=(7, 7),
          strides=(2, 2),
          weights=[conv1weights.numpy(), torch.zeros(64)],
          input_shape=input_shape)

    else:
      self.conv1 = torch.nn.Conv2D(
          filters=filters[0],
          kernel_size=(7, 7),
          strides=(2, 2),
          input_shape=input_shape)

    self.convd = torch.nn.Conv2D(
        filters=filters[0],
        kernel_size=(7, 7),
        strides=(2, 2),
        input_shape=input_shape)
    self.bnd = torch.nn.BatchNormalization()
    self.relud = torch.nn.ReLU()

    self.bn1 = torch.nn.BatchNormalization()
    self.relu1 = torch.nn.ReLU()

    self.conv2 = torch.nn.Conv2D(
        filters=filters[1], kernel_size=(5, 5), strides=(1, 1))
    self.bn2 = torch.nn.BatchNormalization()
    self.relu2 = torch.nn.ReLU()

    self.conv3 = torch.nn.Conv2D(
        filters=filters[2], kernel_size=(5, 5), strides=(1, 1))
    self.bn3 = torch.nn.BatchNormalization()
    self.relu3 = torch.nn.ReLU()

    self.flatten = torch.nn.Flatten()

    self.mlp = MlpModel(
        None,
        filters[-1] * 2,
        d_action,
        "relu",
        use_mdn,
        dropout=0.0,
        use_sinusoid=False)
    # note: no dropout on top of conv
    # the conv layers seem to help regularize the mlp layers

  def set_batch_size(self, batch_size):
    self.batch_size = batch_size

  #def call(self, x):
  def forward(self, x):
    """FPROP through module.

    Args:
      x: shape: (batch_size, H, W, C)

    Returns:
      shape: (batch_size, self.d_action)
    """
    rgb = self.relu1(self.bn1(self.conv1(
        x[:, :, :, :3])))  # only rgb channels through pre-trained
    depth = self.relud(self.bnd(self.convd(x[:, :, :, 3:])))  # for depth
    x = torch.cat((rgb, depth), axis=-1)

    x = self.relu2(self.bn2(self.conv2(x)))
    x = self.relu3(self.bn3(self.conv3(x)))
    # shape (B, ~H, ~W, C=16)

    x = compute_spatial_soft_argmax(x, self.batch_size)  # shape (B, C, 2)

    x = self.flatten(x)  # shape (B, C*2)

    return self.mlp(x)


class DeepConvMLP:
  """Deep conv MLP module."""

  def __init__(self, input_shape, d_action, use_mdn):
    del use_mdn

    self.batch_size = 4
    self.input_shape = input_shape
    self.d_action = d_action

    channel_depth_dim = 16
    in0, out0 = ResNet43_8s(
        self.input_shape,
        channel_depth_dim,
        prefix="s0_",
        cutoff_early=False,
        include_batchnorm=True)

    # out0 = tf.nn.avg_pool(out0, ksize=(1,4,4,1), strides=(1,4,4,1),
    #                       padding="SAME", data_format="NHWC")

    out0 = compute_spatial_soft_argmax(out0, self.batch_size, 320, 160,
                                       16)  # shape (B, C, 2)

    out0 = torch.nn.Flatten()(out0)
    out0 = torch.nn.Linear( # tf.keras.layers.Dense
        128,
        kernel_initializer="normal",
        bias_initializer="normal",
        activation="relu")(
            out0)
    out0 = torch.nn.Linear( # tf.keras.layers.Dense
        128,
        kernel_initializer="normal",
        bias_initializer="normal",
        activation="relu")(
            out0)
    out0 = torch.nn.Linear( #tf.keras.layers.Dense
        self.d_action, kernel_initializer="normal", bias_initializer="normal")(
            out0)

    #self.model = tf.keras.Model(inputs=[in0], outputs=[out0])

  def set_batch_size(self, x):
    pass

  @property
  def trainable_variables(self):
    return self.model.trainable_variables

  def __call__(self, x):
    return self.model(x)


def main():
  #cfg = tf.config.experimental
  #gpus = cfg.list_physical_devices("GPU")
  #mem_limit = 1024 * 4
  #dev_cfg = [cfg.VirtualDeviceConfiguration(memory_limit=mem_limit)]
  #cfg.set_virtual_device_configuration(gpus[0], dev_cfg)
  
  # check set_per_process_memory_fraction

  conv_mlp = ConvMLP(d_action=3, use_mdn=None)

  img = np.random.randn(7, 320, 160, 3)
  out = conv_mlp(img)
  print(out.shape)


if __name__ == "__main__":
  tf.compat.v1.enable_eager_execution()
  main()
