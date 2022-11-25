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

"""Transport module."""

import sys
sys.path.append('.')

import numpy as np
from ravens.models.resnet import ResNet43_8s, init_weights
from ravens.utils import utils

import torch
import torch.nn.functional as F

import cv2
from torchinfo import summary

from ravens.models.my_pytorch_model2.model import ModelKernel
from ravens.models.my_pytorch_model.model import ModelLabel

class Transport:
  """Transport module."""

  def __init__(self, in_shape, n_rotations, crop_size, preprocess):
    """Transport module for placing.

    Args:
      in_shape: shape of input image.
      n_rotations: number of rotations of convolving kernel.
      crop_size: crop size around pick argmax used as convolving kernel.
      preprocess: function to preprocess input images.
    """
    self.iters = 0
    self.n_rotations = n_rotations
    self.crop_size = crop_size  # crop size must be N*16 (e.g. 96)
    self.preprocess = preprocess

    self.pad_size = int(self.crop_size / 2)
    self.padding = np.zeros((3, 2), dtype=int)
    self.padding[:2, :] = self.pad_size

    in_shape = np.array(in_shape)
    in_shape[0:2] += self.pad_size * 2
    in_shape = tuple(in_shape)

    # Crop before network (default for Transporters in CoRL submission).
    kernel_shape = (self.crop_size, self.crop_size, in_shape[2])

    if not hasattr(self, 'output_dim'):
      self.output_dim = 3
    if not hasattr(self, 'kernel_dim'):
      self.kernel_dim = 3

    self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print('device:{}'.format(self.device))

    # 2 fully convolutional ResNets with 57 layers and 16-stride
    self.model1 = ResNet43_8s(in_shape, self.output_dim).to(self.device)
    self.model2 = ResNet43_8s(kernel_shape, self.kernel_dim).to(self.device)
    #self.model1 = ModelLabel()
    #self.model2 = ModelKernel()
    #self.optim = torch.optim.Adam(list(self.model1.parameters()) + list(self.model2.parameters()), lr=1e-4)

    self.optim = torch.optim.Adam([
                {'params': self.model1.parameters()},
                {'params': self.model2.parameters()}
            ], lr=1e-4)#, momentum=0.9)

    #self.optim = torch.optim.Adam(self.model1.parameters(), lr=1e-4)
    self.metric = []#torch.mean#(name='loss_transport')

    #self.model1.apply(init_weights)
    #self.model2.apply(init_weights)

    # if not self.six_dof:
    #   in0, out0 = ResNet43_8s(in_shape, output_dim, prefix="s0_")
    #   if self.crop_bef_q:
    #     # Passing in kernels: (64,64,6) --> (64,64,3)
    #     in1, out1 = ResNet43_8s(kernel_shape, kernel_dim, prefix="s1_")
    #   else:
    #     # Passing in original images: (384,224,6) --> (394,224,3)
    #     in1, out1 = ResNet43_8s(in_shape, output_dim, prefix="s1_")
    # else:
    #   in0, out0 = ResNet43_8s(in_shape, output_dim, prefix="s0_")
    #   # early cutoff just so it all fits on GPU.
    #   in1, out1 = ResNet43_8s(
    #       kernel_shape, kernel_dim, prefix="s1_", cutoff_early=True)

  # def set_bounds_pixel_size(self, bounds, pixel_size):
  #   self.bounds = bounds
  #   self.pixel_size = pixel_size

  def correlate(self, in0, in1, softmax):
    """Correlate two input tensors."""

    #test
    if False:
      inA = torch.Tensor([[1,2,3],[4,5,6],[7,8,9]]).reshape(1,1,3,3)
      inB = torch.Tensor([[11,12,13],[14,15,16],[17,18,19]]).reshape(1,1,3,3)
      print('ab:{} {}'.format(inA.shape, inB.shape))
      #inA = torch.Tensor([[1,4,7],[2,5,8],[3,6,9]])
      #inB = torch.Tensor([[11,15,17],[12,15,18],[13,16,19]])
      output = F.conv2d(inA, inB, padding='same')#, data_format='NHWC')
      print('abo:{} {} {}'.format(inA.shape, inB.shape, output.shape))
      exit(0)

    #print('in0, in1:{}'.format((in0.shape, in1.shape)))
    output = F.conv2d(in0, in1)#, data_format='NHWC')
    if softmax:
      #print('softmax')
      output_shape = output.shape
      output = torch.reshape(output, (1, np.prod(output.shape)))
      output = F.softmax(output).cpu().detach().numpy()
      output = np.float32(output).reshape(output_shape[1:])
    #print('output:{}'.format(output.shape))
    #output = torch.nn.functional.normalize(output)
    #exit(0)
    return output

  #https://stackoverflow.com/questions/64197754/how-do-i-rotate-a-pytorch-image-tensor-around-its-center-in-a-way-that-supports
  def get_rot_mat(self, theta):
      theta = torch.tensor(theta)
      return torch.tensor([[torch.cos(theta), -torch.sin(theta), 0],
                          [torch.sin(theta), torch.cos(theta), 0]])


  def rot_img(self, x, theta, dtype):
      rot_mat = self.get_rot_mat(theta)[None, ...].type(dtype).repeat(x.shape[0],1,1)
      grid = F.affine_grid(rot_mat, x.size()).type(dtype)
      x = F.grid_sample(x, grid)
      return x


  def forward(self, in_img, p, softmax=True):
    """Forward pass."""
    img_unprocessed = np.pad(in_img, self.padding, mode='constant')
    input_data = self.preprocess(img_unprocessed.copy())
    in_shape = (1,) + input_data.shape
    input_data = input_data.reshape(in_shape)
    in_tensor = torch.from_numpy(input_data).to(self.device)#tf.convert_to_tensor(input_data, dtype=tf.float32)

    if True:
      #print('split:{}'.format((type(img_unprocessed), img_unprocessed.shape)))
      split0, split1 = np.split(img_unprocessed, 2, axis=2)
      #print('split:{}'.format((split0.shape, split1.shape)))
      es = split0
      cv2.imshow('rgb', es / 255)
      es = split1
      cv2.imshow('depth', es * 10)
      cv2.waitKey(1)

    # Rotate crop.
    pivot = np.array([p[1], p[0]]) + self.pad_size
    rvecs = self.get_se2(self.n_rotations, pivot)

    # Crop before network (default for Transporters in CoRL submission).
    crop = torch.from_numpy(input_data.copy()).to(self.device)#tf.convert_to_tensor(input_data.copy(), dtype=tf.float32)
    #print('At crop0:{} | {}'.format(crop.shape, rvecs.shape))
    crop = crop.repeat([self.n_rotations,1,1,1])
    #print('Ct crop0:{} | {}'.format(crop.shape, rvecs.shape))
    # Permute for the rotation BWHC -> BCHW
    #crop = crop.permute(0, 3, 2, 1)
    # tensorflow.org/api_docs/python/tf/raw_ops/ImageProjectiveTrsansformV3
    #print('Bt crop0:{} | {}'.format(crop.shape, rvecs.shape))
    dtype =  torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
    for i in range(0, self.n_rotations):
      #Rotation by np.pi/2 with autograd support:
      rotated_im = self.rot_img(crop[i].unsqueeze(0), i * 2 * np.pi / self.n_rotations, dtype) # Rotate image by 90 degrees.
      crop[i] = rotated_im
      # if the crop has been permuted
      #c = crop[i,:3,:,:].permute(2,1,0)
      # otherwise
      #c = crop[i,:,:,:3]
      #print('crop:{}'.format(c.shape))
      #es = c.detach().cpu().numpy()
      #cv2.imshow('crop' + str(i), es)
    #cv2.waitKey(0)
    #exit(0)
    # Permute back from rotation BCHW -> BWHC
    #crop = crop.permute(0, 3, 2, 1)



    #crop = tfa_image.transform(crop, rvecs, interpolation='NEAREST') # todo
    #print('t crop1:{}'.format(crop.shape))
    crop = crop[:, p[0]:(p[0] + self.crop_size),
                p[1]:(p[1] + self.
                crop_size), :]
    #print('crop:{}'.format(crop))
    #print('crop shape:{}'.format(crop.shape))
    #exit(0)
    #logits, kernel_raw = self.model([in_tensor, crop])

    # No good (the data it looks like interleaved)
    #in_tensor_p = in_tensor.permute(0, 3, 1, 2) #BCWH
    #crop_p = crop.permute(0, 3, 1, 2)

    #print('in_tensor:{}'.format(in_tensor.shape))
    #print('crop:{}'.format(crop.shape))
    #exit(0)

    # compared to the tensorflow, pytorch has different order
    # BHWC -> BCHW
    #in_tensor_p = in_tensor.permute(0, 3, 2, 1) #BCHW
    #crop_p = crop.permute(0, 3, 2, 1)
    in_tensor_p = in_tensor.permute(0, 3, 1, 2) #BCHW
    crop_p = crop.permute(0, 3, 1, 2)

    #print('in_tensor_p:{}'.format(in_tensor_p.shape))
    #print('crop_p:{}'.format(crop_p.shape))
    #exit(0)

    #in_tensor_p = in_tensor#.permute(0, 1, 2) #BCWH
    #crop_p = crop#.permute(0, 3, 1, 2)
    #print('ABC:{}'.format((in_tensor.shape, crop.shape)))

    #print('in_tensor_p:{}'.format(in_tensor_p))
    #print('crop_p shape:{}'.format(crop_p.shape))
    #print('in_tensor_p shape:{}'.format(in_tensor_p.shape))
    #exit(0)

    import time
    start_time = time.time()
    logits = self.model1(in_tensor_p)
    kernel_raw = self.model2(crop_p)
    #print("--- %s seconds ---" % (time.time() - start_time))    

    #print('logits:{}'.format(logits.shape))
    #print('kernel_raw:{}'.format(kernel_raw.shape))

    if False:
      es = logits[0].permute(1, 2, 0).detach().cpu().numpy()
      #es = logits[0].detach().cpu().numpy()
      cv2.imshow('esL', es)
      es = kernel_raw[0].permute(1, 2, 0).detach().cpu().numpy()
      #es = kernel_raw[0].detach().cpu().numpy()
      cv2.imshow('esK', es)
      #print(es.shape)
      cv2.waitKey(1)

    #print('crop:{}'.format(crop.shape))
    #summary(self.model2, input_size=crop_p.shape)
    #exit(0)

    #print('logits:{}'.format(logits))
    #print('logits shape:{}'.format(logits.shape))
    #print('kernel_raw:{}'.format(kernel_raw))
    #print('kernel_raw shape:{}'.format(kernel_raw.shape))
    #exit(0)

    # Crop after network (for receptive field, and more elegant).
    # logits, crop = self.model([in_tensor, in_tensor])
    # # crop = tf.identity(kernel_bef_crop)
    # crop = tf.repeat(crop, repeats=self.n_rotations, axis=0)
    # crop = tfa_image.transform(crop, rvecs, interpolation='NEAREST')
    # kernel_raw = crop[:, p[0]:(p[0] + self.crop_size),
    #                   p[1]:(p[1] + self.crop_size), :]

    # Obtain kernels for cross-convolution.
    #kernel_paddings = ([0, 0], [0, 1], [0, 1], [0, 0])
    # NOTE
    # Padding dimension is reversed
    #kernel_paddings = (0,0,0,1,0,1)
    kernel_paddings = (0,1,0,1,0,0) # <<<<
    #print('kernel_raw:{}'.format(kernel_raw.shape))
    kernel = torch.nn.functional.pad(kernel_raw, kernel_paddings, mode='constant')
    #print('kernel:{}'.format(kernel.shape))
    #logits = logits.permute(0, 1, 3, 2)
    #kernel = kernel.permute(0, 1, 3, 2)

    #print('logits:{}'.format(logits.shape))
    #print('kernel:{}'.format(kernel.shape))


    es = logits[0].permute(1, 2, 0).detach().cpu().numpy()
    cv2.imshow('esL', es)
    es = kernel[0].permute(1, 2, 0).detach().cpu().numpy()
    cv2.imshow('esK', es * 10)
    cv2.waitKey(1)


    #logits = logits.permute(0, 1, 3, 2)
    #kernel = kernel.permute(0, 1, 3, 2)

    #print('logits:{}'.format(logits))
    #print('logits shape:{}'.format(logits.shape))
    #print('kernel_raw:{}'.format(kernel))
    #print('kernel_raw shape:{}'.format(kernel.shape))

    return self.correlate(logits, kernel, softmax)

  def train(self, in_img, p, q, theta, backprop=True):
    """Transport pixel p to pixel q.

    Args:
      in_img: input image.
      p: pixel (y, x)
      q: pixel (y, x)
      theta: rotation label in radians.
      backprop: True if backpropagating gradients.

    Returns:
      loss: training loss.
    """
    #print('transport p:{} q:{} theta:{}'.format(p, q, theta))
    self.metric = []#.reset_states()
    if True:

      #print('transport::train')
      #print('in_img:{}'.format(in_img.shape))

      output = self.forward(in_img, p, softmax=False)
      #print('outputS:{}'.format(output.shape))
      #exit(0)

      itheta = theta / (2 * np.pi / self.n_rotations)
      itheta = np.int32(np.round(itheta)) % self.n_rotations

      # Get one-hot pixel label map.      
      label_size = (self.n_rotations,) + in_img.shape[:2]
      #print('label_size:{}'.format(label_size))
      label = np.zeros(label_size)
      label[itheta, q[0], q[1]] = 1
      #print('label:{}'.format((q, itheta)))
      #print('labelshape:{}'.format(label.shape))

      # extract the label associated to the p crop region marked with
      # q point.
      #labelcrop = label[:,p[0]:(p[0] + self.crop_size),
      #            p[1]:(p[1] + self.crop_size)]
      #print('labelcrop:{}'.format(labelcrop.shape))
      #exit(0)


      es = label[itheta,:,:]
      cv2.imshow('transport_label', es * 10)
      #for i in range(0, 36):
      #  es = label[i,:,:]
      #  cv2.imshow('transport_label' + str(i), es * 10)
      #es = labelcrop[itheta,:,:]
      #print('es0:{}'.format(es.shape))
      #cv2.imshow('transport_labelcrop', es * 100)
      #for i in range(0, 36):
      #  es = np.reshape(output[0,i,:,:].detach().cpu().numpy(), (320, 160))
      #  cv2.imshow('transport_output' + str(i), es)
      es = np.reshape(output[0,itheta,:,:].detach().cpu().numpy(), (320, 160))
      cv2.imshow('transport_output', es)
      cv2.waitKey(1)


      #print('lo:{}'.format((label.shape, output.shape)))

      # Get loss (default).
      label = label.reshape(1, np.prod(label.shape))
      label = torch.tensor(label).to(self.device)#, dtype=tf.float32)
      output = torch.reshape(output, (1, np.prod(output.shape)))

      # Get loss only for the observer itheta matrix
      #label = label[itheta].reshape(1, np.prod(label[itheta].shape))
      #label = torch.tensor(label).to(self.device)#, dtype=tf.float32)
      #output = torch.reshape(output[0,itheta,:,:], (1, np.prod(output[0,itheta,:,:].shape)))

      #print('lo:{}'.format((label.shape, output.shape)))


      #print('outputB:{}'.format(output.shape))
      #print('label:{}'.format(label.shape))
      #print('lo:{}'.format((label.shape, output.shape)))
      #lab_reshape = label[:, 0:1000000].view(1000, -1)
      #print('label_reshape:{}'.format(lab_reshape))
      #output_reshape = output[:, 0:1000000].view(1000, -1)
      #print('output_reshape:{}'.format(output_reshape))

      #es = lab_reshape.detach().cpu().numpy()
      #cv2.imshow('label', es * 10)
      #es = output_reshape.detach().cpu().numpy()
      #cv2.imshow('output', es)
      #cv2.waitKey(1)

      #print('ol:{}'.format((output.shape, label.shape)))

      #loss = F.cross_entropy(output, label)
      loss = torch.nn.functional.cross_entropy(output, label)
      #print('loss:{}'.format(loss.shape))
      loss = torch.mean(loss)
      #print('loss:{}'.format((loss, loss.shape)))

      # Backpropagate
      if backprop:
        self.optim.zero_grad()
        loss.backward()
        self.optim.step()
        self.metric.append(loss)

    self.iters += 1
    #print('loss:{}'.format(loss.item()))
    return loss.cpu().detach().numpy()#np.float32(loss)

  def get_se2(self, n_rotations, pivot):
    """Get SE2 rotations discretized into n_rotations angles counter-clockwise."""
    rvecs = []
    for i in range(n_rotations):
      theta = i * 2 * np.pi / n_rotations
      rmat = utils.get_image_transform(theta, (0, 0), pivot)
      rvec = rmat.reshape(-1)[:-1]
      rvecs.append(rvec)
    return np.array(rvecs, dtype=np.float32)

  def load(self, fname1, fname2):
    self.model1.load_state_dict(torch.load(fname1))
    self.model2.load_state_dict(torch.load(fname2))

  def save(self, fname1, fname2):
    torch.save(self.model1.state_dict(), fname1)
    torch.save(self.model2.state_dict(), fname2)


def main():
  transport = Transport(
        in_shape=(320,160,6),
        n_rotations=36,
        crop_size=64,
        preprocess=utils.preprocess)

if __name__ == "__main__":
    main()