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

"""Attention module."""

import sys
sys.path.append('.')
import numpy as np
import torch
import torch.nn.functional as F

from ravens.models.resnet import ResNet36_4s
from ravens.models.resnet import ResNet43_8s, init_weights
from ravens.utils import utils

import cv2 
import torchinfo
from torchviz import make_dot, make_dot_from_trace

class Attention:
  """Attention module."""

  def __init__(self, in_shape, n_rotations, preprocess, lite=False):
    self.n_rotations = n_rotations
    self.preprocess = preprocess

    max_dim = np.max(in_shape[:2])

    self.padding = np.zeros((3, 2), dtype=int)
    pad = (max_dim - np.array(in_shape[:2])) / 2
    self.padding[:2] = pad.reshape(2, 1)

    in_shape = np.array(in_shape)
    in_shape += np.sum(self.padding, axis=1)
    in_shape = tuple(in_shape)

    # Initialize fully convolutional Residual Network with 43 layers and
    # 8-stride (3 2x2 max pools and 3 2x bilinear upsampling)
    if lite:
      print('00000')
      d_in, d_out = ResNet36_4s(in_shape, 1)
    else:
      print('11111')
      self.model = ResNet43_8s(in_shape, 1)
    self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print('device:{}'.format(self.device))
    self.model.to(self.device)
    #self.model.apply(init_weights)

    #self.model = tf.keras.models.Model(inputs=[d_in], outputs=[d_out])    FIX THIS PART (probably no need)
    self.optim = torch.optim.Adam(self.model.parameters(), lr=1e-4)
    self.metric = []#torch.mean
    #torch.autograd.set_detect_anomaly(True)

  def forward(self, in_img, softmax=True):
    """Forward pass."""
    in_data = np.pad(in_img, self.padding, mode='constant')
    in_data = self.preprocess(in_data)
    in_shape = (1,) + in_data.shape
    in_data = in_data.reshape(in_shape)
    in_tens = torch.from_numpy(in_data) #.to(float)   WARNING EXPECTED TO BE, dtype=tf.float32)

    # Rotate input.
    pivot = np.array(in_data.shape[1:3]) / 2
    rvecs = self.get_se2(self.n_rotations, pivot)
    #print('in_tens:{}'.format(in_tens.shape))
    in_tens = in_tens.repeat([self.n_rotations, 1, 1, 1])#, axis=0) #tf.repear
    #print('in_tens:{}'.format(in_tens.shape))
    #in_tens = tfa_image.transform(in_tens, rvecs, interpolation='NEAREST')   # temporarily skip

    # Forward pass.
    in_tens = torch.split(in_tens, self.n_rotations) #tf.split
    logits = ()
    for x in in_tens:
      #print('x shape:{}'.format(x.shape))
      x_t = x.permute(0, 3, 1, 2).to(self.device)
      #print('x_t shape:{}'.format(x_t.shape))
      res = self.model(x_t)
      #print('Ares shape:{}'.format(res.shape))
      logits += (res,) #x
      #print('res:{}'.format(res))
      #exit(0)
      #torchinfo.summary(self.model, input_size=x_t.shape, col_names = ('input_size', 'output_size', 'num_params', 'kernel_size', 'mult_adds'))
      #make_dot(res, params=dict(self.model.named_parameters()), show_attrs=True, show_saved=True).render("rnn_torchviz", format="png")
      #exit(0)

    logits = torch.concat(logits, axis=0) # tf.concat
    #print('attention logits:{}'.format(logits.shape))

    #es = np.reshape(logits.detach().cpu().numpy(),(320,320))
    #cv2.imshow('attention_logits', es)
    #cv2.waitKey(1)


    # Rotate back output.
    rvecs = self.get_se2(self.n_rotations, pivot, reverse=True)
    #print('rvecs:{}'.format(rvecs))
    #logits = tfa_image.transform(logits, rvecs, interpolation='NEAREST')
    c0 = self.padding[:2, 0]
    c1 = c0 + in_img.shape[:2]
    #print('c01:{}'.format((c0, c1)))
    # compared to the tensorflow, pytorch has different order
    # BHWC -> BCHW
    logits = logits[:, :, c0[0]:c1[0], c0[1]:c1[1]]


    es = np.reshape(logits.detach().cpu().numpy(),(320,160))
    cv2.imshow('attention_logits', es)
    cv2.waitKey(1)

    #print('logits shape:{}'.format(logits.shape))
    #logits = logits.permute(3, 1, 2, 0)
    #print('logits shape:{}'.format(logits.shape))
    output = torch.reshape(logits, (1, np.prod(logits.shape)))
    #print('attention output shape:{}'.format(output.shape))
    if softmax:
      output = F.softmax(output).cpu().detach().numpy()
      #print('outputp:{}'.format(output))
      output = np.float32(output).reshape(logits.shape[1:])
    #print('output:{}'.format(output.shape))

    return output


  '''
  Note: It rotates the image so that the target is always aligned
        in an expected way.
        The full rotated output size is maxw,maxh but later crop
        around the expected target point for the size of the
        expected image.
        For the rotation in pytorch, an image rotation equal to
        pi/36 (or 2pi/36) should do the job.

        In the attention, the n_rotations is equal to 1.
        The network focus only on the position/location.
  '''

  def train(self, in_img, p, theta, backprop=True):
    """Train."""

    #print('attention p:{} theta:{}'.format(p, theta))

    self.metric = []
    output = self.forward(in_img, softmax=False)
    #print('attention output:{}'.format(output.shape))

    # Get label.
    #print('self.n_rotations:{}'.format(self.n_rotations))
    theta_i = theta / (2 * np.pi / self.n_rotations)
    theta_i = np.int32(np.round(theta_i)) % self.n_rotations
    label_size = in_img.shape[:2] + (self.n_rotations,)
    label = np.zeros(label_size, dtype=float)
    label[p[0], p[1], theta_i] = 1
    #label = np.ones(label_size, dtype=float)
    #label[:,:,:] = 1.0

    #print('label:{}'.format(label))
    #print('output:{}'.format(output))


    es = label[:,:,theta_i]
    cv2.imshow('attention_label', es * 10)
    #es = np.reshape(output.detach().cpu().numpy(),(320,160))
    #print('es1:{}'.format(es.shape))
    #cv2.imshow('attention_output', es)
    #cv2.waitKey(1)

    label = label.reshape(1, np.prod(label.shape))
    label = torch.from_numpy(label)#, dtype=tf.float32)

    #print('label:{}'.format(label.shape))
    #print('output:{}'.format(output.shape))
    #exit(0)

    if False:
      # Example of target with class indices
      input = torch.randn(3, 5, requires_grad=True)
      target = torch.randint(5, (3,), dtype=torch.int64)
      print('it:{}'.format((input.shape, target.shape)))
      loss = F.cross_entropy(input, target)
      loss.backward()
      print('loss0:{}'.format((input, target, loss)))
    
      # Example of target with class probabilities
      input = torch.randn(3, 5, requires_grad=True)
      target = torch.randn(3, 5).softmax(dim=1)
      print('it:{}'.format((input.shape, target.shape)))
      loss = F.cross_entropy(input, target)
      loss.backward()
      print('loss1:{}'.format((input, target, loss)))

      x = torch.FloatTensor([[1.,0.,0.],
                            [0.,1.,0.],
                            [0.,0.,1.]])
      y = torch.LongTensor([0,1,2])
      print('xy:{}'.format((x.shape, y.shape)))
      print(torch.nn.functional.cross_entropy(x, y))
      print(F.softmax(x, 1).log())
      print(F.log_softmax(x, 1))
      print(F.nll_loss(F.log_softmax(x, 1), y))




      logits = torch.FloatTensor([[1.0, 0.0, 3.1], [0.5, 1.0, 0.0], [0.0, 5.2, 1.0]])
      labels = torch.LongTensor([1.1, 0.2, 2.3])
      print('logits:{}'.format((logits.shape, labels.shape)))
      for i in range(0, 10):
        loss = torch.nn.functional.cross_entropy(logits, labels) #tf.nn.softmax_cross_entropy_with_logits
        print('lossloss:{}'.format(loss))
        print('ll:{}'.format((logits, labels)))

        # Backpropagate
        self.optim.zero_grad()
        loss.backward()
        self.optim.step()
        self.metric.append(loss)

      exit(0)


    # Get loss.
    #print('type:{}'.format((type(output), type(label))))
    loss = torch.nn.functional.cross_entropy(output.to(self.device).float(), label.to(self.device).float()) #tf.nn.softmax_cross_entropy_with_logits
    #loss = torch.nn.functional.mse_loss(output.to(self.device).float(), label.to(self.device).float()) #tf.nn.softmax_cross_entropy_with_logits
    loss = torch.mean(loss) # tf.reduce_mean


    # Backpropagate
    self.optim.zero_grad()
    loss.backward()
    self.optim.step()
    self.metric.append(loss)

    return loss.cpu().detach().numpy()

  def load(self, path):
    #self.model.load_weights(path)
    self.model.load_state_dict(torch.load(path))

  def save(self, filename):
    torch.save(self.model.state_dict(), filename)

  def get_se2(self, n_rotations, pivot, reverse=False):
    """Get SE2 rotations discretized into n_rotations angles counter-clockwise."""
    rvecs = []
    for i in range(n_rotations):
      theta = i * 2 * np.pi / n_rotations
      theta = -theta if reverse else theta
      rmat = utils.get_image_transform(theta, (0, 0), pivot)
      rvec = rmat.reshape(-1)[:-1]
      rvecs.append(rvec)
    return np.array(rvecs, dtype=np.float32)

def main():
  attention = Attention([64,64,3], 10, False)

if __name__ == "__main__":
  main()
