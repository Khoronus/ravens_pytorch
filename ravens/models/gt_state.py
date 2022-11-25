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

"""MLP ground-truth state module."""

import torch
import torch.nn as nn

class MlpModel(nn.Module):
  """MLP ground-truth state module."""

  def __init__(self,
               batch_size,
               d_obs,
               d_action,
               activation="relu",
               mdn=False,
               dropout=0.2,
               use_sinusoid=True):
    super(MlpModel, self).__init__()
    self.normalize_input = True

    self.use_sinusoid = use_sinusoid
    if self.use_sinusoid:
      k = 3
      ks = 3
    else:
      k = 1
      ks = 1

    in_features = 64
    self.fc1 = torch.nn.Linear( # tf.keras.layers.Dense
        in_features=d_obs * k, out_features=128)
    self.drop1 = torch.nn.Dropout(dropout)
    self.fc2 = torch.nn.Linear( # tf.keras.layers.Dense
        in_features=128 + (d_obs * k), out_features=128)
    self.drop2 = torch.nn.Dropout(dropout)

    self.fc3 = torch.nn.Linear(
        128 + (d_obs * k), d_action)

    self.mdn = mdn
    if self.mdn:
      k = 26
      self.mu = torch.nn.Linear(128 + (d_obs * ks), (d_action * k))
      # Variance should be non-negative, so exp()
      self.logvar = torch.nn.Linear(128 + (d_obs * ks), k)

      # mixing coefficient should sum to 1.0, so apply softmax
      self.pi = torch.nn.Linear(128 + d_obs * ks,
          k)
      self.softmax = torch.nn.Softmax()
      self.temperature = 2.5

  def reset_states(self):
    pass

  def set_normalization_parameters(self, obs_train_parameters):
    """Set normalization parameters.

    Args:
      obs_train_parameters: dict with key, values:
        - 'mean', numpy.ndarray of shape (obs_dimension)
        - 'std', numpy.ndarray of shape (obs_dimension)
    """
    self.obs_train_mean = obs_train_parameters["mean"]
    self.obs_train_std = obs_train_parameters["std"]

  #def call(self, x):
  def forward(self, x):
    """FPROP through module.

    Args:
      x: shape: (batch_size, obs_dimension)

    Returns:
      shape: (batch_size, action_dimension)  (if MDN)
      shape of pi: (batch_size, num_gaussians)
      shape of mu: (batch_size, num_gaussians*action_dimension)
      shape of var: (batch_size, num_gaussians)
    """
    #print('x:{}'.format(x.shape))
    obs = x * 1.0

    # if self.normalize_input:
    #   x = (x - self.obs_train_mean) / (self.obs_train_std + 1e-7)

    def cs(x):
      if self.use_sinusoid:
        sin = torch.sin(x)
        cos = torch.cos(x)
        #print('cs:{} {} {}'.format(x.shape, cos.shape, sin.shape))
        return torch.cat((x, cos, sin), axis=1)
      else:
        return x

    x = self.drop1(self.fc1(cs(obs)))
    #print('x0:{}'.format(x.shape))
    #print('x1:{}'.format(cs(obs).shape))
    x = torch.cat((x, cs(obs)), axis=1)
    #print('x2:{}'.format(x.shape))

    x = self.drop2(self.fc2(x))

    x = torch.cat((x, cs(obs)), axis=1)

    if not self.mdn:
      x = self.fc3(x)
      return x

    else:
      pi = self.pi(x)
      pi = pi / self.temperature
      pi = self.softmax(pi)

      mu = self.mu(x)
      var = torch.exp(self.logvar(x))
      return (pi, mu, var)

def main():
  batch_size = 5
  obs_dim = 10
  act_dim = 8
  use_mdn = True
  model = MlpModel(
      batch_size, obs_dim, act_dim, 'relu', use_mdn, dropout=0.1)
  t1 = torch.randn([batch_size,obs_dim]) # as channels first in pytorch
  res = model(t1)
  print('t1:{}'.format(res))

if __name__ == "__main__":
    main()
