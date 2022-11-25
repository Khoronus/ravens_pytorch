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
    else:
      k = 1

    self.fc1 = torch.nn.Linear( # tf.keras.layers.Dense
        128,
        input_shape=(batch_size, d_obs * k),
        kernel_initializer="normal",
        bias_initializer="normal",
        activation=activation)
    self.drop1 = torch.nn.Dropout(dropout)
    self.fc2 = torch.nn.Linear( # tf.keras.layers.Dense
        128,
        kernel_initializer="normal",
        bias_initializer="normal",
        activation=activation)
    self.drop2 = torch.nn.Dropout(dropout)

    self.fc3 = torch.nn.Linear(
        d_action, kernel_initializer="normal", bias_initializer="normal")

    self.mdn = mdn
    if self.mdn:
      k = 26
      self.mu = torch.nn.Linear((d_action * k),
                                      kernel_initializer="normal",
                                      bias_initializer="normal")
      # Variance should be non-negative, so exp()
      self.logvar = torch.nn.Linear(
          k, kernel_initializer="normal", bias_initializer="normal")

      # mixing coefficient should sum to 1.0, so apply softmax
      self.pi = torch.nn.Linear(
          k, kernel_initializer="normal", bias_initializer="normal")
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

  def call(self, x):
    """FPROP through module.

    Args:
      x: shape: (batch_size, obs_dimension)

    Returns:
      shape: (batch_size, action_dimension)  (if MDN)
      shape of pi: (batch_size, num_gaussians)
      shape of mu: (batch_size, num_gaussians*action_dimension)
      shape of var: (batch_size, num_gaussians)
    """
    obs = x * 1.0

    # if self.normalize_input:
    #   x = (x - self.obs_train_mean) / (self.obs_train_std + 1e-7)

    def cs(x):
      if self.use_sinusoid:
        sin = torch.sin(x)
        cos = torch.cos(x)
        return torch.cat((x, cos, sin), axis=1)
      else:
        return x

    x = self.drop1(self.fc1(cs(obs)))
    x = self.drop2(self.fc2(torch.cat((x, cs(obs)), axis=1)))

    x = torch.cat((x, cs(obs)), axis=1)

    if not self.mdn:
      x = self.fc3(x)
      return x

    else:
      pi = self.pi(x)
      pi = pi / self.temperature
      pi = self.softmax(pi)

      mu = self.mu(x)
      var = tf.math.exp(self.logvar(x))
      return (pi, mu, var)

def main():
  batch_size = 10
  obs_dim = 10
  act_dim = 6
  use_mdn = True
  self.model = MlpModel(
      batch_size, obs_dim, act_dim, 'relu', use_mdn, dropout=0.1)

if __name__ == "__main__":
    main()
