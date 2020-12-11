import numpy as np
from numpy.random import randn

class RNN:
  def __init__(self, insize, osize, hsize=64):
    self.Weighthtoh = randn(hsize, hsize) / 1000
    self.Weightintoh = randn(hsize, insize) / 1000
    self.Weighthtoout = randn(osize, hsize) / 1000

    self.bias_hid = np.zeros((hsize, 1))
    self.bias_out = np.zeros((osize, 1))

  def forward(self, inputs):
    hid = np.zeros((self.Weighthtoh.shape[0], 1))
    self.last_inputs = inputs
    self.last_hids = { 0: hid }

    for ind, inp in enumerate(inputs):
      hid = np.tanh(self.Weightintoh @ inp + self.Weighthtoh @ hid + self.bias_hid)
      self.last_hids[ind + 1] = hid
    out = self.Weighthtoout @ hid + self.bias_out

    return out, hid

  def back_propogation(self, dy, learn_rate=2e-2):
    n = len(self.last_inputs)
    dWeighthtoout = dy @ self.last_hids[n].T
    dbias_out = dy

    dWeighthtoh = np.zeros(self.Weighthtoh.shape)
    dWeightintoh = np.zeros(self.Weightintoh.shape)
    dbias_hid = np.zeros(self.bias_hid.shape)

    dh = self.Weighthtoout.T @ dy
    for t in reversed(range(n)):
      temp = ((1 - self.last_hids[t + 1] ** 2) * dh)
      dbias_hid += temp
      dWeighthtoh += temp @ self.last_hids[t].T
      dWeightintoh += temp @ self.last_inputs[t].T

      dh = self.Weighthtoh @ temp

    for d in [dWeightintoh, dWeighthtoh, dWeighthtoout, dbias_hid, dbias_out]:
      np.clip(d, -1, 1, out=d)
    #Clip is to prevent the exploding gradient problem

    # Updating the weights and bias of the layers
    self.Weighthtoh -= learn_rate * dWeighthtoh
    self.Weightintoh -= learn_rate * dWeightintoh
    self.Weighthtoout -= learn_rate * dWeighthtoout
    self.bias_hid -= learn_rate * dbias_hid
    self.bias_out -= learn_rate * dbias_out
