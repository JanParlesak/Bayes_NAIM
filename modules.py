import torch 
import torch.nn as nn
from . import BayesLinear

class Model(nn.Module):
  def __init__(self, in_features, out_features):
    super(Model, self).__init__()

    self.layer1 = BayesLinear(in_features = in_features, out_features=100)
    self.activation = nn.ReLU()
    self.layer2 = BayesLinear(in_features=100, out_features= out_features)

  def forward(self, x):

    kl_sum = 0

    out, kl = self.layer1(x)
    kl_sum += kl
    out = self.activation(out)

    out, kl = self.layer2(out)

    kl_sum += kl

    return out, kl_sum



class BayesFeature(nn.module):

  def __init__(self,
                 shallow_units: int,   # number of neurons in first layer
                 hidden_units = [],  # tuple of numbers of hidden units
                 activation = nn.ReLU(),
                 ):
        super().__init__()

        # Define Layers
        self.layers = nn.ModuleList([
            BayesLinear(shallow_units if i == 0 else hidden_units[i - 1], hidden_units[i])
            for i in range(len(hidden_units))
        ])

        self.output_layer = BayesLinear(hidden_units[-1], 1)

        self.activation = activation


  def forward(self, x):
    kl_sum = 0
    for i, layer in enumerate(self.layers):
      x, kl  = layer(x)
      kl_sum += kl
      x = self.activation(x)
    x, kl = self.output_layer(x)
    kl_sum += kl
    return x, kl 
  


class BayesNAM(torch.nn.Module):
  def __init__(self,
                n_features,
                shallow_units: int,   # number of neurons in first layer
                hidden_units = [],  # tuple of numbers of hidden units
                activation = nn.ReLU(),
                dropout: float = .5,
                return_output_lis = False
                ):
      super().__init__()

      self.shallow_units = shallow_units
      self.hidden_units = hidden_units
      self.activation = activation

      self.n_features = n_features
      self.return_output_lis = return_output_lis

      self.feature_nns = nn.ModuleList([
            BayesFeature(shallow_units=shallow_units,
                      hidden_units=hidden_units,
                      activation=activation,
                      dropout=dropout)
            for i in range(n_features)
        ])

      self.bias = nn.Parameter(torch.zeros(1))   #does the bias need a prior?

  def forward(self, x):
    eta = self.bias
    output_lis = []
    kl_total = 0 
    for feature, mod in zip(x.T, self.feature_nns):   #Kullback Leibler is additive include proof 
      feature = feature.unsqueeze(-1)
      out, kl = mod(feature)
      kl_total += kl
      output_lis.append(out)

    if self.return_output_lis:
      return output_lis

    else:
       conc_out = torch.cat(output_lis, dim=-1)
       out = torch.sum(conc_out, dim=-1) + self.bias

       return out, kl_total
    

