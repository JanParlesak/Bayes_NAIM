import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import os
import os.path as osp
import seaborn as sns
from modules import *

from diffae.templates import cxr128_autoenc_130M
from diffae.experiment import LitModel


def load_encoder(device):
   
   model_resolution = 128
   T_inv = "200"
   T_step = 100

   model_config = cxr128_autoenc_130M() 

   print(model_config.name)

   diffae_weight = ""
   diffae_latent = ""
   cls_weight = ""

   weight_dir_path = f'diffae/checkpoints/{model_config.name}'
   if not osp.exists(weight_dir_path): os.makedirs(weight_dir_path)
   model_weights =  '{weight_dir_path}/last.ckpt' '{model_download_path}'
   latent_weights =  '{weight_dir_path}/latent.pkl' '{latents_download_path}'

   classifer_config = cxr128_autoenc_130M()
   weight_dir_path = f'diffae/checkpoints/{classifer_config.name}'
   if not osp.exists(weight_dir_path): os.makedirs(weight_dir_path)
   checkpoint = '{weight_dir_path}/last.ckpt' '{cls_download_path}'

   device = device
   conf = cxr128_autoenc_130M()
   # print(conf.name)
   pretrained_encoder = LitModel(conf)
   state = torch.load(f'diffae/checkpoints/last.ckpt', map_location='cpu', weights_only=False)
   pretrained_encoder.load_state_dict(state['state_dict'], strict=False)
   
   return pretrained_encoder

def make_model(config, n_features):
    
    if config["name"] == "bnaim":

      device = config["device"]
      mode = config["mode"]
      
      pretrained_encoder = load_encoder(device=device)
      pretrained_encoder.ema_model.eval()
      pretrained_encoder.ema_model.to(device)

      bayes_mlp = BayesResFeature(n_input = 512, hid_dim = [500,500,500,500]) 
      bayes_nam = BayesNAM(n_features = n_features, hidden_units = [100, 100, 100], dropout_rate = config["dropout_rate"], feature_dropout_rate = config["feature_dropout_rate"],
                            prior_scale = config["prior_scale"])

      model = BayesImageNAM(pretrained_encoder = pretrained_encoder, bayes_mlp = bayes_mlp, bayes_feat_nam = bayes_nam)

    elif config["name"] == "bnam":

      model = BayesNAM(n_features = n_features, hidden_units = [100, 100, 100], dropout_rate= config["dropout_rate"], feature_dropout_rate = config["feature_dropout_rate"], prior_scale = config["prior_scale"])
    
    elif config["name"] == "nam":

      model = NAM(n_features=n_features, shallow_units=20, hidden_units=(100, 100, 100), activation= torch.nn.ReLU(), dropout= config["dropout_rate"], feature_dropout = config["feature_dropout_rate"])

    return model

