from .modules import Model

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.optim import Adam


def validate(model, val_loader, loss_fun, n_samples):

    val_loss = []

    model.eval()

    with torch.no_grad():
        for i, batch in enumerate(val_loader):

            input, target = batch

            target = target.to(device)
            input = input.to(device)

            output = []
            kl_div = []

            for sample in range(n_samples):
                out, kl = model(input)
                output.append(out)
                kl_div.append(kl)

            mean_pred = torch.mean(torch.stack(output), dim = 0)
            kl_loss = torch.mean(torch.stack(kl_div), dim = 0)
            log_lik_loss = loss_fun(mean_pred, target)
            loss = log_lik_loss + kl_loss
            val_loss.append(loss.cpu())

        mean_loss = np.mean(np.array(val_loss))

    return mean_loss



def train(model, optimizer, loss_fun, trainset, device, n_epochs, batch_size, n_samples, print_mod = 1):

    loss_lis = []
    overall_loss = []
    val_loss = []

    model = model.to(device)

    for epoch in range(n_epochs):

        for i, batch in enumerate(trainset):

            x, y = batch
            x = x.to(device)
            y = y.to(device)

            output = []
            kl_div = []

            for _ in range(n_samples):  #Not sure if sampling here is needed check that
                out, kl = model(x)
                output.append(out)
                kl_div.append(kl)

            # out, kl = model(x)


            mean_pred = torch.mean(torch.stack(output), dim = 0)
            kl_loss = torch.mean(torch.stack(kl_div), dim = 0)

            #print(kl_loss)

            loss = loss_fun(mean_pred, y)
            scaled_kl = kl_loss/batch_size
            loss += scaled_kl  #ELBO Loss add if loos_fun is negative log_likelihood

            #loss = loss_fun(out, y)
            #loss += kl * 0.1     # Why does this improve training so much? kl / batch_size and add sampling again


            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_lis.append(loss.cpu().detach())

        if epoch % print_mod == 0:

            mean_loss = np.mean(np.array(loss_lis))
            overall_loss += mean_loss
            loss_lis = []

            #validation_loss = validate(model, valset, loss_fun, n_samples)

            #val_loss.append(validation_loss)


            print(f'Epoch nr {epoch}: mean_train_loss = {mean_loss}')
            #, , validation_loss =  {validation_loss}')

    return overall_loss




def sample(model, n_samples, testloader):

  model.eval()
  mean_pred_list = []
  std_list = []

  with torch.no_grad():

    for i, batch in enumerate(testloader):

      output_mc = []

      data, target = batch

      data = data.to(device)
      target = target.to(device)

      for _ in range(n_samples):
        out, _ = model.forward(data)
        output_mc.append(out)

      output = torch.stack(output_mc)

      mean_pred_batch = torch.mean(output, dim = 0)
      std_batch = torch.sqrt(torch.var(output, dim = 0))
      mean_pred_list.append(mean_pred_batch)
      std_list.append(std_batch)

  return mean_pred_list, std_list





