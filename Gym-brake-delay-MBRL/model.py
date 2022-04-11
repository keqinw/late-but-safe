import os
import numpy as np
import torch
import torch.nn as nn
import operator
from functools import reduce
from utils.util import ZFilter

HIDDEN1_UNITS = 150
HIDDEN2_UNITS = 150
HIDDEN3_UNITS = 150

import logging
log = logging.getLogger('root')


class PENN(nn.Module):
    """
    (P)robabilistic (E)nsemble of (N)eural (N)etworks
    """

    def __init__(self, num_nets, state_dim, action_dim, learning_rate, device=None):
        """
        :param num_nets: number of networks in the ensemble
        :param state_dim: state dimension
        :param action_dim: action dimension
        :param learning_rate:
        """

        super().__init__()
        self.num_nets = num_nets
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.device = device

        # Log variance bounds
        self.max_logvar = torch.tensor(-3 * np.ones([1, self.state_dim]), dtype=torch.float, device=self.device)
        self.min_logvar = torch.tensor(-7 * np.ones([1, self.state_dim]), dtype=torch.float, device=self.device)

        # Create or load networks
        self.networks = nn.ModuleList([self.create_network(n) for n in range(self.num_nets)]).to(device=self.device)
        self.opt = torch.optim.Adam(self.networks.parameters(), lr=learning_rate)

    def forward(self, inputs):
        if not torch.is_tensor(inputs):
            inputs = torch.tensor(inputs, device=self.device, dtype=torch.float)
        return [self.get_output(self.networks[i](inputs)) for i in range(self.num_nets)]

    def get_output(self, output):
        """
        Argument:
          output: the raw output of a single ensemble member
        Return:
          mean and log variance
        """
        mean = output[:, 0:self.state_dim]
        raw_v = output[:, self.state_dim:]
        logvar = self.max_logvar - nn.functional.softplus(self.max_logvar - raw_v)
        logvar = self.min_logvar + nn.functional.softplus(logvar - self.min_logvar)
        return mean, logvar

    def get_loss(self, targ, mean, logvar):
        # TODO: write your code here
        sq_err = (mean - targ)**2 
        loss = sq_err / logvar.exp()
        loss += logvar
        return loss.sum()


        #return self.loss_fn(mean, targ, logvar.exp())

    def create_network(self, n):
        layer_sizes = [self.state_dim + self.action_dim, HIDDEN1_UNITS, HIDDEN2_UNITS, HIDDEN3_UNITS]
        layers = reduce(operator.add,
                        [[nn.Linear(a, b), nn.ReLU()]
                         for a, b in zip(layer_sizes[0:-1], layer_sizes[1:])])
        layers += [nn.Linear(layer_sizes[-1], 2 * self.state_dim)]
        return nn.Sequential(*layers)

    def train_model(self, inputs, targets, batch_size=128, num_train_itrs=5):
        """
        Training the Probabilistic Ensemble (Algorithm 2)
          :param inputs : state and action inputs. Assumes that inputs are standardized.
          :prams targets: resulting states
        """
        # TODO: write your code here

        losses = []
        rmses = []
        for i in range(num_train_itrs):
            batch_idx = np.random.randint(len(inputs), size=batch_size)
            X_train = torch.tensor(inputs[batch_idx]).float()
            y_train = torch.tensor(targets[batch_idx])
            loss = 0
            rmse = 0
            self.opt.zero_grad()
            output = self.forward(X_train)
            for j, net in enumerate(self.networks):
                mean, logvar = output[j]
                loss += self.get_loss(y_train, mean, logvar)
                rmse += ((mean - y_train)**2).mean().sqrt().item()
            loss.backward()
            self.opt.step()
            losses.append(loss.item())
            rmses.append(rmse)
        return losses, rmses















