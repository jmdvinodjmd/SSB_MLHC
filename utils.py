# set working directory
from random import SystemRandom
import pandas as pd
import numpy as np
import xgboost as xgb
# from hyperopt import hp, fmin, tpe, STATUS_OK, Trials
from sklearn.model_selection import cross_val_score
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import RobustScaler

import os
import pickle
from sklearn.model_selection import train_test_split
import wandb

import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import GridSearchCV
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn import metrics

# from torch._C import float32
import argparse
from asyncio.log import logger
import os, math
import logging
import torch
import numpy as np

import torch.nn as nn
import torch

import pickle
import json
import numpy as np
import random
from torch.utils.data import Dataset, DataLoader, random_split, WeightedRandomSampler
import torch
import os
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import load_iris
from sklearn.datasets import load_breast_cancer

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import torch.nn as nn
import torch
import math
import pandas as pd
import random

# Importing matplotlib and seaborn
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

import sys
import warnings

import numpy as np
from sklearn.utils import shuffle
import pandas as pd

from IPython.display import HTML, display
# import tabulate

from adapt.feature_based import DANN, WDGRL
from adapt.instance_based import KLIEP, KMM
import keras
from keras.optimizers.legacy import Adam
from keras import Model, Sequential
from keras.layers import Dense, Input, Dropout, Conv2D, MaxPooling2D, Flatten, Reshape, GaussianNoise, BatchNormalization

# import utils
# from utils import *

if not sys.warnoptions:
    warnings.simplefilter("ignore")
###########################################

def get_data_dict(name, r, c, n):
    data_dict = {}
    # load data path
    with open('paths.json', 'r') as json_file:
        file_path = json.load(json_file)[name]

    if name == 'no.pkl': #or name == 'diabetes_bias_data.pkl'or name == 'covid_bias_data.pkl':
        with open(file_path, 'rb') as file:
            data = pickle.load(file)
            data_name = str(n[0])+'R'+str(r[0])+'C'+str(c[0])
            data_dict[data_name] = [data['X_train'], data['y_train'],
                                            data['c_train'], data['X_val'],
                                            data['y_val'], data['c_val'],
                                            data['X_test'], data['y_test'],
                                            data['c_test']]

    else:
        with open(file_path, 'rb') as file:
            data = pickle.load(file)
        
        # print('--------', data.keys())

        # prepare a dictionary of data for easy processing
        for ri in r:
            for ci in c:
                for ni in n:
                    data_name = str(ni)+'R'+str(ri)+'C'+str(ci)
                    data_dict[data_name] = [data['X_train'+data_name], data['y_train'+data_name],
                                            data['c_train'+data_name], data['X_val'+data_name],
                                            data['y_val'+data_name], data['c_val'+data_name],
                                            data['X_test'+data_name], data['y_test'+data_name],
                                            data['c_test'+data_name]]
    
    return data_dict

def dict_to_file(name, results_sizes):
    '''
    Convert the nested dictionary to a dataframe and save
    '''
    flat_data = []
    for size, size_dict in results_sizes.items():
        for repeat, repeat_dict in size_dict.items():
            repeat_dict['Size'] = size
            repeat_dict['Repeat'] = repeat
            flat_data.append(repeat_dict)

    # Convert Results into Dataframe and save
    df = pd.DataFrame(flat_data)
    # Save DataFrame to Excel
    df.to_excel('./results/'+name +'.xlsx', index=False)

    logger.info(df)
    
def robust_scaler(X_train, X_test):
    # separate binary features
    bin_features = X_train.columns[X_train.nunique()==2]
    num_features = list(set(X_train.columns) - set(bin_features))

    # scale the data
    scaler = RobustScaler()
    scaler.fit(X_train[num_features])

    X_train[num_features] = scaler.transform(X_train[num_features])
    X_test[num_features] = scaler.transform(X_test[num_features])

    return X_train, X_test


def makedirs(dirname):
    if not os.path.exists(dirname):
        os.makedirs(dirname)

def save_checkpoint(state, save, epoch):
    if not os.path.exists(save):
        os.makedirs(save)
    filename = os.path.join(save, 'checkpt-%04d.pth' % epoch)
    torch.save(state, filename)

def get_logger(logpath, filepath, package_files=[], displaying=True, saving=True, debug=False):
    logger = logging.getLogger()
    if debug:
        level = logging.DEBUG
    else:
        level = logging.INFO
    logger.setLevel(level)
    if (logger.hasHandlers()):
        logger.handlers.clear()
    if saving:
        info_file_handler = logging.FileHandler(logpath, mode='w')
        info_file_handler.setLevel(level)
        logger.addHandler(info_file_handler)
    if displaying:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(level)
        logger.addHandler(console_handler)
    logger.info(filepath)

    for f in package_files:
        logger.info(f)
        with open(f, 'r') as package_f:
            logger.info(package_f.read())

    return logger

def print_metrics_binary_classification(y_true, prediction_probs, setting, verbose=1, logger=None, wandb=None):
    # if sum(y_true) != 0.:
    #     logger.info("Number of labeled examples: {}".format(y_true.shape[0]))
    #     logger.info("Number of examples with mortality 1: {}".format(sum(y_true == 1.)))
    # else:
    #     logger.info("Warning: Couldn't compute AUC -- all examples are from the same class")
    #     return

    prediction_probs = np.array(prediction_probs)
    # prediction_probs2 = np.transpose(np.append([1 - prediction_probs], [prediction_probs], axis=0))

    # define thresholds
    thresholds = np.arange(0, 1, 0.001)
    # evaluate each threshold
    scores = [metrics.f1_score(y_true, (prediction_probs >= t).astype('int')) for t in thresholds]
    # get best threshold
    ix = np.argmax(scores)
    best_threshold = thresholds[ix]
    # predictions = (prediction_probs >= best_threshold).astype(int)
    # f1 = metrics.f1_score(y_true, predictions, average='macro')
    # rate_y = float(y_true.sum()/y_true.shape[0])
    # best_threshold = np.clip(best_threshold, rate_y, rate_y*4)
    logger.info('Best Threshold=%.3f, f1-score=%.3f' % (best_threshold, scores[ix]))
    # logger.info('nonclip: %.3f, after clip Threshold=%.3f, f1-score=%.3f, event rate: %.3f' % (thresholds[ix], best_threshold, scores[ix], rate_y))
    
    # # calculate roc curves
    # fpr, tpr, thresholds = metrics.roc_curve(y_true, prediction_probs)
    # # calculate the g-mean for each threshold
    # gmeans = np.sqrt(tpr * (1-fpr))
    # # locate the index of the largest g-mean
    # ix = np.argmax(gmeans)
    # best_threshold = thresholds[ix]

    # predictions = (prediction_probs >= best_threshold).astype(int)
    # f2 = metrics.f1_score(y_true, predictions, average='macro')
    # # logger.info('Best Threshold=%f, G-Mean=%.3f' % (thresholds[ix], gmeans[ix]))
    
    # predictions = prediction_probs2.argmax(axis=1)
    predictions = (prediction_probs > best_threshold).astype(int)

    # # cf = metrics.confusion_matrix(y_true, predictions, labels=range(2))
    # # if verbose:
    # #     logger.info('Confusion matrix:')
    # #     logger.info(cf)

    acc = metrics.accuracy_score(y_true, predictions)
    precision = metrics.precision_score(y_true, predictions)
    recall = metrics.recall_score(y_true, predictions)
    # (precisions, recalls, thresholds) = metrics.precision_recall_curve(y_true, predictions)
    # auprc = metrics.auc(recalls, precisions)
    f1macro = metrics.f1_score(y_true, predictions, average='macro')

    auroc = metrics.roc_auc_score(y_true, prediction_probs)
    # logger.info('withoutclip:'+str(f1)+' with clip:' +str(f1macro) + ' f2:' + str(f2))
    auprc = metrics.average_precision_score(y_true, prediction_probs)
    
    results = {
               setting +' AUROC': auroc,
               setting +' AUPRC': auprc,
            #    setting +' Accuracy': acc,
            #    setting +' Precision': precision,
            #    setting +' Recall': recall,
            #    setting +' F1-score': f1macro
               }
        
    if verbose:
        for key in results:
            logger.info('{} = {:.4f}'.format(key, results[key]))
    if wandb is not None:
        wandb.log(results)

    return results, best_threshold

def get_loaders(data, weights=None, batch_size=64, is_train=True, device='cpu'):
    # Convert the data to PyTorch tensors
    input_size = data[0].shape[1]
    if weights is not None:
        weights = torch.tensor(weights, dtype=torch.float32, device=device)
    if len(data)==2:
        Xb = torch.tensor(data[0], dtype=torch.float32, device=device)
        yb = torch.tensor(data[1], dtype=torch.float32, device=device)

        dataset = Dataset(Xb, yb, weights=weights)

    elif len(data)==3:
        Xb = torch.tensor(data[0], dtype=torch.float32, device=device)
        yb = torch.tensor(data[1], dtype=torch.float32, device=device)
        s = torch.tensor(data[2], dtype=torch.float32, device=device)

        dataset = Dataset(Xb, yb, s, weights=weights)

    elif len(data)==5:
        Xb = torch.tensor(data[0], dtype=torch.float32, device=device)
        yb = torch.tensor(data[1], dtype=torch.float32, device=device)
        Xu = torch.tensor(data[2], dtype=torch.float32, device=device)
        yu = torch.tensor(data[3], dtype=torch.float32, device=device)
        s = torch.tensor(data[4], dtype=torch.float32, device=device)

        dataset = Dataset(Xb, yb, Xu, yu, s, weights=weights)

    train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=is_train)

    return train_loader, input_size

class Dataset(torch.utils.data.Dataset):
  '''
    Creates dataset in Pytorch.
    It can take 2, 3 or 5 arrays for biased (X,y) and unbiased (X,y,s) datasets.
  '''
  def __init__(self, *args, weights=None):
        self.args = args
        self.weights = weights
        if len(args)==2:
            self.Xb = args[0]
            self.yb = args[1]
        elif len(args)==3:
            self.Xb = args[0]
            self.yb = args[1]
            self.s = args[2]
        elif len(args)==5:
            self.Xb = args[0]
            self.yb = args[1]
            self.Xu = args[2]
            self.yu = args[3]
            self.s = args[4]

  def __len__(self):
        'Denotes the total number of samples in (Xb, yb)'
        return self.yb.shape[0]

  def __getitem__(self, index):
      batch = []
      if len(self.args)==2:
        Xb = self.Xb[index,:]
        yb = self.yb[index]
        
        batch.append(Xb)
        batch.append(yb)

      elif len(self.args)==3:
        Xb = self.Xb[index,:]
        yb = self.yb[index]
        s = self.s[index]
        
        batch.append(Xb)
        batch.append(yb)
        batch.append(s)

      elif len(self.args)==5:
        Xb = self.Xb[index,:]
        yb = self.yb[index]
        Xu = self.Xu[index,:]
        yu = self.yu[index]
        s = self.s[index]

        batch.append(Xb)
        batch.append(yb)
        batch.append(Xu)
        batch.append(yu)
        batch.append(s)
      # check for weights
      if self.weights is not None:
        batch.append(self.weights[index])
    
      return batch
  

class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=10, verbose=False, delta=0, path='./results/checkpoint.pt', trace_func=print, logger=None):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
            trace_func (function): trace print function.
                            Default: print
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_auc_max = np.Inf
        self.delta = delta
        self.path = path
        # self.trace_func = trace_func
        self.logger = logger

    def __call__(self, val_auc, model):

        score = -val_auc

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_auc, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            self.logger.info(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_auc, model)
            self.counter = 0

    def save_checkpoint(self, val_auc, model):
        '''Saves model when validation acu increase.'''
        if self.verbose:
            self.logger.info(f'Validation auc increased ({self.val_auc_max:.6f} --> {val_auc:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.path)
        self.val_auc_max = val_auc


###############################################################
##### NN related things #######################################
class MLP(nn.Module):
    '''
        Define your MLP model for binary classification
    '''
    def __init__(self, input_size, hidden_sizes, output_size=1, dropout_prob=0.10):
        super(MLP, self).__init__()
        layers = []
        prev_size = input_size
        
        # Adding hidden layers
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.BatchNorm1d(hidden_size))  # Batch normalization
            layers.append(nn.Dropout(dropout_prob))    # Dropout
            prev_size = hidden_size
        
        # Output layer
        layers.append(nn.Linear(prev_size, output_size))
        layers.append(nn.Sigmoid())  # Sigmoid for binary classification
        
        # Combining all layers
        self.model = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.model(x)

class AdvNet(nn.Module):
    '''
    Define your MLP model for binary classification with adversarial training
    '''
    def __init__(self, input_size, phi_units=[100], head_sizes=[100], adv_units=[100], dropout_rate=0.10, grad_rev=False):
        super(AdvNet, self).__init__()
        self.grad_rev = grad_rev
        phi_units = [input_size] + phi_units
        head_sizes = phi_units[-1:] + head_sizes
        adv_units = phi_units[-1:] + adv_units

        self.phi = nn.ModuleList([])
        for i in range(len(phi_units)-1):
            self.phi.append(nn.Linear(phi_units[i], phi_units[i+1]))
            self.phi.append(nn.ReLU())
            self.phi.append(nn.BatchNorm1d(phi_units[i+1]))
            self.phi.append(nn.Dropout(dropout_rate))

        self.task_head = nn.ModuleList([])
        for i in range(len(head_sizes)-1):
            self.task_head.append(nn.Linear(head_sizes[i], head_sizes[i+1]))
            self.task_head.append(nn.ReLU())
            self.task_head.append(nn.BatchNorm1d(head_sizes[i+1]))
            self.task_head.append(nn.Dropout(dropout_rate))
        #output layer
        self.task_head.append(nn.Linear(head_sizes[-1], 1))
        self.task_head.append(nn.Sigmoid())

        self.censoring_head = nn.ModuleList([])
        for i in range(len(adv_units)-1):
            self.censoring_head.append(nn.Linear(adv_units[i], adv_units[i+1]))
            self.censoring_head.append(nn.ReLU())
            self.censoring_head.append(nn.BatchNorm1d(adv_units[i+1]))
            self.censoring_head.append(nn.Dropout(dropout_rate))
        #output layer
        self.censoring_head.append(nn.Linear(adv_units[-1], 1))
        self.censoring_head.append(nn.Sigmoid())

        self.bce_loss = nn.BCELoss()

    def forward(self, X, alpha=1.0):
        out_phi = X
        for layer in self.phi:
            out_phi = layer(out_phi)

        out_task = out_phi
        for layer in self.task_head:
            out_task = layer(out_task)

        if self.grad_rev:
            out_cencoring = GradientReversalLayer.apply(out_phi, alpha)
        else:
            out_cencoring = out_phi
        for layer in self.censoring_head:
            out_cencoring = layer(out_cencoring)

        return out_task.squeeze(dim=-1), out_cencoring.squeeze(dim=-1), out_phi

    def loss(self, out_task, out_cencoring, y, s, alpha_task=1.0, alpha_cencoring=1.0, disc_factor=0.0):
        loss_task = self.bce_loss(out_task[s==0], y[s==0])
        loss_cencoring = self.bce_loss(out_cencoring, s)

        loss = alpha_task * loss_task + alpha_cencoring * loss_cencoring #+ disc_factor * self._max_mean_discrepancy(X, t)

        return loss

class GradientReversalLayer(torch.autograd.Function):
    '''
      Layer for gradient reversal
    '''
    @staticmethod
    def forward(ctx, x, alpha):
        # store the context for backpropagation
        ctx.alpha = alpha
        # no-op for forward pass
        return x

    @staticmethod
    def backward(ctx, grad_output):
        output = -ctx.alpha * grad_output
        return output, None

def weighted_binary_cross_entropy(pred, y, weight=None):
    loss = torch.nn.BCELoss(reduction='none')
    if weight is None:
        return torch.mean(loss(pred, y))
    else:
        return torch.mean(weight * loss(pred, y))
    
# Create a PyTorch model wrapper for use in GridSearchCV
def create_model(model_name, params, input_size, output_size=1, grad_rev=False, device='gpu'):
    hidden_sizes = params['hidden_sizes']
    if 'head_sizes' in params:
        head_sizes = params['head_sizes']
    drop_rate = params['drop_rate']
    lr = params['lr']

    if model_name == 'MLP' or model_name == 'IPW':
        model = MLP(input_size, hidden_sizes, output_size, dropout_prob=drop_rate).to(device)
    else:
        model = AdvNet(input_size=input_size, phi_units=hidden_sizes, head_sizes=head_sizes, adv_units=head_sizes, 
                       dropout_rate=drop_rate, grad_rev=grad_rev).to(device)

    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCELoss()  # Binary Cross Entropy Loss for binary classification
    
    return model, optimizer, criterion


def get_loss(setting, model, model_name, criterion, epoch, batch, logger):
    if model_name == 'Adversarial' or model_name == 'Multitasking':
        grad_rev_multip = 1
        epoch_div = 100
        grl_alpha = grad_rev_multip * (2.0 / (1. + np.exp(-10.0 * (epoch/epoch_div if epoch<epoch_div else 1.0))) - 1)

        if len(batch)==3:
            Xb = batch[0]
            yb = batch[1]
            s = batch[2]

            out_pred, out_sens, _ = model(Xb, grl_alpha)
            loss = model.loss(out_pred, out_sens, yb, s)
            pred_y, y = out_pred.squeeze(dim=-1), yb
            pred_s, s = out_sens.squeeze(dim=-1), s
            out = [pred_y, y, pred_s, s]        
    else:
        Xb = batch[0]
        yb = batch[1]
        preds = model(Xb).squeeze(dim=-1)
        if model_name == 'IPW':
            loss = weighted_binary_cross_entropy(preds, yb, weight=batch[2])
        else:
            loss = criterion(preds, yb)
        labels = yb
        out = [preds, labels]

    return loss, out

def train_model(model, model_name, train_loader, val_loader, optimizer, criterion, early_stopping, logger, epochs=100, plot=False, wandb=None):
    train_loss_all, val_loss_all=[],[]

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        val_loss = 0
        
        for batch in train_loader:
            # Xb, yb, Xu, yu, s = batch[0], batch[1], batch[2], batch[3], batch[4]
            # skip if batch has only point
            if batch[0].shape[0]==1:
                continue

            optimizer.zero_grad()
            # print(batch[0].shape, batch[1].shape)
            loss, _ = get_loss('Train', model, model_name, criterion, epoch, batch, logger)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        train_loss = train_loss/len(train_loader)
        train_loss_all.append(train_loss)

        # Evaluate the model on the validation set
        # with torch.no_grad():
        model.eval()
        for batch in val_loader:
            loss, _ = get_loss('Train', model, model_name, criterion, epoch, batch, logger)
            val_loss += loss.item()

        val_loss = val_loss / len(val_loader)

        early_stopping(val_loss, model)
        if early_stopping.early_stop:
            # logger.info("Early stopping....")
            break

        val_loss_all.append(val_loss)
        logger.info('Epoch %d train loss: %.3f val loss: %.3f' % (epoch + 1, train_loss, val_loss))
        wandb.log({"Train loss":train_loss})
        wandb.log({"Val loss":val_loss})

    if plot:
        fig, ax = plt.subplots()
        ax.plot(train_loss, label='Training Loss')
        ax.plot(val_loss, label='Validation Loss')
        ax.set_title('Loss Curve')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.legend()
        plt.show()
    
    return model

def evaluate_model(setting, test_loader, model, model_name, criterion, logger, epoch, device, wandb):
    test_loss = 0
    predictions = torch.Tensor([]).to(device)
    preds_sensoring = torch.Tensor([]).to(device)
    labels = torch.Tensor([]).to(device)
    s = torch.Tensor([]).to(device)

    model.eval()
    # with torch.no_grad():
    for batch in test_loader:
        loss, output_all = get_loss(setting, model, model_name, criterion, epoch, batch, logger)
        predictions = torch.cat((predictions, output_all[0]))
        labels = torch.cat((labels, output_all[1])) 
        if model_name == 'Adversarial' or model_name == 'Multitasking':
            preds_sensoring = torch.cat((preds_sensoring, output_all[2]))
            s = torch.cat((s, output_all[3]))
        
        test_loss += loss.item()

    test_loss = test_loss / len(test_loader)

    if model_name == 'Adversarial' or model_name == 'Multitasking':
        if setting == 'Val':
            predictions, labels = predictions[s==0], labels[s==0]

    # calculate metrics
    wandb.log({setting+" loss":test_loss})
    logger.info('{} Epoch: {:04d} | {} Loss {:.6f}'.format(setting, epoch, setting, test_loss))

    # calculate performance metrics
    results, best_threshold = print_metrics_binary_classification(labels.data.cpu().numpy(),
                predictions.data.cpu().numpy(), setting, verbose=1, logger=logger, wandb=wandb)

    if (model_name == 'Multitasking') and not (setting == 'TB-EU:Multitasking'):
        logger.info('Check performance on sensoring prediction only to get best threshold.')
        results2, best_threshold = print_metrics_binary_classification(s.data.cpu().numpy(),
                                    preds_sensoring.data.cpu().numpy(), setting, verbose=1, logger=logger, wandb=wandb)
        results['C-'+setting+' AUROC'] = results2[setting+' AUROC']
        results['C-'+setting+' AUPRC'] = results2[setting+' AUPRC']
    
    logger.info('\n')
    
    if model_name == 'Adversarial' or model_name == 'Multitasking':
        return results, best_threshold, preds_sensoring
    else:
        return results, best_threshold

################################
def get_encoder(hidden):
    model = Sequential()
    model.add(Flatten())
    for h in hidden:
        model.add(Dense(h, activation="relu"))
        model.add(Dropout(0.10))
        # model.add(BatchNormalization())

    return model

def get_disc(hidden):
    model = Sequential()
    model.add(Flatten())
    for h in hidden[:-1]:
        model.add(Dense(h, activation="relu"))
    model.add(Dense(hidden[-1], activation="sigmoid"))

    return model

def get_mlp(hidden, lr):
    model = Sequential()
    model.add(Flatten())
    for h in hidden[:-1]:
        model.add(Dense(h, activation="relu"))
        model.add(Dropout(0.10))
        # model.add(BatchNormalization())
    model.add(Dense(hidden[-1], activation="sigmoid"))
    model.compile(optimizer=Adam(lr),
              loss=keras.losses.BinaryCrossentropy(),
              metrics=[keras.metrics.AUC()])

    return model

def grid_search_adapt(model_name, X_train, y_train, X_val, y_val, X_test, ckpt_path, param_grid, EPOCHS, logger, wandb, device):
    best_score = 0
    best_params = None
    results = {}

    for encoder in param_grid['hidden']:
        for task in param_grid['task']:
            for lambda_ in param_grid['lambda']:
                for lr in param_grid['lr']:
                    run_name = str(lambda_) +' lambda:'+ str(task) +' task:' + str(encoder) +' encoder:'
                    logger.info('\n\nRunning: '+ run_name)
                    params = {'lambda_':lambda_, 'encoder': get_encoder(encoder), 'task':get_disc(task), 'discriminator':get_disc(task)}

                    # train model
                    callback = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, min_delta=0.001)
                    model = DANN(Xt=X_test, loss="bce", optimizer=Adam(lr), metrics=["acc"], random_state=0, callbacks=callback, **params)
                    logger.info(model)
                    print
                    # wandb.watch(model)
                    model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=EPOCHS, verbose=0)
                    predictions = model.predict(X_val)
                    # evaluate model
                    res, best_threshold = print_metrics_binary_classification(y_val, predictions, model_name, verbose=1, logger=logger, wandb=wandb)
                    results[run_name] = res[model_name + ' AUROC']

                    if res[model_name + ' AUROC'] > best_score:
                        best_score = res[model_name + ' AUROC']
                        best_params = {'lr': lr, 'discriminator': task, 'task': task, 'encoder': encoder,'lambda_': lambda_}

    logger.info("Best Parameters:" + str(best_params))
    logger.info("Best Score:"+ str(best_score))

    return best_params, best_score, results

def grid_search_kmm_kliep(model_name, X_train, y_train, X_val, y_val, X_test, ckpt_path, param_grid, EPOCHS, logger, wandb, device):
    best_score = 0
    best_params = None
    results = {}

    for task in param_grid['task']:
        for lr in param_grid['lr']:
            run_name = str(task) +' task:' + str(lr) +' lr:'
            logger.info('\n\nRunning: '+ run_name)
            params = {'estimator':get_mlp(task, lr)}

            # train model
            callback = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, min_delta=0.001)
            if model_name == 'KMM':
                model = KMM(Xt=X_test, random_state=0, callbacks=callback, **params)
            elif model_name == 'KLIEP':
                model = KLIEP(Xt=X_test, random_state=0, callbacks=callback, **params)
            logger.info(model)
            # wandb.watch(model)
            model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=EPOCHS, verbose=0)
            predictions = model.predict(X_val)
            # evaluate model
            res, best_threshold = print_metrics_binary_classification(y_val, predictions, model_name, verbose=1, logger=logger, wandb=wandb)
            results[run_name] = res[model_name + ' AUROC']

            if res[model_name + ' AUROC'] > best_score:
                best_score = res[model_name + ' AUROC']
                best_params = {'lr': lr, 'task': task}

    logger.info("Best Parameters:" + str(best_params))
    logger.info("Best Score:"+ str(best_score))

    return best_params, best_score, results


def grid_search_WDGRL(model_name, X_train, y_train, X_val, y_val, X_test, ckpt_path, param_grid, EPOCHS, logger, wandb, device):
    best_score = 0
    best_params = None
    results = {}

    for encoder in param_grid['hidden']:
        for task in param_grid['task']:
            for disc in param_grid['task']:
                for lambda_ in param_grid['lambda']:
                    for lr in param_grid['lr']:
                        run_name = str(lambda_) +' lambda:'+ str(task) +' task:' + str(encoder) +' encoder:' + str(disc) +' discriminator:'
                        logger.info('\n\nRunning: '+ run_name)
                        params = {'lambda_':lambda_, 'encoder': get_encoder(encoder), 'task':get_encoder(task), 'discriminator':get_disc(disc)}
                        print(get_encoder(encoder))

                        # assert False

                        # train model
                        callback = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, min_delta=0.001)
                        model = WDGRL(Xt=X_test, optimizer=Adam(lr), metrics=["acc"], random_state=0, callbacks=callback, **params, copy=True)
                        logger.info(model)
                        # wandb.watch(model)
                        model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=EPOCHS, verbose=0)
                        predictions = model.predict(X_val)
                        # evaluate model
                        res, best_threshold = print_metrics_binary_classification(y_val, predictions, model_name, verbose=1, logger=logger, wandb=wandb)
                        results[run_name] = res[model_name + ' AUROC']

                        if res[model_name + ' AUROC'] > best_score:
                            best_score = res[model_name + ' AUROC']
                            best_params = {'lr': lr, 'discriminator': task, 'task': task, 'encoder': encoder,'lambda_': lambda_}

    logger.info("Best Parameters:" + str(best_params))
    logger.info("Best Score:"+ str(best_score))

    return best_params, best_score, results

def grid_search_MLP(model_name, loader_train, loader_val, input_size, ckpt_path, param_grid, epochs, logger, wandb, device):
    best_score = 0
    best_params = None
    results = {}

    for drop_rate in param_grid['drop_rate']:
        for hidden_sizes in param_grid['hidden_sizes']:
            for head_sizes in param_grid['head_sizes']:
                for lr in param_grid['lr']:
                    run_name = str(hidden_sizes) +' head_sizes:' + str(head_sizes) +' lr:' + str(lr) +' drop_rate:' + str(drop_rate)
                    logger.info('\n\nRunning: '+ run_name)
                    params = {'drop_rate':drop_rate, 'hidden_sizes':hidden_sizes, 'head_sizes':head_sizes, 'lr':lr}

                    if model_name == 'MLP' or model_name == 'IPW':
                        model, optimizer, criterion = create_model(model_name, params, input_size, output_size=1, device=device)
                    else:
                        if model_name =='Multitasking':
                            grad_rev = False
                        else:
                            grad_rev = True
                        model, optimizer, criterion = create_model(model_name, params, input_size, output_size=1, grad_rev=grad_rev, device=device)

                    early_stopping = EarlyStopping(patience=10, path=ckpt_path, verbose=True, logger=logger)
                    logger.info(model)
                    wandb.watch(model)

                    # train model
                    model = train_model(model, model_name, loader_train, loader_val, optimizer, criterion, early_stopping, logger, epochs=epochs, plot=False, wandb=wandb)
                    # evaluate model
                    auroc = evaluate_model('Val', loader_val, model, model_name, criterion, logger, -1, device, wandb)
                    # if model_name != 'MLP':
                    auroc = auroc[0]

                    results[run_name] = auroc['Val AUROC']

                    if auroc['Val AUROC'] > best_score:
                        best_score = auroc['Val AUROC']
                        best_params = {'hidden_sizes': hidden_sizes, 'head_sizes': head_sizes, 'drop_rate': drop_rate, 'lr': lr}

    logger.info("Best Parameters:" + str(best_params))
    logger.info("Best Score:"+ str(best_score))

    return best_params, best_score, results

###############################################################
