import os
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms

def to_img(x):
    # convert a tensor vectorized image to a numpy image of shape 28 x 28
    if torch.is_tensor(x):
        x = x.cpu().data.numpy()
    x = x.reshape([-1, 28, 28])
    return x

def plot_reconstructions_AE(model, test_loader, device='cpu'):
    """
    Plot 10 reconstructions from the test set. The top row is the original
    digits, the bottom is the decoder reconstruction.
    The middle row is the encoded vector.
    """
    # encode then decode
    data, _ = next(iter(test_loader))
    data = data.view([-1, 784]) # the size -1 is inferred from other dimensions, shape (batch size, 784)
    data.requires_grad = False
    data = data.to(device)
    true_imgs = data
    encoded_imgs = model.encode(data)
    decoded_imgs = model.decode(encoded_imgs)
    
    true_imgs = to_img(true_imgs)
    decoded_imgs = to_img(decoded_imgs)
    encoded_imgs = encoded_imgs.cpu().data.numpy()
    
    n = 10
    plt.figure(figsize=(20, 10))
    for i in range(n):
        # display original
        ax = plt.subplot(3, n, i + 1)
        plt.imshow(true_imgs[i], interpolation="nearest", 
                   vmin=0, vmax=1)
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        
        ax = plt.subplot(3, n, i + 1 + n)
        plt.imshow(encoded_imgs[np.newaxis,i,:].T, interpolation="nearest", 
                   vmin=0, vmax=1)
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        # display reconstruction
        ax = plt.subplot(3, n, i + 1 + n + n)
        plt.imshow(decoded_imgs[i], interpolation="nearest", 
                   vmin=0, vmax=1)
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    
    plt.show()
    
def plot_reconstructions_PCA(model, test_loader):
    """
    Plot 10 reconstructions from the test set. The top row is the original
    digits, the bottom is the decoder reconstruction.
    The middle row is the encoded vector.
    """
    # encode then decode
    data, _ = next(iter(test_loader))
    data = data.view([-1, 784]).numpy()
    true_imgs = data
    encoded_imgs = model.encode(data)
    decoded_imgs = model.decode(encoded_imgs)
    
    true_imgs = to_img(true_imgs)
    decoded_imgs = to_img(decoded_imgs)
    
    n = 10
    plt.figure(figsize=(20, 10))
    for i in range(n):
        # display original
        ax = plt.subplot(3, n, i + 1)
        plt.imshow(true_imgs[i,:], interpolation="nearest", 
                   vmin=0, vmax=1)
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        
        ax = plt.subplot(3, n, i + 1 + n)
        plt.imshow(encoded_imgs[np.newaxis,i,:].T, interpolation="nearest", 
                   vmin=0, vmax=1)
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        # display reconstruction
        ax = plt.subplot(3, n, i + 1 + n + n)
        plt.imshow(decoded_imgs[i,:], interpolation="nearest", 
                   vmin=0, vmax=1)
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    
    plt.show()

    
def display_digits(X, figsize=(20, 20)):
    """
    X: shape (n_i, n_j, digit_size, digit_size)
    Display an array of (n_i x n_j) images of size (digit_size x digit_size) pixels.
    """
    n_i = X.shape[0]
    n_j = X.shape[1]
    digit_size = X.shape[2]
    figure = np.zeros((digit_size * n_i, digit_size * n_j))
    
    for i in range(n_i):
        for j in range(n_j):            
            x = i * digit_size
            y = j * digit_size
            figure[x:x + digit_size, y:y + digit_size] = X[i,j,:,:]
    
    plt.figure(figsize=figsize)
    plt.imshow(figure, cmap='Greys_r')
    plt.gca().axes.get_xaxis().set_visible(False)
    plt.gca().axes.get_yaxis().set_visible(False)
    plt.show()