import numpy as np
import matplotlib.pyplot as plt


def plot_single_comparison(X, Y_tag, Y, ref):
    """
    Plots a single comparison between the prediction and target
    :param X: np.array, the masked input image
    :param Y_tag: np.array, the predicted image
    :param Y: np.array, the target image (or the unmasked input image)
    :param ref: np.array, the reference image
    :return:
    """
    
    fig = plt.figure(figsize=(15, 15))
    fig.set_facecolor('white')
    plt.subplot(2, 2, 1)
    plt.imshow(X, vmin=.1, vmax=.9)
    plt.title('input')
    plt.colorbar()

    plt.subplot(2, 2, 2)
    plt.imshow(Y_tag, vmin=.1, vmax=.9)
    plt.title('prediction')
    plt.colorbar()

    plt.subplot(2, 2, 4)
    plt.imshow(Y, vmin=.1, vmax=.9)
    plt.title('target')
    plt.colorbar()

    plt.subplot(2, 2, 3)
    plt.imshow(Y - ref, vmin=-.1, vmax=.1)
    plt.title('target-prediction')
    plt.colorbar()
    plt.show()

    return


def plot_log(train_loss, val_loss, referenceMSE):
    """
    Plots the log curve of the loss function
    :param train_loss: array or array-like object, training loss
    :param val_loss: array or array-like object, validation loss
    :param referenceMSE: array or array-like object, containing the reference MSE for the experiment
    :return:
    """
    
    plt.semilogy(train_loss, label='train')
    plt.semilogy(val_loss, label='val')
    if referenceMSE is not None:
        plt.semilogy(val_loss * 0 + referenceMSE, label='ref')
        plt.semilogy(val_loss * 0 + referenceMSE * np.pi / 4, label='ref*pi/4')
        plt.semilogy(val_loss * 0 + referenceMSE / 2, label='ref/2')
    plt.show()

    return


def plot_log_log(train_loss, val_loss, referenceMSE):
    """
    Plots the log-log curve of the loss function
    :param train_loss: array or array-like object, training loss
    :param val_loss: array or array-like object, validation loss
    :param referenceMSE: array or array-like object, containing the reference MSE for the experiment
    :return:
    """
    
    plt.loglog(train_loss, label='train')
    plt.loglog(val_loss, label='val')
    if referenceMSE is not None:
        plt.loglog(val_loss * 0 + referenceMSE, label='ref')
        plt.loglog(val_loss * 0 + referenceMSE * np.pi / 4, label='ref*pi/4')
        plt.loglog(val_loss * 0 + referenceMSE / 2, label='ref/2')
    plt.show()

    return


def plot_runtime_error(epochNum, train_loss, val_loss, referenceMSE):
    """
    Plots the continuous loss for the training loop
    :param epochNum: int, the current epoch number
    :param train_loss: array or array-like object, training loss
    :param val_loss: array or array-like object, validation loss
    :param referenceMSE: array or array-like object, containing the reference MSE for the experiment
    :return:
    """
    
    fig = plt.figure(1)
    plt.clf()
    fig.set_facecolor('white')

    if epochNum < 50:
        plot_log(train_loss, val_loss, referenceMSE)
    else:
        plot_log_log(train_loss, val_loss, referenceMSE)

    plt.legend()
    plt.show()
    plt.pause(0.001)

    return
