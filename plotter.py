import numpy as np
import matplotlib.pyplot as plt

def plot_single_comparison(X, Y_tag, Y, ref):
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

    return


def plot_log(train_loss, val_loss, referenceMSE):
    plt.semilogy(train_loss, label='train')
    plt.semilogy(val_loss, label='val')
    if referenceMSE is not None:
        plt.semilogy(val_loss * 0 + referenceMSE, label='ref')
        plt.semilogy(val_loss * 0 + referenceMSE *np.pi /4, label='ref*pi/4')
        plt.semilogy(val_loss * 0 + referenceMSE / 2, label='ref/2')

    return


def plot_log_log(train_loss, val_loss, referenceMSE):
    plt.loglog(train_loss, label='train')
    plt.loglog(val_loss, label='val')
    if referenceMSE is not None:
        plt.loglog(val_loss * 0 + referenceMSE, label='ref')
        plt.loglog(val_loss * 0 + referenceMSE *np.pi /4, label='ref*pi/4')
        plt.loglog(val_loss * 0 + referenceMSE / 2, label='ref/2')

    return


def plot_runtime_error(epochNum, train_loss, val_loss, referenceMSE):
    fig = plt.figure(1)
    plt.clf()
    fig.set_facecolor('white')

    if epochNum < 50:
        plot_log(train_loss, val_loss, referenceMSE)
    else:
        plot_log_log(train_loss, val_loss, referenceMSE)

    plt.legend()
    plt.show(block=False)
    plt.pause(0.001)

    return
