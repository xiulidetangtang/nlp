import torch
import matplotlib.pyplot as plt
from utils import loadfrom


def plot_loss_epoch(filename = './pics/loss_list_epoch6.pickle', ylim = [0.59, 0.64]):
    loss_epoch = loadfrom(filename)

    #print(loss_epoch)
    epochs = [i + 1 for i in range(len(loss_epoch))]

    fig, axe = plt.subplots()
    axe.plot(epochs, loss_epoch, color = 'forestgreen', label = "Train loss")
    axe.set_ylabel("Loss")
    axe.set_xlabel("Epoch")
    axe.set_title("Train loss at each epoch")

    plt.ylim(ylim)
    plt.savefig('./loss1.png')
    plt.show()

def plot_loss_valid(filename = './pics/loss_list_valid6.pickle', ylim = [0.5, 1.5]):
    loss_valid = loadfrom(filename)

    #print(loss_epoch)
    iters = [(i + 1)*2 for i in range(len(loss_valid))]

    fig, axe = plt.subplots()
    axe.plot(iters, loss_valid, color = 'darkgreen', label = "Validation loss")
    axe.set_ylabel("Loss")
    axe.set_xlabel("Iteration")
    axe.set_title("Validation loss at iterations")

    plt.ylim(ylim)
    plt.savefig('./loss2.png')
    plt.show()

if __name__ == '__main__':
    plot_loss_valid()