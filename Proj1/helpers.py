import torch
import matplotlib.pyplot as plt

def show(n):
    plt.figure()
    plt.imshow(input_train[n,0])
    plt.show()
    plt.imshow(input_train[n,1])
    plt.show()
    print('classes :\t 0: {};\t 1: {}; label: {}'.format(*classes_train[n], target_train[n]))
