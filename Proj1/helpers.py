import torch
import matplotlib.pyplot as plt
import sys

def show(n):
    plt.figure()
    plt.imshow(input_train[n,0])
    plt.show()
    plt.imshow(input_train[n,1])
    plt.show()
    print('classes :\t 0: {};\t 1: {}; label: {}'.format(*classes_train[n], target_train[n]))

def update_progress(progress,message=""):
    # update_progress() : Displays or updates a console progress bar
    ## Accepts a float between 0 and 1. Any int will be converted to a float.
    ## A value under 0 represents a 'halt'.
    ## A value at 1 or bigger represents 100%
    barLength = 20 # Modify this to change the length of the progress bar
    status = ""
    if isinstance(progress, int):
        progress = float(progress)
    if not isinstance(progress, float):
        progress = 0
        status = "error: progress var must be float\r\n"
    if progress < 0:
        progress = 0
        status = "Halt...\r\n"
    if progress >= 1:
        progress = 1
        status = "Done...\r\n"
    block = int(round(barLength*progress))
    text = "\rLearning : [{0}] {1}% {2} {3}".format( "="*block + " "*(barLength-block), round(progress*100,1),
                                                    status,message)
    sys.stdout.write(text)
sys.stdout.flush()
