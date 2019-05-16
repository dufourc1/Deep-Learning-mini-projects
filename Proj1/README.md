# MNIST classification with a twist
###### deep learning first miniproject


The purpose of this project was:

> given as input a series of [2×14×14] tensor, corresponding to pairs of 14 × 14 grayscale images, it predicts for each pair if the first digit is lesser or equal to the second.


In order to do this, we test several architectures, all described below.

## Libraries used
We used the following libraries for this project, with Python 3.6.5


 Computational:

    torch

Graphical:

    matplotlib (as plt)



## Prerequisites



The folder structure has to be the following:

    .
    ├── results                           # Results cited in report.pdf   
        ├── plots
        └── csv
    ├── src                               # Source files
        └── test.py
    ├─ run.py
    ├─ report.pdf
    └── README.md


## Models

### SiameseNet

Implemented in `SiameseNet.py`.

The SiameseNet is depicted below, with in white the inputs, in blue the layers and in red the auxiliary losses. The branch shares is the same above and below, and is just represented as 2 entities for convenience, since it is really only one model.

<p align="center">

![alt text](results/plots/Siamese.png)
</p>

The Branch takes as input a [14x14] image as input and output a [10] tensor representing the class to which the number represented in the input belongs.

The pooling layer takes a [20] tensor as input and output a [2] tensor representing the class of the pair of grayscale images.

The training with auxiliary losses is as follows, iterating at each epoch:
- first pass the input through the branch and compute the auxiliary losses, and do a backward pass on the branch.
- then do a full forward pass including the pooling layer, and a backward pass trough the pooling layer and the branch.



## Test files





## Authors

* *William Cappelletti*
* *Charles Dufour*
* *Fanny Sue*
