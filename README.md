# Results of VGG16

## To Reproduce the Results of Neural Network Grafting

### Preliminaries

- build dir `[MDIR]` to save trained neural networks
- build dir `[MLDIR]` to save training logs
- build dir `[GLDIR]` to save grafting los

### Train the neural network with different regularizers

- to train the neural networks with $l_{1,2}$ regularizer: where `[S]` is the random seed (could be set from {1, 2, 3, 4, 5}), `[W]` is the regularization parameter (could be set from {z5, 1, 2, 4, 8, 10})

```
python train_2nn.py --seed [S] --gpu 0 --exp_id graft-cifar10.vgg16_nobn_l12_[W] --modeldir [MDIR] --logdir [MLDIR]
```

- to train the neural network with $l_{2}$ regularizer: where `[S]` is the random seed (could be set from {1, 2, 3, 4, 5}), `[W]` is the regularization parameter (could be set from {1, 4, 16, 24, 28})

```
python train_2nn.py --seed [S] --gpu 0 --exp_id graft-cifar10.vgg16_nobn_l2_[W] --modeldir [MDIR] --logdir [MLDIR]
```
