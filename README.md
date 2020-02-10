# Results of VGG16

## To Reproduce the Results of Neural Network Grafting

### Preliminaries

- build dir `models` to save trained neural networks
- build dir `mlogs` to save training logs
- build dir `glogs` to save grafting los

### Train the neural network with different regularizers

- to train the neural networks with $l_{1,2}$ regularizer: where `[S]` is the random seed (could be set from {1, 2, 3, 4, 5}), `[W]` is the regularization parameter (could be set from {z5, 1, 2, 4, 8, 10})

```
python train_2nn.py --seed [S] --gpu 0 --exp_id graft-cifar10.vgg16_nobn_l12_[W] --modeldir models/ --logdir mlogs/
```

- to train the neural network with $l_{2}$ regularizer: where `[S]` is the random seed (could be set from {1, 2, 3, 4, 5}), `[W]` is the regularization parameter (could be set from {1, 4, 16, 24, 28})

```
python train_2nn.py --seed [S] --gpu 0 --exp_id graft-cifar10.vgg16_nobn_l2_[W] --modeldir models/ --logdir mlogs/
```

### Exp1: Grafting two NN with same $l_{1,2}$ regularization parameter but different initializations

### Exp2: Grafting two NN with different $l_{1,2}$ regularization parameter and different initializations

### Exp3: Grafting two NN with same $l_{2}$ regularization parameter but different initializations

### Exp4: Grafting two NN with different layers

### Exp5: Grafting two NN with one not fully trained

### Exp6: Grafting two NN with same training epochs

