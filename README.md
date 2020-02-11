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

*save models using a different name*: after the model are trained, the model is saved in the dir like `models/graft-cifar10.vgg16_nobn_l12_[W]/sd1_20-02-11-08-45`, here `sd1` represent the random seed used and `20-02-11-08-45` represent the time starting training. It is recommended that we use the following command to change the dir name for simplicity of following commands.

```
mv -r models/graft-cifar10.vgg16_nobn_l12_[W]/sd1_20-02-11-08-45 models/graft-cifar10.vgg16_nobn_l12_[W]/sd1
```

### Exp1: Grafting two NN with same $l_{1,2}$ regularization parameter but different initializations

To run a experiment for a specific random seed `[S]` and regularization parameter `[W]`:

enumerate `[S]` in `{1,2,3,4,5}` and `[W]` in `{z5, 1, 2, 4, 8, 10}` in GPU device 0

```
for i in $(seq 2 13)
do
python train_graft.py --modeldir ../../data/cifar-100-models/graft-cifar10.vgg16_nobn_l12_[W]/sd[S]/epoch180/ --model1dir ../../data/cifar-100-models/graft-cifar10.vgg16_nobn_l12_[W]/sd[S]/epoch180/ --exp_id graft-cifar10.vgg16_nobn_l12_[W] --statlogdir glogs/c10-vgg16-l12-w[W]-s[S]/graft_$i.pkl --logdir mlogs/ --gpu 0 --seed 0 --nanase $i
done
```

### Exp2: Grafting two NN with different $l_{1,2}$ regularization parameter and different initializations

To run a experiment for a specific random seed `[S]` and regularization parameter `1` and `[W]`:

enumerate `[S]` in `{1,2,3,4,5}` and `[W]` in `{4, 8, 10}` in GPU device 0

```
for i in $(seq 2 13)
do
python train_graft.py --modeldir ../../data/cifar-100-models/graft-cifar10.vgg16_nobn_l12_1/sd[S]/epoch180/ --model1dir ../../data/cifar-100-models/graft-cifar10.vgg16_nobn_l12_[W]/sd[S]/epoch180/ --exp_id graft-cifar10.vgg16_nobn_l12_1 --statlogdir glogs/c10-vgg16-l12-w[W]_1-s[S]/graft_$i.pkl --logdir mlogs/ --gpu 2 --seed 0 --nanase $i
python train_graft.py --modeldir ../../data/cifar-100-models/graft-cifar10.vgg16_nobn_l12_[W]/sd[S]/epoch180/ --model1dir ../../data/cifar-100-models/graft-cifar10.vgg16_nobn_l12_1/sd[S]/epoch180/ --exp_id graft-cifar10.vgg16_nobn_l12_[W] --statlogdir glogs/c10-vgg16-l12-w1_[W]-s[S]/graft_$i.pkl --logdir mlogs/ --gpu 0 --seed 0 --nanase $i
done
```

### Exp3: Grafting two NN with same $l_{2}$ regularization parameter but different initializations

To run a experiment for a specific random seed `[S]` and regularization parameter `[W]`:

enumerate `[S]` in `{1,2,3,4,5}` and `[W]` in `{1, 4, 16, 24, 28}` in GPU device 0

```
for i in $(seq 2 13)
do
python train_graft.py --modeldir ../../data/cifar-100-models/graft-cifar10.vgg16_nobn_l2_[W]/sd[S]/epoch180/ --model1dir ../../data/cifar-100-models/graft-cifar10.vgg16_nobn_l2_[W]/sd[S]/epoch180/ --exp_id graft-cifar10.vgg16_nobn_l2_[W] --statlogdir glogs/c10-vgg16-l2-w[W]-s[S]/graft_$i.pkl --logdir mlogs/ --gpu 0 --seed 0 --nanase $i
done
```

### Exp4: Grafting two NN with different layers

### Exp5: Grafting two NN with one not fully trained

### Exp6: Grafting two NN with same training epochs

