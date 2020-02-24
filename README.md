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

To run a experiment for a specific random seed `[S]` for regularization parameter `1` ($\ell_{1,2}$ regularizer):

enumerate `[S]` in `{1, 2, 3, 4, 5}`

```
for i in $(seq 2 12)
do
let jmin=$i+1
for j in $(seq $jmin 13)
do
echo 'exp4' $i $j
python train_graft2.py --modeldir models/graft-cifar10.vgg16_nobn_l12_1/sd[S]/epoch180/ --model1dir models/graft-cifar10.vgg16_nobn_l12_1/sd[S]/epoch180/ --exp_id graft-cifar10.vgg16_nobn_l12_1 --pnanase $i --nanase $j --statlogdir glogs/c10-vgg16-l12-1-sep-s[S]/graft_$i-$j.pkl --logdir mlogs  --gpu 0 --seed 0 >tmp
done
done
```

### Exp5: Grafting two NN with one not fully trained

To run a experiment for a specific random seed `[S]` for regularization parameter `1` ($\ell_{1,2}$ regularizer):

Don't need to build subdir, first try one seed 

```
for k in 1 2 5 8 20 40 60
do
let t=$k-1
mkdir glogs/c10-vgg16-l12-e$t-b-s[S]/
mkdir glogs/c10-vgg16-l12-e$t-f-s[S]/
for i in $(seq 2 13)
do
python train_graft.py --modeldir models/graft-cifar10.vgg16_nobn_l12_1/sd[S]/epoch180/ --model1dir models/graft-cifar10.vgg16_nobn_l12_1/sd[S]/epoch$t/ --exp_id graft-cifar10.vgg16_nobn_l12_1 --statlogdir glogs/c10-vgg16-l12-e$t-f-s[S]/graft_$i.pkl --gpu 0 --seed 0 --nanase $i
python train_graft.py --modeldir models/graft-cifar10.vgg16_nobn_l12_1/sd[S]/epoch$t/ --model1dir models/graft-cifar10.vgg16_nobn_l12_1/sd[S]/epoch180/ --exp_id graft-cifar10.vgg16_nobn_l12_1 --statlogdir glogs/c10-vgg16-l12-e$t-b-s[S]/graft_$i.pkl --gpu 0 --seed 0 --nanase $i
done
done
```

### Exp6: Grafting two NN with same training epochs

To run a experiment for a specific random seed `[S]` for regularization parameter `1` ($\ell_{1,2}$ regularizer):

Don't need to build subdir, first try one seed 

```
for k in 1 2 5 8 10 20 30 40 50 60 90 120 150 180
do
let t=$k-1
mkdir glogs/c10-vgg16-l12-w1-sm-e$t-s[S]/
for i in $(seq 2 13)
do
python train_graft.py --modeldir models/graft-cifar10.vgg16_nobn_l12_1/sd[S]/epoch$t/ --model1dir models/graft-cifar10.vgg16_nobn_l12_1/sd[S]/epoch$t/ --exp_id graft-cifar10.vgg16_nobn_l12_1 --statlogdir glogs/c10-vgg16-l12-w1-sm-e$t-s0/graft_$i.pkl --logdir mlogs/ --gpu 0 --seed 0 --nanase $i
done
done
```
