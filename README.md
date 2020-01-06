# Results of VGG16

## To Reproduce the Result of VGG training

```
python train.py --gpu 0 --seed 0 --exp_id sl.vgg16_nobn_l2
python train.py --gpu 0 --seed 0 --exp_id sl.vgg16_bn_l2
```

Top-1 Test Accuracy: 70.91% (no batchnorm), 71.94% (batchnorm)

Top-1 Train Accuracy: 99.80% (no batchnorm)

## To Reproduce the Result of Neural Network Grafting

First train two neural networks separately with different inintializations

```
python train_2nn.py --gpu 0 --seed 0 --exp_id graft.vgg16_nobn_l12
```

then do neural network grafting:

```
python train_graft.py --modeldir ../../data/cifar-100-models/graft.vgg16_nobn_l12/19-12-30-15-36/epoch190/ --exp_id graft.vgg16_nobn_l12 --gpu 1 --seed 0
```

## To Reproduce the Results of Neural Network Hybrid


```
python train_knn.py --gpu 0 --seed 0 --exp_id hybrid.vgg16_nobn_l12
```
