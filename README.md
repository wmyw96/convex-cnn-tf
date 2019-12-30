# Results of VGG16

## To Reproduce the Result of VGG training

```
python train.py --gpu 0 --seed 0 --exp_id sl.vgg16_nobn_l2
python train.py --gpu 0 --seed 0 --exp_id sl.vgg16_bn_l2
```

Top-1 Test Accuracy: 70.91% (no batchnorm), 71.94% (batchnorm)
Top-1 Train Accuracy: 99.80% (no batchnorm)

