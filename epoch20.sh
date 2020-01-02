for i in $(seq 2 13)
do
python train_graft.py --modeldir ../../data/cifar-100-models/graft.vgg16_nobn_l12/20-01-01-14-07/epoch180/ --model1dir ../../data/cifar-100-models/graft.vgg16_nobn_l12/20-01-01-14-07/epoch10/ --exp_id graft.vgg16_nobn_l12 --gpu 1 --seed 0 --nanase $i >epoch20_$i.log
done

