
for i in $(seq 2 12)
do
let jmin=$i+1
for j in $(seq $jmin 13)
do
echo $i $j
python train_graft2.py --modeldir ../../data/cifar100-models/graft-cifar100.vgg16_nobn_l12_4/sd1/epoch200/ --model1dir ../../data/cifar100-models/graft-cifar100.vgg16_nobn_l12_4/sd1/epoch200/ --exp_id graft-cifar100.vgg16_nobn_l12_4 --pnanase $i --nanase $j --statlogdir glogs/c100-vgg16-l12-4-sep-s1/graft_$i-$j.pkl --logdir mlogs  --gpu 1 --seed 0 >tmp
done
done

