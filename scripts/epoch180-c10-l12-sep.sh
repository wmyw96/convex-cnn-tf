for i in $(seq 2 12)
do
let jmin=$i+1
for j in $(seq $jmin 13)
do
echo $i $j
python train_graft2.py --modeldir ../../data/cifar-100-models/graft-cifar10.vgg16_nobn_l12_10/20-02-02-13-22/epoch180/ --model1dir ../../data/cifar-100-models/graft-cifar10.vgg16_nobn_l12_10/20-02-02-13-22/epoch180/ --exp_id graft-cifar10.vgg16_nobn_l12_1 --gpu 1 --seed 0 --pnanase $i --nanase $j >log/c10_vgg16_l12_sep_$i,$j.log
done
done


