#!bin/bash
i=0
for w in {3..7..2}
do
for d in {3..9..2}
do
for l in {0.001,0.01}
do
for act in "tanh" "sigmoid" 
do
((i++))
python train.py --datascale 8000 --a 1 --b 1 --c 0 --d 0 --width $w --depth $d --learn_rate $l --batch_size 100 --epoch 3000 --activate $act --input_dim 1
echo $i th,done.
done
done
done
done
