#The file is used for a grid search

for SEED in 1228
do
    for Head in 2 3 4
    do
          for Layer in 8 6 4
          do
                for WARM in 4000 8000 16000
                do
                      for MUL in 0.1 2 20 200
                      do
                          for B in 8 16 32
                          do
                            for T in 1 2
                            do
                                for epoch in 80
                                do
python train_KD.py -data_pkl ./data/pre_data.pkl -output_dir output -n_layers $Layer -n_head $Head \
-warmup $WARM -lr_mul $MUL -epoch $epoch -b $B -save_mode best -T $T -loss 2 -TorS teacher -isRandMask -seed $SEED

python train_KD.py -data_pkl ./data/pre_data.pkl -output_dir output -n_layers $Layer -n_head $Head \
-warmup $WARM -lr_mul $MUL -epoch $epoch -b $B -save_mode best -T $T -loss 2 \
-teacher_path model_teacher.chkpt -alpha 0.10 -TorS Stud1 -isRandMask -seed $SEED

python train_KD.py -data_pkl ./data/pre_data.pkl -output_dir output -n_layers $Layer -n_head $Head \
-warmup $WARM -lr_mul $MUL -epoch $epoch -b $B -save_mode best -T $T -loss 2 \
-teacher_path model_Stud1.chkpt -alpha 0.20 -TorS Stud2 -isRandMask -seed $SEED

python train_KD.py -data_pkl ./data/pre_data.pkl -output_dir output -n_layers $Layer -n_head $Head \
-warmup $WARM -lr_mul $MUL -epoch $epoch -b $B -save_mode best -T $T -loss 2 \
-teacher_path model_Stud2.chkpt -alpha 0.30 -TorS Stud3 -isRandMask -seed $SEED

python train_KD.py -data_pkl ./data/pre_data.pkl -output_dir output -n_layers $Layer -n_head $Head \
-warmup $WARM -lr_mul $MUL -epoch $epoch -b $B -save_mode best -T $T -loss 2 \
-teacher_path model_Stud3.chkpt -alpha 0.40 -TorS Stud4 -isRandMask -seed $SEED

python train_KD.py -data_pkl ./data/pre_data.pkl -output_dir output -n_layers $Layer -n_head $Head \
-warmup $WARM -lr_mul $MUL -epoch $epoch -b $B -save_mode best -T $T -loss 2 \
-teacher_path model_Stud4.chkpt -alpha 0.50 -TorS Stud5 -isRandMask -seed $SEED

python train_KD.py -data_pkl ./data/pre_data.pkl -output_dir output -n_layers $Layer -n_head $Head \
-warmup $WARM -lr_mul $MUL -epoch $epoch -b $B -save_mode best -T $T -loss 2 \
-teacher_path model_Stud5.chkpt -alpha 0.60 -TorS Stud6 -isRandMask -seed $SEED

python train_KD.py -data_pkl ./data/pre_data.pkl -output_dir output -n_layers $Layer -n_head $Head \
-warmup $WARM -lr_mul $MUL -epoch $epoch -b $B -save_mode best -T $T -loss 2 \
-teacher_path model_Stud6.chkpt -alpha 0.70 -TorS Stud7 -isRandMask -seed $SEED

python train_KD.py -data_pkl ./data/pre_data.pkl -output_dir output -n_layers $Layer -n_head $Head \
-warmup $WARM -lr_mul $MUL -epoch $epoch -b $B -save_mode best -T $T -loss 2 \
-teacher_path model_Stud7.chkpt -alpha 0.80 -TorS Stud8 -isRandMask -seed $SEED

python train_KD.py -data_pkl ./data/pre_data.pkl -output_dir output -n_layers $Layer -n_head $Head \
-warmup $WARM -lr_mul $MUL -epoch $epoch -b $B -save_mode best -T $T -loss 2 \
-teacher_path model_Stud8.chkpt -alpha 0.90 -TorS Stud9 -isRandMask -seed $SEED

python train_KD.py -data_pkl ./data/pre_data.pkl -output_dir output -n_layers $Layer -n_head $Head \
-warmup $WARM -lr_mul $MUL -epoch $epoch -b $B -save_mode best -T $T -loss 2 \
-teacher_path model_Stud9.chkpt -alpha 0.90 -TorS Stud10 -isRandMask -seed $SEED

python train_KD.py -data_pkl ./data/pre_data.pkl -output_dir output -n_layers $Layer -n_head $Head \
-warmup $WARM -lr_mul $MUL -epoch $epoch -b $B -save_mode best -T $T -loss 2 \
-teacher_path model_Stud10.chkpt -alpha 0.90 -TorS Stud11 -isRandMask -seed $SEED
done
done
done
done
done
done
done
done


