
#  train teacher model
python train_KD.py -data_pkl ./data/pre_data.pkl -output_dir output -n_head 2 -n_layer 8 -warmup 128000 -lr_mul 200 \
-epoch 100 -b 16 -unmask 0.5 -T 1 -isRandMask -TorS teacher

#  train student model
python train_KD.py -data_pkl ./data/pre_data.pkl -output_dir output -n_head 2 -n_layer 8 -warmup 128000 -lr_mul 200 \
-epoch 100 -b 16 -unmask 0.5 -T 1 -isRandMask -TorS Stud1 -teacher_path model_teacher.chkpt -alpha 0.10

# Continue to distill， train the next generation of students
python train_KD.py -data_pkl ./data/pre_data.pkl -output_dir output -n_head 2 -n_layer 8 -warmup 128000 -lr_mul 200 \
-epoch 100 -b 16 -unmask 0.5 -T 1 -isRandMask -TorS Stud2 -teacher_path model_Stdu1.chkpt -alpha 0.20

python train_KD.py -data_pkl ./data/pre_data.pkl -output_dir output -n_head 2 -n_layer 8 -warmup 128000 -lr_mul 200 \
-epoch 100 -b 16 -unmask 0.5 -T 1 -isRandMask -TorS Stud3 -teacher_path model_Stdu2.chkpt -alpha 0.30

# Continue to distill，omitted here

