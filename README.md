# Transformer Self-Distillation

 
This is a PyTorch implementation of the "Transformer Self-Distillation: A Robust Unsupervised Framework for Geochemical Anomaly Recognition".
	
## Hardware requirements
- two Nvidia RTX 3080Ti GPUs or higher

## Dependencies required

> + Ubuntu 16.04
> + Python 3.7
> + Pytorch 1.3.0
> + dill 0.3.3
> + tqdm 4.64.0

## Usage
### 1. Data preprocessing

   Run `process_data.py` to generate `.pkl` files.

### 2. Model Training
   Run `Run_train_KD.sh`, or directly run:
   ```bash
   python train_KD.py -data_pkl ./data/pre_data.pkl -output_dir output -n_head 2 -n_layer 8 -warmup 128000 -lr_mul 200 -epoch 100 -b 16 -unmask 0.5 -T 1 -isRandMask -TorS teacher
   python train_KD.py -data_pkl ./data/pre_data.pkl -output_dir output -n_head 2 -n_layer 8 -warmup 128000 -lr_mul 200 -epoch 100 -b 16 -unmask 0.5 -T 1 -isRandMask -TorS Stud1 -teacher_path model_teacher.chkpt -alpha 0.10  
```
   Key parameters:
    
   - `-TorS`: Set to `teacher` for initial teacher model training. If training a knowledge distillation model, `-TorS` is set to other, such as `Student1`.
    
   - `-teacher_path`: Path to previous generation model (as teacher model, for student training)
    
   Grid search:
   
   - If necessary, use the `gridsearch.sh` to find the optimal parameters (If necessary).
  

### 3. Geochemical Anomaly Detection
    
We use the trained Transformer model for the reconstruction of geochemical data and geochemical anomaly detection. 
   1. Generate geochemical data `prediction.pkl` using `process_data.py`

   2. Prepare trained model (after multiple generations of distillation), and save it as `model_best.chkpt`

   3. Run `Anomaly_detection_run.sh`,or:
   ```bash
   python anomaly_detection.py -data_pkl ./data/prediction.pkl -raw_data ./data/prediction.csv -model ./model/model_best.chkpt -output prediction
   ```

## Data
    
Required Files:
1. `pos_feature.csv`: Geochemical coordinates and element concentrations
    + Columns: X, Y, element_1, element_2, ..., element_n
2. `Au.csv`: Known mine site coordinates
    + Columns: X, Y (mine coordinates)

Put the above data into the `data` folder in csv format.

---
# Acknowledgement

- The transformer框架 implementation borrows heavily from  in some parts of the Transformer's architecture.
  - Transformer architecture implementation borrows from [attention-is-all-you-need-pytorch](https://github.com/jadore801120/attention-is-all-you-need-pytorch) in some parts of the Transformer's architecture.
  - Builds upon the author's previous work in [attention-is-all-you-need-pytorch](https://github.com/jadore801120/attention-is-all-you-need-pytorch)