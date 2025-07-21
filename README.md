
# EM-Derived Self-Distillation for Geochemical Anomaly Recognition

This is a PyTorch implementation of the paper:

**"Expectation-Maximization-Derived Self-distillation Meets Transformer: A Robust Unsupervised Deep Learning Approach for Geochemical Anomaly Recognition"**

This framework formulates geochemical background reconstruction as an unsupervised learning task guided by self-distillation and inspired by the Expectation-Maximization (EM) algorithm. The model progressively improves anomaly recognition performance through multiple generations of knowledge transfer.

---

## Hardware Requirements

* Two Nvidia RTX 3080Ti GPUs or higher recommended

---

## Dependencies

> * Ubuntu 16.04
> * Python 3.7
> * PyTorch 1.3.0
> * dill 0.3.3
> * tqdm 4.64.0

---

## Usage

### 1. Data Preprocessing

Prepare geochemical input data and run:

```bash
python process_data.py
```

This will generate required `.pkl` files for training and evaluation.

---

### 2. Unsupervised Model Training via Self-Distillation

Run the main script:

```bash
bash Run_train_KD.sh
```

Or execute step-by-step:

```bash
# Step 1: Train the initial model (acts as the first-generation teacher)
python train_KD.py -data_pkl ./data/pre_data.pkl -output_dir output \
-n_head 2 -n_layer 8 -warmup 128000 -lr_mul 200 -epoch 100 -b 16 \
-unmask 0.5 -T 1 -isRandMask -TorS teacher

# Step 2: Train the first student model with self-distillation 
python train_KD.py -data_pkl ./data/pre_data.pkl -output_dir output \
-n_head 2 -n_layer 8 -warmup 128000 -lr_mul 200 -epoch 100 -b 16 \
-unmask 0.5 -T 1 -isRandMask -TorS Stud1 -teacher_path model_teacher.chkpt -alpha 0.10

# Step 3: Train the second student model using the first student as teacher with self-distillation
python train_KD.py -data_pkl ./data/pre_data.pkl -output_dir output \
-n_head 2 -n_layer 8 -warmup 128000 -lr_mul 200 -epoch 100 -b 16 \
-unmask 0.5 -T 1 -isRandMask -TorS Stud2 -teacher_path model_Stud1.chkpt -alpha 0.20

# Step 4: Train the third student model using the second student as teacher
python train_KD.py -data_pkl ./data/pre_data.pkl -output_dir output \
-n_head 2 -n_layer 8 -warmup 128000 -lr_mul 200 -epoch 100 -b 16 \
-unmask 0.5 -T 1 -isRandMask -TorS Stud3 -teacher_path model_Stud2.chkpt -alpha 0.30

# ...
# Continue this iterative distillation process for further generations as needed.
```

#### Key Arguments

* `-TorS`: Set to `teacher` for training the first model. For subsequent generations, use identifiers like `Stud1`, `Stud2`, etc.
* `-teacher_path`: Path to the teacher model used for distillation.
* `-alpha`: Balancing weight between student loss and teacher guidance.
* `-isRandMask`: Whether to apply random masking (EM-inspired partial input strategy).

#### Note:

Each new student benefits from both the labeled guidance (implicitly via prior generation) and the EM-inspired reconstruction loss.

You can continue the distillation for as many generations as needed, adjusting the -alpha parameter progressively to emphasize teacher guidance.


#### Hyperparameter Search

Use `gridsearch.sh` for tuning parameters if needed.

---

### 3. Geochemical Anomaly Detection

After training, run anomaly detection using the final distilled model:

```bash
# Ensure you have a prepared .pkl and a trained model
python anomaly_detection.py -data_pkl ./data/prediction.pkl -raw_data ./data/prediction.csv -model ./model/model_best.chkpt -output prediction
```

---

## Input Data Format

Place the following files in the `data/` folder:

1. **`pos_feature.csv`**: Geochemical sample locations and attributes

   * Columns: `X`, `Y`, `element_1`, `element_2`, ..., `element_n`
2. **`Au.csv`**: Known mineralized sites (for evaluation only)

   * Columns: `X`, `Y`

---

## Citation

If you find this code useful, please cite our paper (BibTeX will be added upon publication).

---

## Acknowledgments

* Some components of the model backbone are adapted from [attention-is-all-you-need-pytorch](https://github.com/jadore801120/attention-is-all-you-need-pytorch)
* Builds upon earlier work: [Transformer-For-Geochemical-Anomaly-Detection](https://github.com/ysyBrenda/Transformer-For-Geochemical-Anomaly-Detection)

---

如果你还希望添加模型结构图、可视化结果图或论文链接（arXiv/DOI），我也可以帮你更新。是否需要添加这些内容？
