<div align="center">
<img src="https://user-images.githubusercontent.com/74038190/212284115-f47cd8ff-2ffb-4b04-b5bf-4d1c14c0247f.gif">
</div>

<div align="center">

# Context Aware Grounded Teacher for Source Free Object Detection

<div align="center">

[![](https://img.shields.io/badge/website-grounded_teacher-purple)]()
[![](https://img.shields.io/badge/demo-hugginface-blue)]()
[![](https://img.shields.io/badge/Arxiv-paper-red?style=plastic&logo=arxiv)]()
[![](https://img.shields.io/badge/-Linkedin-blue?style=plastic&logo=Linkedin)]() 
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)]()
</div>

<img src="https://github.com/user-attachments/assets/4d94ee45-185b-4e10-b97e-b0a36e3ff42e" width="800px">

</div>

---

## 🎯 What is Grounded_Teacher?

🔥 Check out our [website]() for more overview!

---

## 🔧 Installation

Due to dependency conflicts, this project requires **two separate environments**. Follow the instructions below to set up each environment correctly.

### 🔹 Source Train Environment
***Requirements:***
- Python >= 3.8
- PyTorch = 1.7.1 and torchvision that matches the PyTorch installation.
- Linux, CUDA >= 11.0
- Install Faster-RCNN:
   ```bash
   cd Source/lib
   python setup.py build develop
   ```
- Other requirements:
   ```bash
   cd ../
   pip install -r requirements.txt
   ```

> ⚠️ For 30XX series GPUs, set CUDA architecture:
   ```bash
   export TORCH_CUDA_ARCH_LIST="8.0"
   ```

### 🔹 Other Environment
***Requirements:***
- Python ≥ 3.6
- PyTorch ≥ 1.5 and torchvision that matches the PyTorch installation.
- Detectron2 == 0.6
- Other requirements:
   ```bash
   cd Expert
   pip install -r assets/requirements/requirements.txt
   ```

---

## 📁 Dataset Preparation

You can download the Medical dataset from [here]([https://drive.google.com/file/d/1YUbe9al-gmTnHA5tCX-rBqmMyhRaj_-y/view?usp=drive_link](https://drive.google.com/drive/u/1/folders/1NbruKnIWlKvj3VDlGQ-1l_v8H3UJbXzB)) 

City to Foggy dataset Structure:
```
└── cityscapes/
     ├── gtFine/
     |   ├── train/
     |   └── test/
     |   └── val/
     ├── leftImg8bit/
     |   ├── train/
     |   └── test/
     |   └── val/
└── cityscapes_foggy/
     ├── gtFine/
     |   ├── train/
     |   └── test/
     |   └── val/
     └── leftImg8bit/
         ├── train/
         └── test/
         └── val/
```


Other datasets must follow the Pascal VOC format structure:
```
datasets/
└── VOC_format_dataset/
    ├── Annotations/               # XML annotation files
    ├── ImageSets/
    │   └── Main/
    │       ├── train.txt          # List of training image IDs
    │       └── test.txt           # List of testing image IDs
    └── JPEGImages/                # Original images
```

---

## 🔄 Pretrained Model Setup

### 🔸 VGG-16 Backbone
For the VGG backbone, we use converted weights from [CMT](https://github.com/Shengcao-Cao/CMT/tree/main).

1. Download the pretrained weights from [Google Drive Link](https://drive.google.com/file/d/1N7TlzRwQfewiREiF_37mG3FqNm5PCNOW/view?usp=drive_link)
2. Place the weights file at: `checkpoints/vgg16_bn-6c64b313_converted.pth`

### 🔸 BoimedParse Pretrained model
1. Download the Pretrained model checkpoint from [Google Drive Link](https://drive.google.com/file/d/1az08YAvSf2KEbX429bAu2l_4lbbpHHIM/view?usp=drive_link).
2. Place the weights file at: `Expert/pretrained/biomedparse_v1.pt`

---

## 🚀 Training and Evaluation
The implementation follows a step-by-step process for domain adaptation in medical image analysis, requiring switching between environments for each step:

We are demonstrating the DDSM to RSNA.

### ▶️ Step 1: Source Training on DDSM
First, ensure the `VOC_MEDICAL` path is correctly set in your `Grounded_Teacher/Source/lib/datasets/config_dataset.py`.

Download [vgg16_caffe.pth](https://drive.google.com/file/d/1YUbe9al-gmTnHA5tCX-rBqmMyhRaj_-y/view?usp=drive_link) and then change the path in `Grounded_Teacher/Source/lib/model/utils/config.py`.

- 📋 Train on source domain:
   ```bash
     cd Source
     python trainval_pretrain_adv.py \
        --dataset voc_medical_train \
        --dataset_t voc_medical \
        --net vgg16 \
        --log_ckpt_name "DDSMSource" \
        --save_dir "output"
   ```
   
- 🏷️ Generate pseudo-labels:
   ```bash
     python psudo_label_generation.py \
        --dataset_t voc_medical \
        --net vgg16 \
        --log_ckpt_name "PseudoL_ddsm2rsna" \
        --save_dir "output" \
        --load_name "output/vgg16/DDSMSource/lg_adv_session_1_epoch_6_step_10000.pth"
   ```

- 📦 Generate a new RSNA directory containing source labels by executing the `scripts/GenerateSF.ipynb` notebook.
    ```
    cd ../scripts
    python generateSF.py
    ```

### ▶️ Step 3: Generate Expert Labels
> 🔄 Make sure to switch your environment from source_train to other
- Generate expert labels:
   ```bash
    cd ../Expert
    python prediction.py --root "<DATASET_PATH>"
   ```
   This will create the file `rsna_results.txt`.

### ▶️ Step 4: Run the Grounded Teacher

- 📝 Update configuration files:
   - Set TRAIN_LABEL to RSNA_sf with Source pseudo-labels
   - Set TRAIN_UNLABEL to RSNA
   - Set TEST to RSNA with ground truth
   - Set EXPERT_PATH to RSNA Expert pseudo-Labels

- 🏃 Run training:
   ```bash
   python train_net.py \
     --num-gpus 1 \
     --config configs/faster_rcnn_VGG_cross_city.yaml \
     OUTPUT_DIR output/ddsm2rsna
   ```
- 📈 Calculate Froc:
   ```bash
   python eval.py --setting ddsm2rsna --root output/ddsm2rsna
   ```

---

## 📊 Results and Model Parameters

<div align="center">
We conduct all experiments with batch size 4, on 4 NVIDIA A100 GPUs.
</div>

<div align="center">
   
### 🔄 rsna2inbreast: RSNA-BSD1K → INBreast

| backbone | training stage     | R@0.3 | logs & weights         |
|:--------:|:-------------------:|:-----:|:-------------------------:|
| vgg16 | source_only         | 0.31 | [logs](https://drive.google.com/file/d/1M4o5p1TrRL2hcRSGzqQ7iwcDQ2-RWHnu/view?usp=sharing) & [weights](https://drive.google.com/file/d/1ptSa8iqZ63OrlqXdeXE-n1-DexoW3KC7/view?usp=sharing) |
| vgg16 | cross_domain_mae    | 0.28  | [logs](https://drive.google.com/file/d/1VF6xZ7yeT9DHUWwy9e2LYRDamBZn5qRf/view?usp=sharing) & [weights](https://drive.google.com/file/d/1PsW8m8n-JsJj94B5uRR3px2JxPL8p69c/view?usp=sharing) |

</div>

---

<div align="center">

### 🔄 ddsm2rsna: DDSM → RSNA-BSD1K

| backbone | training stage     | R@0.3 | logs & weights         |
|:--------:|:-------------------:|:-----:|:-------------------------:|
| vgg16 | source_only         | 0.30  | [logs](https://drive.google.com/file/d/1i2-aiUKIiog2VPr8gwVj46XFDRAFJRnZ/view?usp=sharing) & [weights](https://drive.google.com/file/d/1tGrr54qquBGDt78JDJzm3e76iJsKkU4_/view?usp=sharing) |
| vgg16 | cross_domain_mae    | 0.43  | [logs](https://drive.google.com/file/d/1iANgehqF6tY648Au5ViaJ0NxNXjaLkdh/view?usp=sharing) & [weights](https://drive.google.com/file/d/1uHortZqQpTbVI7GGX-C6CnEIBPsaYPRZ/view?usp=sharing) |

</div>

---

<div align="center">

### 🔄 ddsm2inbreast: DDSM → INBreast

| backbone | training stage     | R@0.3 | logs & weights         |
|:--------:|:-------------------:|:-----:|:-------------------------:|
| vgg16 | source_only         | 0.24  | [logs](https://drive.google.com/file/d/1i2-aiUKIiog2VPr8gwVj46XFDRAFJRnZ/view?usp=sharing) & [weights](https://drive.google.com/file/d/1fUJisx8yPb53QeurStZzjAyJeM_k_wKG/view?usp=sharing) |
| vgg16 | cross_domain_mae    | 0.43  | [logs](https://drive.google.com/file/d/1QkpZz82-MMenFHwQDb55d5rvt0Cq2IVL/view?usp=sharing) & [weights](https://drive.google.com/file/d/103clW1PjrACI_QHRzxiDZdfKZnSciLgt/view?usp=sharing) |

</div>

---


## 📚 Citation
<div align="left">
  <p> If you find this work useful, please cite our paper </p>
</div>

```bibtex
@inproceedings{groundedteacher2025,
  title={Context Aware Grounded Teacher for Source Free Object Detection},
  author={},
  booktitle={},
  year={2025}
}
```

---

## 🙏 Acknowledgments

Our implementation builds upon these excellent works:
- [CAT](https://github.com/mkLabs/CAT): Exploiting Inter-Class Dynamics for Domain Adaptive Object Detection
- [AASFOD](https://github.com/aasfod/official): Source-Free Object Detection architecture
- [BiomedParse](https://github.com/microsoft/BiomedParse): A Foundation Model for Biomedical Objects

<div align="center">
<img src="https://user-images.githubusercontent.com/74038190/212284115-f47cd8ff-2ffb-4b04-b5bf-4d1c14c0247f.gif">
</div>
