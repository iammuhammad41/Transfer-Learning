# Transfer Learning with PyTorch
Starting with a model pretrained on a large dataset (e.g., ImageNet, COCO) and fine-tuning it on a smaller, task-specific dataset. This technique is crucial when labeled data is limited and helps improve model performance by leveraging learned features from large, general datasets.

A simple end‑to‑end example demonstrating how to fine‑tune pretrained vision models on your own datasets:

* **Image Classification** using a ResNet‑50 pretrained on ImageNet
* **Object Detection** using a Faster R‑CNN (ResNet‑50‑FPN backbone) pretrained on COCO



## 🔍 Project Structure

```
.
├── transfer_learning.py   # main training script
├── requirements.txt       # Python dependencies
└── README.md              # this file
```


## 📦 Setup

1. **Clone & install**

   ```bash
   git clone <repo_url>
   cd Neural-Transfer-Learning
   pip install -r requirements.txt
   ```

2. **Download datasets** (make sure these are mounted under `/kaggle/input` or adjust paths):

   * **ImageNet Val**

     ```
     /kaggle/input/imagenet-object-localization-challenge/
         ├── ILSVRC/Data/CLS-LOC/val    # images
         ├── LOC_val_solution.csv       # labels
         └── LOC_synset_mapping.txt     # synset→name map
     ```
   * **COCO2017**

     ```
     /kaggle/input/coco2017/
         ├── train2017/                 # train images
         ├── val2017/                   # val images
         └── annotations/
             ├── instances_train2017.json
             └── instances_val2017.json
     ```


## 🚀 Usage

```bash
python transfer_learning.py
```

By default this will:

1. **Fine‑tune ResNet50** on the ImageNet validation set (3 epochs, classification)
2. **Fine‑tune Faster R‑CNN** on COCO2017 (2 epochs, detection)

Modify the top‐level `train_imagenet_classification` / `train_coco_detection` calls for different hyperparameters (epochs, batch size, learning rate, etc.).


## 🛠️ Requirements

```txt
torch
torchvision
pycocotools
pandas
Pillow
```

Install via:

```bash
pip install torch torchvision pycocotools pandas pillow
```


## 📖 About Transfer Learning

* **Why?**
  When you have **limited labeled data**, starting from a model pretrained on a large, generic dataset (e.g., ImageNet or COCO) lets you leverage its learned representations. Fine‑tuning adapts those features to your specific task—often yielding better accuracy and faster convergence than training from scratch.

* **How?**

  1. Load a pretrained backbone+head.
  2. Replace the final classification/regression layers to match your target labels.
  3. Train (often with a lower learning rate) on your smaller dataset.

