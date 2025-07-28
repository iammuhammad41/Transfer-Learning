import os
from PIL import Image
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms, models
from torchvision.datasets import CocoDetection
from pycocotools.coco import COCO

#######################################
# IMAGE CLASSIFICATION: ImageNet 
#######################################

class ImageNetValDataset(Dataset):
    def __init__(self, images_dir, labels_csv, synset_map_txt, transform=None):
        # load labels
        df = pd.read_csv(labels_csv, names=['ImageId','Label'], header=0)
        df['Label'] = df['Label'].str.split().str[0]
        self.df = df.set_index('ImageId')
        # load synset→human name (optional, for printing)
        self.syn2name = {}
        for line in open(synset_map_txt):
            sp = line.strip().split(' ',1)
            if len(sp)==2:
                syn,name = sp
                self.syn2name[syn] = name.split(',')[0]
        self.images_dir = images_dir
        self.transform = transform
        
        # build list of (filename, class_idx)
        self.synsets = sorted(self.df['Label'].unique())
        self.syn2idx = {s:i for i,s in enumerate(self.synsets)}
        self.samples = []
        for fname in os.listdir(images_dir):
            img_id,ext = fname.split('.')
            if img_id in self.df.index:
                syn = self.df.loc[img_id,'Label']
                self.samples.append((fname, self.syn2idx[syn]))
                
    def __len__(self):
        return len(self.samples)
    def __getitem__(self, i):
        fname, label = self.samples[i]
        img = Image.open(os.path.join(self.images_dir,fname)).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, label

def train_imagenet_classification(
    images_dir="/kaggle/input/imagenet-object-localization-challenge/ILSVRC/Data/CLS-LOC/val",
    labels_csv="/kaggle/input/imagenet-object-localization-challenge/LOC_val_solution.csv",
    synmap_txt="/kaggle/input/imagenet-object-localization-challenge/LOC_synset_mapping.txt",
    epochs=5, batch_size=64, lr=1e-4
):
    # transforms matching ImageNet
    tfm = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
          mean=[0.485,0.456,0.406],
          std=[0.229,0.224,0.225]
        )
    ])
    ds = ImageNetValDataset(images_dir, labels_csv, synmap_txt, transform=tfm)
    # split 80/20 train/val
    n = len(ds)
    n_train = int(0.8*n)
    train_ds, val_ds = random_split(ds, [n_train, n-n_train])
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=4)

    # load pretrained ResNet50
    model = models.resnet50(pretrained=True)
    # replace fc
    n_classes = len(ds.synsets)
    model.fc = nn.Linear(model.fc.in_features, n_classes)
    model = model.cuda()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        # ——— train ———
        model.train()
        running_loss = 0.0
        for imgs, labels in train_loader:
            imgs, labels = imgs.cuda(), labels.cuda()
            optimizer.zero_grad()
            logits = model(imgs)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()*imgs.size(0)
        print(f"[Epoch {epoch+1}/{epochs}] Train Loss: {running_loss/len(train_loader.dataset):.4f}")

        # ——— validate ———
        model.eval()
        correct = 0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.cuda(), labels.cuda()
                preds = model(imgs).argmax(dim=1)
                correct += (preds==labels).sum().item()
        acc = correct/len(val_loader.dataset)
        print(f"           Val Acc : {acc:.4f}")

    # save
    torch.save(model.state_dict(), "resnet50_imagenet_finetuned.pth")
    print("Saved finetuned classification model.")

#######################################
# OBJECT DETECTION: COCO2017 
#######################################

def collate_fn(batch):
    return tuple(zip(*batch))

def train_coco_detection(
    base_dir="/kaggle/input/coco2017",
    epochs=3,
    batch_size=4,
    lr=1e-4
):
    train_ann = os.path.join(base_dir,"annotations","instances_train2017.json")
    val_ann   = os.path.join(base_dir,"annotations","instances_val2017.json")
    train_img = os.path.join(base_dir,"train2017")
    val_img   = os.path.join(base_dir,"val2017")

    # transforms: to tensor only (Faster R‑CNN expects [0,1] RGB)
    transform = transforms.Compose([transforms.ToTensor()])

    train_ds = CocoDetection(train_img, train_ann, transform=transform)
    val_ds   = CocoDetection(val_img,   val_ann,   transform=transform)
    train_loader = DataLoader(train_ds,
                              batch_size=batch_size,
                              shuffle=True,
                              num_workers=4,
                              collate_fn=collate_fn)
    val_loader   = DataLoader(val_ds,
                              batch_size=batch_size,
                              shuffle=False,
                              num_workers=4,
                              collate_fn=collate_fn)

    # load pretrained Faster R‑CNN
    model = models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    model = model.cuda()

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.SGD(params, lr=lr, momentum=0.9, weight_decay=1e-4)

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        for imgs, targets in train_loader:
            imgs = list(img.cuda() for img in imgs)
            # each target is a list of dicts: coco returns category_id, bbox... rename to 'labels','boxes'
            formatted_tgt = []
            for t in targets:
                boxes = torch.tensor([obj['bbox'] for obj in t], device='cuda')
                # coco bbox = [x,y,w,h] → convert to [x1,y1,x2,y2]
                boxes[:,2:] = boxes[:,:2] + boxes[:,2:]
                labels = torch.tensor([obj['category_id'] for obj in t], device='cuda')
                formatted_tgt.append({'boxes': boxes, 'labels': labels})
            loss_dict = model(imgs, formatted_tgt)
            loss = sum(loss_dict.values())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"[Epoch {epoch+1}/{epochs}] Detection train loss: {total_loss/len(train_loader):.4f}")

        # (optionally evaluate mAP on val set here)

    torch.save(model.state_dict(), "fasterrcnn_coco_finetuned.pth")
    print("Saved finetuned detection model.")

#######################################
# RUN BOTH
#######################################

if __name__=="__main__":
    torch.backends.cudnn.benchmark = True
    print("** Fine‑tuning ResNet50 on ImageNet‑Val subset **")
    train_imagenet_classification(epochs=3, batch_size=128, lr=1e-4)

    print("\n** Fine‑tuning Faster R‑CNN on COCO2017 **")
    train_coco_detection(epochs=2, batch_size=2, lr=5e-5)
