import os
import json
import traceback
import argparse
import time
import pandas as pd
import gc
import matplotlib.pyplot as plt


from typing import Optional
from PIL import Image

from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split

import torchvision.transforms.v2 as v2
import torchvision.models as models
from torchvision import transforms

import torch

from visualization import plot_history
from cnn_basline import SimpleCNN, BaselineCNN


def get_image_metadata(
    target_filename: str, 
    dataset: str, 
    annotation_path: str,
    split_name: str = 'train', 
    full_image_path: str = None
    ):
    
    allowed_categories = []
    match dataset:
        case "extra_1_.coco": allowed_categories = list(range(5))
        case "extra_2_.coco": allowed_categories = list(range(17))
        case "extra_3_.coco": allowed_categories = list(range(2, 13))
        case "main_.coco":    allowed_categories = [0, 1, 2, 4, 5]

    if not os.path.exists(annotation_path):
        return None

    with open(annotation_path, 'r') as f:
        coco_data = json.load(f)

    image_id = None
    for img in coco_data['images']:
        if img['file_name'].startswith(target_filename) or \
           img.get('extra', {}).get('name') == target_filename:
            image_id = img['id']
            break

    if image_id is None:
        return None

    image_annotations = [
        ann for ann in coco_data['annotations']
        if ann['image_id'] == image_id and ann['category_id'] in allowed_categories
    ]

    return {
        "filename":    target_filename,
        "full_path":   full_image_path or os.path.join('./', target_filename),
        "dataset":     dataset,
        "split":       split_name,
        "total_balls": len(image_annotations),
        "ball_list":   [ann['category_id'] for ann in image_annotations],
        "raw_bboxes":  [ann['bbox'] for ann in image_annotations],
    }
    

                        
class ApplyTransform(torch.utils.data.Dataset):
    def __init__(self, subset, transform, inference_only=False):
        self.subset = subset
        self.transform = transform
        self._inference_only = inference_only
    def __getitem__(self, index):
        if self._inference_only:
            x = self.subset[index]
            return self.transform(x)
        
        x, y = self.subset[index]
        return self.transform(x), y
    def __len__(self):
        return len(self.subset)
        

class Pool_table_Dataset(Dataset):
    """Simple dataset wrapper expecting a metadata list with all the information necessary
    """

    def __init__(self, metadata:list[dict], transform: Optional[transforms.Compose] = None, inference_only = False):
        self._meta_data = metadata
        self._inference_only = inference_only
        self.samples = []  # list of (path, count)
        
        for d in  self._meta_data:
            if not inference_only:
                self.samples.append((os.path.join('./', d['full_path']), d['total_balls']))
            else:
                self.samples.append(os.path.join('./', d))

        self.train_transform = transform or v2.Compose([
            v2.ToImage(),
            # force to uint8 here to satisfy the JPEG transform
            v2.ToDtype(torch.uint8, scale=True), 
            
            v2.Resize((416, 416), transforms.InterpolationMode.BILINEAR),

            # Augmentations now have guaranteed uint8 inputs
            v2.RandomHorizontalFlip(p=0.5),
            v2.RandomApply(transforms=[v2.GaussianBlur(kernel_size=3)], p=0.3),
            v2.JPEG((50, 100)),
            
            # convert back to float for the actual neural network
            v2.ToDtype(torch.float32, scale=True),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])

        # validation/test set can't be augmented but does need to be resized identically 
        self.val_transform = transform or v2.Compose([
            v2.ToImage(),
            v2.Resize((416, 416), transforms.InterpolationMode.BILINEAR),
            v2.ToDtype(torch.float32, scale=True),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        if not self._inference_only:
            path, count = self.samples[idx]
        else:
            path = self.samples[idx]

        image = Image.open(path).convert("RGB")

        if not self._inference_only:
            return image, count
        else:
            return image
        


def build_regression_model(backbone='resnet18', pretrained=True):
    if backbone == 'resnet18':
        model = models.resnet18(weights='DEFAULT' if pretrained else None)
        model.fc = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(model.fc.in_features, 1)
        )
    elif backbone == 'efficientnet_b0':
        model = models.efficientnet_b0(weights='DEFAULT' if pretrained else None)
        in_features = model.classifier[1].in_features
        model.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(in_features, 1)
        )
    elif backbone == 'baseline_1':
        model = SimpleCNN()
    elif backbone == 'baseline_2':
        model = BaselineCNN()
    return model

def make_preds(model, dataloader, device):
    model.eval()
    all_preds = []
    with torch.no_grad():
        for images in dataloader:
            images = images.to(device)
            outputs = model(images)
            preds = outputs.squeeze(1).round().cpu().numpy()
            all_preds.extend(preds.tolist())

    return all_preds

def evaluate_model(model, dataloader, device):
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            outputs = model(images)
            preds = outputs.squeeze(1).round().cpu().numpy()
            all_preds.extend(preds.tolist())
            all_labels.extend(labels.numpy().tolist())
    mae = mean_absolute_error(all_labels, all_preds)
    mse = mean_squared_error(all_labels, all_preds)

    # Filter out zero-label samples before computing MAPE, so it doesn't explode to meaningless values
    non_zero = [(y, p) for y, p in zip(all_labels, all_preds) if y != 0]
    if non_zero:
        ys, ps = zip(*non_zero)
        mape = mean_absolute_percentage_error(list(ys), list(ps))
    else:
        mape = float('nan')  # undefined if all labels are zero

    excluded = len(all_labels) - len(non_zero)
    print(f"MAPE computed on {len(non_zero)} samples ({excluded} zero-label samples excluded)")
    return mae, mape, mse, all_preds


def parse_args():
    parser = argparse.ArgumentParser(description="Ball counting training pipeline")
    parser.add_argument("--data-dir", type=str, required=True, help="Root directory of dataset")
    parser.add_argument("--output-dir", type=str, required=True, help="Directory where models are saved")
    parser.add_argument("--backbone", type=str, default="efficientnet_b0", help="Architecture of the cnn model")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--pretrained", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--train", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--inference-only", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--no-cuda", action="store_true")
    return parser.parse_args()



if __name__ == "__main__":

    args = parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    DATA_PATH = args.data_dir

    if not args.inference_only:
        #------------------- METADATA EXTRACTING BLOCK -------------------
        datasets = os.listdir(DATA_PATH)
        total_metadata = []
        for dataset in datasets:
            print("="*90)
            print(f"For dataset: {dataset}")
            print("="*90)
            splits = os.listdir(os.path.join(DATA_PATH, dataset))
            print(splits)
            for split in splits:
                if not split.endswith(".txt"):
                    img_pths = os.listdir(os.path.join(DATA_PATH, dataset, split))
                    annotation_path = os.path.join(DATA_PATH, dataset, split, "_annotations.coco.json")
                    for img_pth in img_pths:
                        if not img_pth.endswith(".json"):
                            try:
                                full_path = os.path.join(DATA_PATH, dataset, split, img_pth)
                                meta = get_image_metadata(
                                    target_filename=img_pth,
                                    dataset=dataset,
                                    annotation_path=annotation_path,
                                    split_name=split,
                                    full_image_path=full_path,
                                )
                                if isinstance(meta, dict):
                                    total_metadata.append(meta)
                            except Exception as e:
                                print(("="*10+"ERROR")*5+"="*10)
                                print(traceback.format_exc())
                                print(("="*10+"ERROR")*5+"="*10)
                                break
        #------------------- METADATA EXTRACTING BLOCK -------------------
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # fixed seed for the split of the dataset
        generator = torch.Generator().manual_seed(42)

        print(f"Running on device: {device}")

        # dataset and splits, mantaining main_.coco test is in the final test plit
        main_test_meta = [m for m in total_metadata if m['split'] == 'test']
        trainval_meta  = [m for m in total_metadata if not (m['split'] == 'test')]


        print(f"Test set: {len(main_test_meta)} samples")
        print(f"Train/val:            {len(trainval_meta)} samples")

        trainval_dataset = Pool_table_Dataset(trainval_meta)
        test_dataset     = Pool_table_Dataset(main_test_meta)

        total_train_val = len(trainval_dataset)
        n_train = int(total_train_val * 0.9)
        n_val   = total_train_val - n_train

        train_set, val_set = random_split(
            trainval_dataset, [n_train, n_val], generator=generator
        )

        train_loader = DataLoader(
            ApplyTransform(train_set, trainval_dataset.train_transform),
            batch_size=args.batch_size, shuffle=True, num_workers=4
        )
        val_loader = DataLoader(
            ApplyTransform(val_set, trainval_dataset.val_transform),
            batch_size=args.batch_size, shuffle=False, num_workers=4
        )
        test_loader = DataLoader(
            ApplyTransform(test_dataset, test_dataset.val_transform),
            batch_size=args.batch_size, shuffle=False, num_workers=4
        )

        # clear memory before training
        del total_metadata, trainval_dataset, test_dataset, main_test_meta, trainval_meta
        gc.collect()
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Running on device: {device}")
        print(f"Running inference only!!!")

        with open(DATA_PATH, 'r') as f:
            paths = json.load(f)
        ims_path = paths['image_path']

        test_dataset = Pool_table_Dataset(ims_path, inference_only=True)

        test_loader = DataLoader(
            ApplyTransform(test_dataset, test_dataset.val_transform, inference_only=True),
            batch_size=args.batch_size, shuffle=False, num_workers=4
        )

    if args.train and not args.inference_only:

        model = build_regression_model(backbone=args.backbone, pretrained=args.pretrained)

        model = model.to(device)

        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=args.lr)

        # reduces LR by a factor of 10 if val_loss doesn't improve for 5 epochs
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)

        history = {
            "train_loss": [],
            "train_acc": [],
            "val_loss": [],
            "val_acc": [],
        }

        early_stopping_patience = 30 # Stop if no improvement for 30 epochs
        epochs_no_improve = 0
        best_val_loss_for_es = float('inf')

        best_val_acc = 0.0


        #--------------- TRAINING BLOCK ---------------
        print("Start training...")
        for epoch in range(args.epochs):
            model.train()
            running_loss = 0.0
            correct = 0
            total_train = 0
            start_time = time.time()
            for images, labels in train_loader:
                images, labels = images.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs.squeeze(1), labels.float())
                loss.backward()
                optimizer.step()

                running_loss += loss.item() * images.size(0)
                predicted = outputs.squeeze(1).round() # round to nearest integer
                total_train += labels.size(0)
                correct += predicted.eq(labels.float()).sum().item() 
                

            epoch_loss = running_loss / total_train
            epoch_acc = correct / total_train

            # validation
            model.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0
            with torch.no_grad():
                for images, labels in val_loader:
                    images, labels = images.to(device), labels.to(device)
                    outputs = model(images)
                    loss = criterion(outputs.squeeze(1), labels.float())
                    val_loss += loss.item() * images.size(0)
                    predicted = outputs.squeeze(1).round() # round to nearest integer
                    val_total += labels.size(0)
                    val_correct += predicted.eq(labels.float()).sum().item() 
            val_loss = val_loss / val_total
            val_acc = val_correct / val_total

            scheduler.step(val_loss)

            # ------ Early Stopping Logic ------
            if val_loss < best_val_loss_for_es:
                best_val_loss_for_es = val_loss
                epochs_no_improve = 0
                torch.save(model.state_dict(), os.path.join(args.output_dir, "best_model.pth"))
            else:
                epochs_no_improve += 1
                print(f"Early stopping patience: {epochs_no_improve}/{early_stopping_patience}")

            if epochs_no_improve >= early_stopping_patience:
                print(f"Early stopping triggered after {epochs_no_improve} epochs without improvement.")
                break 
            # ---------------------------------

            # record history
            history["train_loss"].append(epoch_loss)
            history["train_acc"].append(epoch_acc)
            history["val_loss"].append(val_loss)
            history["val_acc"].append(val_acc)

            elapsed = time.time() - start_time
            print(f"Epoch {epoch+1}/{args.epochs} - "
                  f"loss: {epoch_loss:.4f}, acc: {epoch_acc:.4f} - "
                  f"val_loss: {val_loss:.4f}, val_acc: {val_acc:.4f} - "
                  f"time {elapsed:.2f}s")

        #--------------- TRAINING BLOCK ---------------

        plot_history(history=history, output_dir=args.output_dir)
        df = pd.DataFrame(history)
        df.to_csv(args.output_dir+"/history.csv")


    # test set evaluation
    model = build_regression_model(backbone=args.backbone, pretrained=False)
    model.load_state_dict(torch.load(os.path.join(args.output_dir, "best_model.pth"), map_location=device))
    model = model.to(device)
    if not args.inference_only:
        mae, mape, mse, all_preds = evaluate_model(model, test_loader, device)
        print(f"Test mae: {mae:.4f}")
        print(f"Test mape: {mape:.4f}")
        print(f"Test mse: {mse:.4f}")
    else:
        all_preds = make_preds(model, test_loader, device)

    output_json_data = []
    for i, path in enumerate(ims_path):
        output_json_data.append({
            "image_path": path,
            "num_balls": all_preds[i],
        })

    # --- NEW: Save the JSON file ---
    with open(os.path.join(args.output_dir, "predictions.json"), "w") as outfile:
        json.dump(output_json_data, outfile, indent=4)
    print("="*30)
    print(f"Saved predictions to {os.path.join(args.output_dir, "predictions.json")}")
    print("="*30)
    # --- NEW: Save the JSON file ---

    if not args.inference_only:
        with open(os.path.join(args.output_dir,"metrics.txt"), mode='w') as f:
            f.write(f"""Test mae: {mae:.4f}
Test mape: {mape:.4f}
Test mse: {mse:.4f}""")
        print("="*30)
        print(f"Saved Metrics to {os.path.join(args.output_dir,"metrics.txt")}")
        print("="*30)
        