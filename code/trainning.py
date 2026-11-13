import numpy as np
from architecture import IndustrialFireSmokeSegmenter
from coco_dataset import CocoInstanceDataset, mask2former_collate_fn
from transformers import Mask2FormerImageProcessor
import torch

print("GPU disponible :", torch.cuda.is_available())

#label of the dataset
id2label = {0: "fire-smoke", 1: "fire", 2: "smoke"}

model_id = "facebook/mask2former-swin-tiny-coco-instance"
processor = Mask2FormerImageProcessor.from_pretrained(model_id)

train_ds = CocoInstanceDataset(
    img_dir="C:\\Users\\User\\Documents\\projets_persos_polimi\\Segmentation\\data\\Fire and Smoke Segmentation.v10-rfdetr-xl.coco-segmentation\\train",
    ann_file="C:\\Users\\User\\Documents\\projets_persos_polimi\\Segmentation\\data\\Fire and Smoke Segmentation.v10-rfdetr-xl.coco-segmentation\\_annotations_train.coco.json",
    processor=processor,
)

segmenter = IndustrialFireSmokeSegmenter(model_id, freeze_backbone=True, id2label=train_ds.id2label)
segmenter.train(train_ds, epochs=5, batch_size=2)