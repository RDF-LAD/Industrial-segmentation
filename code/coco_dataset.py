import json
import os
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw
from torch.utils.data import Dataset


class CocoInstanceDataset(Dataset):
    """
    Dataset d'instance segmentation au format COCO (polygones), prêt pour Mask2Former.

    Chaque item contient :
      - pixel_values : (3, H, W) float, prétraité par le processor
      - pixel_mask   : (H, W) long, 1 partout
      - class_labels : (N,) long, une classe par instance, dans [0, num_labels-1]
      - mask_labels  : (N, H, W) float, un masque binaire par instance (les chevauchements
                       feu/fumée sont conservés)
    """

    def __init__(self, img_dir, ann_file, processor, cat2label=None):
        self.img_dir = img_dir
        self.processor = processor

        with open(ann_file, "r", encoding="utf-8") as f:
            coco = json.load(f)

        self.images = coco["images"]
        self.anns = defaultdict(list)
        for a in coco["annotations"]:
            self.anns[a["image_id"]].append(a)

        # Mapping id COCO -> indice de classe contigu (0..K-1).
        # Roboflow ajoute une catégorie "parent" (id 0, ici "fire-smoke") sans aucune annotation :
        # on ne garde que les catégories réellement utilisées.
        # Pour un jeu de validation, passer le cat2label du train pour garder les mêmes indices.
        names = {c["id"]: c["name"] for c in coco["categories"]}
        if cat2label is None:
            used = sorted({a["category_id"] for a in coco["annotations"]})
            cat2label = {cid: i for i, cid in enumerate(used)}
        self.cat2label = cat2label
        self.id2label = {i: names[cid] for cid, i in cat2label.items()}

    def __len__(self):
        return len(self.images)

    def _polygons_to_mask(self, segmentation, w, h):
        mask = Image.new("L", (w, h), 0)
        draw = ImageDraw.Draw(mask)
        for poly in segmentation:  # une annotation peut avoir plusieurs polygones
            if len(poly) >= 6:  # au moins 3 points
                draw.polygon(poly, outline=1, fill=1)
        return np.array(mask, dtype=np.uint8)

    def __getitem__(self, idx):
        info = self.images[idx]
        image = Image.open(os.path.join(self.img_dir, info["file_name"])).convert("RGB")
        w, h = image.size

        masks, labels = [], []
        for a in self.anns.get(info["id"], []):
            seg = a["segmentation"]
            if a.get("iscrowd", 0) or not isinstance(seg, list):  # RLE non géré ici
                continue
            if a["category_id"] not in self.cat2label:
                continue
            masks.append(self._polygons_to_mask(seg, w, h))
            labels.append(self.cat2label[a["category_id"]])

        # Même prétraitement que celui utilisé à l'inférence (resize + normalisation)
        pixel_values = self.processor(images=image, return_tensors="pt")["pixel_values"][0]
        H, W = pixel_values.shape[-2:]

        if masks:
            mask_labels = torch.from_numpy(np.stack(masks)).float()  # (N, h, w)
            # Les masques suivent la même géométrie que l'image prétraitée
            mask_labels = F.interpolate(mask_labels[None], size=(H, W), mode="nearest")[0]
        else:
            mask_labels = torch.zeros((0, H, W), dtype=torch.float32)

        return {
            "pixel_values": pixel_values,
            "pixel_mask": torch.ones((H, W), dtype=torch.long),
            "class_labels": torch.tensor(labels, dtype=torch.long),
            "mask_labels": mask_labels,
        }


def mask2former_collate_fn(batch):
    """Les images ont la même taille après prétraitement : on empile directement.
    class_labels / mask_labels restent des listes (nombre d'instances variable par image)."""
    return {
        "pixel_values": torch.stack([b["pixel_values"] for b in batch]),
        "pixel_mask": torch.stack([b["pixel_mask"] for b in batch]),
        "mask_labels": [b["mask_labels"] for b in batch],
        "class_labels": [b["class_labels"] for b in batch],
    }