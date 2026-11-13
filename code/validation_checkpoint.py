import csv
import glob
import json
import os
import re
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw
from torchmetrics.detection import MeanAveragePrecision

from architecture import IndustrialFireSmokeSegmenter

# ----------------------------- Configuration -----------------------------
RESULTS_DIR = r'C:\Users\User\Documents\projets_persos_polimi\Segmentation\results'
VALID_DIR = (r'C:\Users\User\Documents\projets_persos_polimi\Segmentation\data'
             r'\Fire and Smoke Segmentation.v10-rfdetr-xl.coco-segmentation\valid')
ANN_FILE = os.path.join(VALID_DIR, "_annotations.coco.json")  # adapte si le nom du fichier diffère

TOPK = 100          # nombre max de prédictions par image (couples requête/classe)
MIN_SCORE = 0.01    # on écarte les prédictions au score négligeable (accélère l'évaluation)
MASK_THRESHOLD = 0.5  # binarisation des masques prédits
# --------------------------------------------------------------------------


def list_checkpoints(results_dir):
    paths = glob.glob(os.path.join(results_dir, "checkpoint-*"))
    paths = [p for p in paths if re.search(r"checkpoint-(\d+)$", p)]
    return sorted(paths, key=lambda p: int(re.search(r"checkpoint-(\d+)$", p).group(1)))


def load_valid(ann_file, name2label):
    """Lit le COCO de validation. Les classes sont mises en correspondance par NOM avec celles du modèle."""
    with open(ann_file, "r", encoding="utf-8") as f:
        coco = json.load(f)
    cat2label = {c["id"]: name2label[c["name"]] for c in coco["categories"] if c["name"] in name2label}
    anns = defaultdict(list)
    for a in coco["annotations"]:
        anns[a["image_id"]].append(a)
    return coco["images"], anns, cat2label


def ground_truth(info, anns, cat2label, img_dir):
    """Image PIL + masques binaires (N, H, W) à la résolution d'origine + labels (N,)."""
    image = Image.open(os.path.join(img_dir, info["file_name"])).convert("RGB")
    w, h = image.size
    masks, labels = [], []
    for a in anns.get(info["id"], []):
        seg = a["segmentation"]
        if a.get("iscrowd", 0) or not isinstance(seg, list) or a["category_id"] not in cat2label:
            continue
        m = Image.new("L", (w, h), 0)
        draw = ImageDraw.Draw(m)
        for poly in seg:
            if len(poly) >= 6:
                draw.polygon(poly, outline=1, fill=1)
        masks.append(torch.from_numpy(np.array(m, dtype=np.uint8)).bool())
        labels.append(cat2label[a["category_id"]])
    if masks:
        return image, torch.stack(masks), torch.tensor(labels, dtype=torch.long)
    return image, torch.zeros((0, h, w), dtype=torch.bool), torch.zeros((0,), dtype=torch.long)


@torch.no_grad()
def predict(segmenter, image):
    """
    Prédictions d'instances sans résolution de chevauchements (feu et fumée peuvent se superposer),
    selon la procédure d'inférence d'instances de Mask2Former :
    top-K sur les couples (requête, classe), score = proba de classe * confiance moyenne du masque.
    Les logits de masque (1/4 de résolution) sont interpolés directement à la taille d'origine.
    """
    w, h = image.size
    inputs = segmenter.processor(images=image, return_tensors="pt").to(segmenter.device)
    out = segmenter.model(**inputs)

    class_probs = out.class_queries_logits[0].softmax(-1)[:, :-1]  # (Q, K) sans la classe "no object"
    mask_logits = out.masks_queries_logits[0]                      # (Q, h/4, w/4)
    num_classes = class_probs.shape[1]

    scores, idx = class_probs.flatten().topk(min(TOPK, class_probs.numel()))
    query_idx, labels = idx // num_classes, idx % num_classes
    keep = scores >= MIN_SCORE
    scores, query_idx, labels = scores[keep], query_idx[keep], labels[keep]
    if scores.numel() == 0:
        return {"masks": torch.zeros((0, h, w), dtype=torch.bool), "scores": torch.zeros(0),
                "labels": torch.zeros(0, dtype=torch.long)}

    logits = F.interpolate(mask_logits[query_idx][None], size=(h, w), mode="bilinear", align_corners=False)[0]
    probs = logits.sigmoid()
    binary = probs > MASK_THRESHOLD
    mask_score = (probs * binary).flatten(1).sum(1) / (binary.flatten(1).sum(1) + 1e-6)
    scores = scores * mask_score

    non_empty = binary.flatten(1).any(1)  # une prédiction sans aucun pixel n'a pas de sens
    return {"masks": binary[non_empty].cpu(), "scores": scores[non_empty].cpu(), "labels": labels[non_empty].cpu()}


def per_class(res):
    classes = torch.atleast_1d(res["classes"]).tolist()
    values = torch.atleast_1d(res["map_per_class"]).tolist()
    return dict(zip(classes, values))


def main():
    ckpts = list_checkpoints(RESULTS_DIR)
    if not ckpts:
        raise SystemExit(f"Aucun checkpoint trouvé dans {RESULTS_DIR}")
    names = [os.path.basename(c) for c in ckpts]
    print("Checkpoints :", names)

    # Un modèle par checkpoint, tous en mémoire pour ne lire chaque image qu'une seule fois
    segmenters = []
    for c in ckpts:
        s = IndustrialFireSmokeSegmenter(checkpoint_path=c, freeze_backbone=False)
        s.model.eval()
        segmenters.append(s)

    id2label = segmenters[0].id2label
    name2label = {v: k for k, v in id2label.items()}
    images, anns, cat2label = load_valid(ANN_FILE, name2label)
    print(f"{len(images)} images de validation | classes : {id2label}")

    # Deux métriques par checkpoint : COCO mAP (0.5:0.95) et AP à IoU = 0.5 uniquement
    # (torchmetrics ne donne pas l'AP50 par classe avec les seuils par défaut)
    m_coco = [MeanAveragePrecision(iou_type="segm", class_metrics=True) for _ in ckpts]
    m_50 = [MeanAveragePrecision(iou_type="segm", iou_thresholds=[0.5], class_metrics=True) for _ in ckpts]

    for i, info in enumerate(images, 1):
        image, gt_masks, gt_labels = ground_truth(info, anns, cat2label, VALID_DIR)
        target = [{"masks": gt_masks, "labels": gt_labels}]
        for k, seg in enumerate(segmenters):
            pred = [predict(seg, image)]
            m_coco[k].update(pred, target)
            m_50[k].update(pred, target)
        if i % 50 == 0 or i == len(images):
            print(f"  {i}/{len(images)} images traitées")

    # Résultats
    rows = []
    for name, mc, m5 in zip(names, m_coco, m_50):
        rc, r5 = mc.compute(), m5.compute()
        pc_map, pc_50 = per_class(rc), per_class(r5)
        row = {"checkpoint": name, "mAP": rc["map"].item(), "AP50": r5["map"].item()}
        for lab, cname in id2label.items():
            row[f"mAP_{cname}"] = pc_map.get(lab, float("nan"))
            row[f"AP50_{cname}"] = pc_50.get(lab, float("nan"))
        rows.append(row)

    cols = list(rows[0].keys())
    print("\n" + "  ".join(f"{c:>16}" for c in cols))
    for r in rows:
        print("  ".join(f"{r[c]:>16}" if c == "checkpoint" else f"{r[c]:>16.4f}" for c in cols))
    best = max(rows, key=lambda r: r["mAP"])
    print(f"\nMeilleur checkpoint selon le mAP : {best['checkpoint']} (mAP = {best['mAP']:.4f})")
    print("Valeur -1 ou nan : classe absente des annotations de validation.")

    csv_path = os.path.join(RESULTS_DIR, "eval_checkpoints.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)

    # Courbes d'évolution
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharex=True)
    x = [r["checkpoint"].replace("checkpoint-", "") for r in rows]
    for ax, prefix, title in [(axes[0], "mAP", "mAP (IoU 0.5:0.95)"), (axes[1], "AP50", "AP50")]:
        ax.plot(x, [r[prefix] for r in rows], "k--o", label="global")
        for cname in id2label.values():
            ax.plot(x, [r[f"{prefix}_{cname}"] for r in rows], "-o", label=cname)
        ax.set_title(title)
        ax.set_xlabel("step du checkpoint")
        ax.grid(alpha=0.3)
        ax.legend()
    plt.tight_layout()
    fig_path = os.path.join(RESULTS_DIR, "eval_checkpoints.png")
    plt.savefig(fig_path, dpi=150)
    print(f"Résultats sauvegardés : {csv_path} et {fig_path}")
    plt.show()


if __name__ == "__main__":
    main()