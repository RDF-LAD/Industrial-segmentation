import argparse
import csv
import json
import os

import matplotlib.pyplot as plt
import numpy as np
import torch

from architecture import IndustrialFireSmokeSegmenter
from validation_checkpoint import ANN_FILE, RESULTS_DIR, VALID_DIR, ground_truth, load_valid, predict

# ----------------------------- Configuration -----------------------------
DEFAULT_CHECKPOINT = r'C:\Users\User\Documents\projets_persos_polimi\Segmentation\results\checkpoint-3525'
IOU_MATCH = 0.5          # IoU minimal pour qu'une prédiction compte comme détection correcte (TP)
TARGET_RECALL = 0.90     # rappel visé pour le seuil "sécurité" (rater un feu coûte cher)
THRESHOLDS = np.round(np.arange(0.05, 0.96, 0.01), 2)
# --------------------------------------------------------------------------


def mask_iou(pred, gt):
    """IoU entre masques binaires : pred (P, H, W), gt (G, H, W) -> (P, G)."""
    p, g = pred.flatten(1).float(), gt.flatten(1).float()
    inter = p @ g.T
    union = p.sum(1)[:, None] + g.sum(1)[None, :] - inter
    return (inter / union.clamp(min=1)).numpy()


def match_image(pred, gt_masks, gt_labels, class_ids, records, n_gt):
    """
    Apparie prédictions et annotations d'une image, classe par classe, par ordre de score décroissant.
    Chaque prédiction reçoit un statut TP/FP et garde son score. Comme l'appariement glouton suit
    l'ordre des scores, filtrer ensuite par seuil ne change pas le statut des prédictions conservées :
    un seul appariement suffit donc pour tous les seuils.
    """
    for c in class_ids:
        gt_c = gt_labels == c
        n_gt[c] += int(gt_c.sum())
        sel = pred["labels"] == c
        if int(sel.sum()) == 0:
            continue
        scores, masks = pred["scores"][sel], pred["masks"][sel]
        order = scores.argsort(descending=True)
        scores, masks = scores[order], masks[order]

        is_tp = np.zeros(len(scores), dtype=bool)
        if gt_c.any():
            iou = mask_iou(masks, gt_masks[gt_c])
            taken = np.zeros(iou.shape[1], dtype=bool)
            for i in range(len(scores)):
                cand = np.where(taken, -1.0, iou[i])
                j = int(cand.argmax())
                if cand[j] >= IOU_MATCH:
                    is_tp[i], taken[j] = True, True
        records[c].append((scores.numpy(), is_tp))


def curves(scores, is_tp, n_gt, thresholds):
    """Précision, rappel et F1 pour chaque seuil. Précision = nan si aucune prédiction n'est conservée."""
    prec, rec, f1 = [], [], []
    for t in thresholds:
        keep = scores >= t
        tp = int((is_tp & keep).sum())
        fp = int((~is_tp & keep).sum())
        fn = n_gt - tp
        prec.append(tp / (tp + fp) if tp + fp > 0 else np.nan)
        rec.append(tp / n_gt if n_gt > 0 else np.nan)
        f1.append(2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn > 0 else 0.0)
    return np.array(prec), np.array(rec), np.array(f1)


def summarize(name, thresholds, prec, rec, f1):
    """Seuil qui maximise le F1, et plus grand seuil atteignant le rappel cible."""
    b = int(np.argmax(f1))
    out = {"best_f1_threshold": float(thresholds[b]), "precision": float(prec[b]),
           "recall": float(rec[b]), "f1": float(f1[b]), "recall_target_threshold": None}
    ok = np.where(rec >= TARGET_RECALL)[0]
    if len(ok):
        k = int(ok[-1])
        out["recall_target_threshold"] = float(thresholds[k])
        out["precision_at_recall_target"] = float(prec[k])
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT, help="dossier du meilleur checkpoint")
    parser.add_argument("--output_dir", default=RESULTS_DIR)
    args = parser.parse_args()

    segmenter = IndustrialFireSmokeSegmenter(checkpoint_path=args.checkpoint, freeze_backbone=False)
    segmenter.model.eval()
    id2label = segmenter.id2label
    class_ids = sorted(id2label.keys())
    name2label = {v: k for k, v in id2label.items()}
    images, anns, cat2label = load_valid(ANN_FILE, name2label)
    print(f"{len(images)} images de validation | classes : {id2label}")

    records = {c: [] for c in class_ids}
    n_gt = {c: 0 for c in class_ids}
    for i, info in enumerate(images, 1):
        image, gt_masks, gt_labels = ground_truth(info, anns, cat2label, VALID_DIR)
        pred = predict(segmenter, image)
        match_image(pred, gt_masks, gt_labels, class_ids, records, n_gt)
        if i % 50 == 0 or i == len(images):
            print(f"  {i}/{len(images)} images traitées")

    # Courbes par classe, puis globales (micro : on cumule TP/FP/FN de toutes les classes)
    results, all_scores, all_tp = {}, [], []
    for c in class_ids:
        s = np.concatenate([r[0] for r in records[c]]) if records[c] else np.zeros(0)
        t = np.concatenate([r[1] for r in records[c]]) if records[c] else np.zeros(0, dtype=bool)
        all_scores.append(s)
        all_tp.append(t)
        results[id2label[c]] = curves(s, t, n_gt[c], THRESHOLDS)
    results["global"] = curves(np.concatenate(all_scores), np.concatenate(all_tp), sum(n_gt.values()), THRESHOLDS)

    # Résumé console + fichiers
    summary = {}
    print(f"\nAppariement à IoU >= {IOU_MATCH} | rappel cible : {TARGET_RECALL}")
    for name, (p, r, f) in results.items():
        summary[name] = summarize(name, THRESHOLDS, p, r, f)
        s = summary[name]
        rt = s["recall_target_threshold"]
        rt_txt = (f"{rt:.2f} (précision {s['precision_at_recall_target']:.3f})" if rt is not None
                  else "jamais atteint")
        print(f"  {name:>8} | meilleur F1 {s['f1']:.3f} au seuil {s['best_f1_threshold']:.2f} "
              f"(P {s['precision']:.3f}, R {s['recall']:.3f}) | seuil pour rappel >= {TARGET_RECALL} : {rt_txt}")

    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, "threshold_curves.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["threshold"] + [f"{m}_{n}" for n in results for m in ("precision", "recall", "f1")])
        for k, t in enumerate(THRESHOLDS):
            w.writerow([t] + [results[n][m][k] for n in results for m in (0, 1, 2)])
    with open(os.path.join(args.output_dir, "recommended_thresholds.json"), "w", encoding="utf-8") as f:
        json.dump({"checkpoint": args.checkpoint, "iou_match": IOU_MATCH, "target_recall": TARGET_RECALL,
                   "thresholds": summary}, f, indent=2, ensure_ascii=False)

    # Graphiques : précision, rappel, F1 en fonction du seuil
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8), sharex=True)
    for ax, (idx, title) in zip(axes, [(0, "Précision"), (1, "Rappel"), (2, "F1")]):
        for name, curve in results.items():
            style = "k--" if name == "global" else "-"
            ax.plot(THRESHOLDS, curve[idx], style, label=name)
        if idx == 2:
            for name, curve in results.items():
                b = int(np.argmax(curve[2]))
                ax.plot(THRESHOLDS[b], curve[2][b], "o", color="red")
        ax.set_title(title)
        ax.set_xlabel("seuil de confiance")
        ax.set_ylim(0.4, 1.02)
        ax.grid(alpha=0.3)
        ax.legend()
    axes[2].set_title("F1 (points rouges : seuils optimaux)")
    plt.tight_layout()
    fig_path = os.path.join(args.output_dir, "threshold_curves.png")
    plt.savefig(fig_path, dpi=150)
    print(f"\nSauvegardé dans {args.output_dir} : threshold_curves.png, threshold_curves.csv, recommended_thresholds.json")
    plt.show()


if __name__ == "__main__":
    main()