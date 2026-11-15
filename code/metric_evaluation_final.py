import argparse
import csv
import json
import os
import time

import numpy as np
from torchmetrics.detection import MeanAveragePrecision

from architecture import IndustrialFireSmokeSegmenter
from validation_checkpoint import RESULTS_DIR, VALID_DIR, ground_truth, load_valid, per_class, predict
from validation_threshold import DEFAULT_CHECKPOINT, IOU_MATCH, match_image

# ----------------------------- Configuration -----------------------------
TEST_DIR = os.path.join(os.path.dirname(VALID_DIR), "test")  # dossier frère de "valid" ; adapte si besoin
# --------------------------------------------------------------------------


def prf(tp, fp, fn):
    precision = tp / (tp + fp) if tp + fp > 0 else float("nan")
    recall = tp / (tp + fn) if tp + fn > 0 else float("nan")
    f1 = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn > 0 else 0.0
    return precision, recall, f1


def resolve_thresholds(args, id2label):
    """Seuils par classe : soit donnés à la main (--thresholds fire=0.4 ...), soit lus dans le JSON de tune_threshold."""
    name2id = {v: k for k, v in id2label.items()}
    if args.thresholds:
        by_name = {kv.split("=")[0]: float(kv.split("=")[1]) for kv in args.thresholds}
    else:
        with open(args.thresholds_json, "r", encoding="utf-8") as f:
            summary = json.load(f)["thresholds"]
        key = "best_f1_threshold" if args.mode == "f1" else "recall_target_threshold"
        by_name = {}
        for name in id2label.values():
            value = summary[name].get(key)
            if value is None:
                print(f"[Attention] Pas de seuil '{key}' pour {name} : utilisation du seuil de meilleur F1.")
                value = summary[name]["best_f1_threshold"]
            by_name[name] = float(value)
    missing = [n for n in id2label.values() if n not in by_name]
    if missing:
        raise ValueError(f"Seuil manquant pour : {missing}")
    return {name2id[n]: t for n, t in by_name.items()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT, help="dossier du checkpoint retenu")
    parser.add_argument("--test_dir", default=TEST_DIR)
    parser.add_argument("--ann_file", default=None, help="défaut : <test_dir>/_annotations.coco.json")
    parser.add_argument("--thresholds_json", default=os.path.join(RESULTS_DIR, "recommended_thresholds.json"))
    parser.add_argument("--mode", choices=["f1", "recall"], default="f1",
                        help="seuils du JSON à utiliser : meilleur F1 ou rappel cible")
    parser.add_argument("--thresholds", nargs="*", help="seuils manuels, ex: fire=0.40 smoke=0.65 (remplace le JSON)")
    parser.add_argument("--output_dir", default=RESULTS_DIR)
    args = parser.parse_args()
    ann_file = args.ann_file or os.path.join(args.test_dir, "_annotations.coco.json")

    segmenter = IndustrialFireSmokeSegmenter(checkpoint_path=args.checkpoint, freeze_backbone=False)
    segmenter.model.eval()
    id2label = segmenter.id2label
    class_ids = sorted(id2label.keys())
    thr = resolve_thresholds(args, id2label)
    print("Seuils utilisés :", {id2label[c]: thr[c] for c in class_ids})

    images, anns, cat2label = load_valid(ann_file, {v: k for k, v in id2label.items()})
    print(f"{len(images)} images de test | classes : {id2label}")

    m_coco = MeanAveragePrecision(iou_type="segm", class_metrics=True)
    m_50 = MeanAveragePrecision(iou_type="segm", iou_thresholds=[0.5], class_metrics=True)
    records = {c: [] for c in class_ids}
    n_gt = {c: 0 for c in class_ids}
    t_list=[]
    for i, info in enumerate(images, 1):
        image, gt_masks, gt_labels = ground_truth(info, anns, cat2label, args.test_dir)
        t0=time.time()
        pred = predict(segmenter, image)
        t1=time.time()
        m_coco.update([pred], [{"masks": gt_masks, "labels": gt_labels}])
        m_50.update([pred], [{"masks": gt_masks, "labels": gt_labels}])
        match_image(pred, gt_masks, gt_labels, class_ids, records, n_gt)
        if i % 50 == 0 or i == len(images):
            print(f"  {i}/{len(images)} images traitées")
        t_list.append(t1-t0)
    print(f'mean inference time : {np.mean(np.array(t_list)):.4} s.')
    rc, r5 = m_coco.compute(), m_50.compute()
    pc_map, pc_50 = per_class(rc), per_class(r5)

    rows, tot = [], [0, 0, 0]
    for c in class_ids:
        s = np.concatenate([r[0] for r in records[c]]) if records[c] else np.zeros(0)
        t = np.concatenate([r[1] for r in records[c]]) if records[c] else np.zeros(0, dtype=bool)
        keep = s >= thr[c]
        tp, fp = int((t & keep).sum()), int((~t & keep).sum())
        fn = n_gt[c] - tp
        p, r, f = prf(tp, fp, fn)
        tot[0] += tp
        tot[1] += fp
        tot[2] += fn
        rows.append({"classe": id2label[c], "seuil": thr[c], "n_gt": n_gt[c], "TP": tp, "FP": fp, "FN": fn,
                     "precision": p, "recall": r, "f1": f,
                     "AP50": pc_50.get(c, float("nan")), "mAP": pc_map.get(c, float("nan"))})
    p, r, f = prf(*tot)
    rows.append({"classe": "global", "seuil": float("nan"), "n_gt": sum(n_gt.values()), "TP": tot[0], "FP": tot[1],
                 "FN": tot[2], "precision": p, "recall": r, "f1": f,
                 "AP50": r5["map"].item(), "mAP": rc["map"].item()})

    cols = list(rows[0].keys())
    print(f"\nRésultats sur le test (appariement IoU >= {IOU_MATCH} pour P/R/F1 ; AP50 et mAP indépendants des seuils)")
    print("  ".join(f"{c:>10}" for c in cols))
    for row in rows:
        print("  ".join(f"{row[c]:>10}" if isinstance(row[c], str) else
                        (f"{row[c]:>10d}" if isinstance(row[c], int) else f"{row[c]:>10.4f}") for c in cols))
    print("\nGlobal : P/R/F1 en micro (TP, FP, FN cumulés), AP50 et mAP en moyenne sur les classes.")
    print("Valeur -1 ou nan : classe absente des annotations de test.")

    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, "test_metrics.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
    with open(os.path.join(args.output_dir, "test_metrics.json"), "w", encoding="utf-8") as f:
        json.dump({"checkpoint": args.checkpoint, "iou_match": IOU_MATCH,
                   "thresholds": {id2label[c]: thr[c] for c in class_ids}, "metrics": rows},
                  f, indent=2, ensure_ascii=False)
    print(f"Sauvegardé dans {args.output_dir} : test_metrics.csv, test_metrics.json")


if __name__ == "__main__":
    main()
    