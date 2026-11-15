"""
Prédit la segmentation sur le jeu de test et, pour chaque image contenant une erreur, produit une figure :
ground truth à gauche, prédiction à droite.

Une erreur est :
  - une fausse détection (FP) : instance prédite (score >= seuil de sa classe) sans annotation correspondante
    de même classe à IoU >= IOU_MATCH ;
  - une instance ratée (FN) : annotation sans prédiction correspondante.
Dans les figures, ces instances sont entourées en blanc et marquées FP ou FN.

Usage :
    python show_errors.py --checkpoint results/checkpoint-17625
    python show_errors.py --checkpoint results/checkpoint-17625 --thresholds fire=0.40 smoke=0.65 --max_show 5
"""
import argparse
import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch

from architecture import IndustrialFireSmokeSegmenter
from validation_checkpoint import RESULTS_DIR, ground_truth, load_valid, predict
from metric_evaluation_final import TEST_DIR, resolve_thresholds
from validation_threshold import DEFAULT_CHECKPOINT, IOU_MATCH, mask_iou

# Couleurs RGB par nom de classe
COLORS = {"fire": (255, 60, 0), "smoke": (40, 120, 255)}
DEFAULT_COLOR = (0, 200, 0)


def analyse_image(pred, gt_masks, gt_labels, thr, class_ids):
    """
    Applique les seuils par classe, puis apparie prédictions et annotations (par classe, score décroissant).
    Renvoie les prédictions conservées et leurs statuts : pred_tp (True = bien détectée) et
    gt_matched (True = annotation retrouvée).
    """
    keep = torch.tensor([float(s) >= thr[int(l)] for s, l in zip(pred["scores"], pred["labels"])], dtype=torch.bool)
    p_masks, p_labels, p_scores = pred["masks"][keep], pred["labels"][keep], pred["scores"][keep]

    pred_tp = np.zeros(len(p_scores), dtype=bool)
    gt_matched = np.zeros(len(gt_labels), dtype=bool)
    for c in class_ids:
        pi = np.where((p_labels == c).numpy())[0]
        gi = np.where((gt_labels == c).numpy())[0]
        if len(pi) == 0 or len(gi) == 0:
            continue
        order = pi[np.argsort(-p_scores[pi].numpy())]
        iou = mask_iou(p_masks[torch.as_tensor(order)], gt_masks[torch.as_tensor(gi)])
        for r, i in enumerate(order):
            cand = np.where(gt_matched[gi], -1.0, iou[r])
            j = int(cand.argmax())
            if cand[j] >= IOU_MATCH:
                pred_tp[i], gt_matched[gi[j]] = True, True
    return p_masks, p_labels, p_scores, pred_tp, gt_matched


def draw_instances(img_rgb, masks, labels, id2label, texts, is_error, alpha=0.5):
    """Superpose des instances (masques booléens numpy) : remplissage translucide, contour, texte.
    Les instances en erreur ont un contour blanc épais."""
    fill = img_rgb.copy()
    colors = [COLORS.get(id2label[int(l)], DEFAULT_COLOR) for l in labels]
    for m, col in zip(masks, colors):
        fill[m] = col
    out = cv2.addWeighted(fill, alpha, img_rgb, 1 - alpha, 0)

    for m, col, text, err in zip(masks, colors, texts, is_error):
        contours, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if err:
            cv2.drawContours(out, contours, -1, (255, 255, 255), 4)
        cv2.drawContours(out, contours, -1, col, 2)
        ys, xs = np.where(m)
        if len(xs) == 0:
            continue
        pos = (int(xs.min()), max(int(ys.min()), 18))
        cv2.putText(out, text, pos, cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(out, text, pos, cv2.FONT_HERSHEY_SIMPLEX, 0.55, col, 1, cv2.LINE_AA)
    return out


def make_figure(image, gt_masks, gt_labels, gt_matched, p_masks, p_labels, p_scores, pred_tp, id2label, title):
    img = np.array(image)
    gt_np = gt_masks.numpy()
    p_np = p_masks.numpy()
    n_fn, n_fp = int((~gt_matched).sum()), int((~pred_tp).sum())

    gt_texts = [id2label[int(l)] + ("" if ok else " FN") for l, ok in zip(gt_labels, gt_matched)]
    pr_texts = [f"{id2label[int(l)]} {float(s):.2f}" + ("" if ok else " FP")
                for l, s, ok in zip(p_labels, p_scores, pred_tp)]

    left = draw_instances(img, gt_np, gt_labels, id2label, gt_texts, ~gt_matched)
    right = draw_instances(img, p_np, p_labels, id2label, pr_texts, ~pred_tp)

    fig, axes = plt.subplots(1, 2, figsize=(13, 6.5))
    axes[0].imshow(left)
    axes[0].set_title(f"Ground truth ({len(gt_labels)} instances, {n_fn} ratée(s) FN)")
    axes[1].imshow(right)
    axes[1].set_title(f"Prédiction ({len(p_scores)} instances, {n_fp} fausse(s) détection(s) FP)")
    for ax in axes:
        ax.axis("off")
    fig.suptitle(title, fontsize=9)
    plt.tight_layout()
    return fig


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT, help="dossier du checkpoint retenu")
    parser.add_argument("--test_dir", default=TEST_DIR)
    parser.add_argument("--ann_file", default=None, help="défaut : <test_dir>/_annotations.coco.json")
    parser.add_argument("--thresholds_json", default=os.path.join(RESULTS_DIR, "recommended_thresholds.json"))
    parser.add_argument("--mode", choices=["f1", "recall"], default="f1")
    parser.add_argument("--thresholds", nargs="*", help="seuils manuels, ex: fire=0.40 smoke=0.65")
    parser.add_argument("--output_dir", default=os.path.join(RESULTS_DIR, "errors"),
                        help="dossier où sont sauvegardées les figures d'erreurs")
    parser.add_argument("--max_show", type=int, default=10,
                        help="nombre de figures affichées à l'écran (toutes sont sauvegardées)")
    args = parser.parse_args()
    ann_file = args.ann_file or os.path.join(args.test_dir, "_annotations.coco.json")

    segmenter = IndustrialFireSmokeSegmenter(checkpoint_path=args.checkpoint, freeze_backbone=False)
    segmenter.model.eval()
    id2label = segmenter.id2label
    class_ids = sorted(id2label.keys())
    thr = resolve_thresholds(args, id2label)
    print("Seuils utilisés :", {id2label[c]: thr[c] for c in class_ids})

    images, anns, cat2label = load_valid(ann_file, {v: k for k, v in id2label.items()})
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"{len(images)} images de test")

    errors, n_ok, tot_fp, tot_fn = [], 0, 0, 0
    for i, info in enumerate(images, 1):
        image, gt_masks, gt_labels = ground_truth(info, anns, cat2label, args.test_dir)
        pred = predict(segmenter, image)
        p_masks, p_labels, p_scores, pred_tp, gt_matched = analyse_image(pred, gt_masks, gt_labels, thr, class_ids)
        n_fp, n_fn = int((~pred_tp).sum()), int((~gt_matched).sum())
        tot_fp += n_fp
        tot_fn += n_fn
        if i % 50 == 0 or i == len(images):
            print(f"  {i}/{len(images)} images traitées")
        if n_fp + n_fn == 0:
            n_ok += 1
            continue

        fig = make_figure(image, gt_masks, gt_labels, gt_matched, p_masks, p_labels, p_scores, pred_tp,
                          id2label, title=info["file_name"])
        path = os.path.join(args.output_dir, f"{os.path.splitext(os.path.basename(info['file_name']))[0]}.png")
        fig.savefig(path, dpi=110)
        plt.close(fig)  # évite d'accumuler des figures ouvertes
        errors.append((n_fp + n_fn, n_fp, n_fn, info["file_name"], path))

    print(f"\n{len(images)} images : {n_ok} sans erreur, {len(errors)} avec erreur(s) "
          f"(total : {tot_fp} FP, {tot_fn} FN, IoU >= {IOU_MATCH})")
    errors.sort(key=lambda e: -e[0])  # les images les plus problématiques d'abord
    for _, fp, fn, name, path in errors:
        print(f"  FP={fp} FN={fn}  {name}")
    print(f"Figures sauvegardées dans {args.output_dir}")

    # Affichage des cas les plus problématiques
    for _, _, _, name, path in errors[: args.max_show]:
        fig = plt.figure(figsize=(13, 6.5))
        plt.imshow(plt.imread(path))
        plt.axis("off")
        plt.tight_layout()
    if errors and args.max_show > 0:
        plt.show()


if __name__ == "__main__":
    main()