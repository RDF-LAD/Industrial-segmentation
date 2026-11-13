import numpy as np
from architecture import IndustrialFireSmokeSegmenter
from coco_dataset import CocoInstanceDataset, mask2former_collate_fn
from transformers import Mask2FormerImageProcessor
import torch
import cv2
import matplotlib.pyplot as plt

# Couleurs RGB par nom de classe (celles de id2label) ; couleur par défaut pour une autre classe
COLORS = {
    "fire": (255, 60, 0),
    "smoke": (40, 120, 255),
}
DEFAULT_COLOR = (0, 200, 0)


def overlay_segmentation(img_rgb, result, id2label, alpha=0.5):
    """
    Superpose les instances prédites sur l'image (remplissage translucide, contour, étiquette + score).
    img_rgb : image (H, W, 3) uint8 en RGB
    result  : sortie de post_process_instance_segmentation (dict 'segmentation' + 'segments_info')
    """
    seg = result["segmentation"].cpu().numpy()  # (H, W), chaque pixel = id d'instance, -1 si aucune
    fill = img_rgb.copy()
    segments = []

    for info in result["segments_info"]:
        mask = seg == info["id"]
        if not mask.any():
            continue
        name = id2label[info["label_id"]]
        color = COLORS.get(name, DEFAULT_COLOR)
        fill[mask] = color
        segments.append((mask, name, info["score"], color))

    # Seuls les pixels masqués diffèrent de l'image d'origine : le mélange ne touche qu'eux
    out = cv2.addWeighted(fill, alpha, img_rgb, 1 - alpha, 0)

    for mask, name, score, color in segments:
        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(out, contours, -1, color, 2)
        ys, xs = np.where(mask)
        x, y = int(xs.min()), max(int(ys.min()), 15)
        cv2.putText(out, f"{name} {score:.2f}", (x+50, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2, cv2.LINE_AA)

    return out


if __name__ == "__main__":
    print("GPU disponible :", torch.cuda.is_available())

    # Load trained model
    model = IndustrialFireSmokeSegmenter(checkpoint_path=
                                         r'C:\Users\User\Documents\projets_persos_polimi\Segmentation\results\checkpoint-17625')

    # test image inference
    img_pth = r'C:\Users\User\Documents\projets_persos_polimi\Segmentation\data\Fire and Smoke Segmentation.v10-rfdetr-xl.coco-segmentation\bonus\img1.webp'
    r1 = model.inference(img_pth, confidence_threshold=0.5)

    print(f"{len(r1['segments_info'])} instance(s) détectée(s) :")
    for s in r1["segments_info"]:
        print(f"  id={s['id']}  classe={model.id2label[s['label_id']]}  score={s['score']:.3f}")

    img = cv2.imread(img_pth)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    overlay = overlay_segmentation(img, r1, model.id2label, alpha=0.5)

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    axes[0].imshow(img)
    axes[0].set_title("Image")
    axes[1].imshow(overlay)
    axes[1].set_title("Segmentation prédite")
    for ax in axes:
        ax.axis('off')
    plt.tight_layout()
    plt.show()