import os

import torch
from PIL import Image
from safetensors.torch import load_file
from transformers import (
    Mask2FormerForUniversalSegmentation,
    Mask2FormerImageProcessor,
    Trainer,
    TrainingArguments
)
from coco_dataset import mask2former_collate_fn


class IndustrialFireSmokeSegmenter:
    def __init__(self, model_id="facebook/mask2former-swin-tiny-coco-instance", freeze_backbone=True,
                 id2label=None, checkpoint_path=None):
        """
        Initialise le processeur, le modèle Mask2Former et gère le gel du backbone.

        checkpoint_path :
          - dossier de checkpoint contenant config.json
            et model.safetensors 
        Si checkpoint_path est None : initialisation depuis les poids COCO de model_id
        (tête de classification réinitialisée pour id2label).
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if checkpoint_path and os.path.isdir(checkpoint_path):
            # Poids fine-tunés
            self.model = Mask2FormerForUniversalSegmentation.from_pretrained(checkpoint_path)
            has_proc = os.path.exists(os.path.join(checkpoint_path, "preprocessor_config.json"))
            self.processor = Mask2FormerImageProcessor.from_pretrained(checkpoint_path if has_proc else model_id)
            print(f"[Info] Modèle chargé depuis le checkpoint : {checkpoint_path}")
        else:
            if id2label is None:
                raise ValueError("id2label est requis.")
            self.processor = Mask2FormerImageProcessor.from_pretrained(model_id)
            self.model = Mask2FormerForUniversalSegmentation.from_pretrained(
                model_id,
                id2label=id2label,
                label2id={v: k for k, v in id2label.items()},
                ignore_mismatched_sizes=True,  # tête de classification : 80 classes COCO -> len(id2label)
            )
            if checkpoint_path:  # fichier .safetensors seul
                state_dict = load_file(checkpoint_path)
                missing, unexpected = self.model.load_state_dict(state_dict, strict=False)
                print(f"[Info] Poids chargés depuis {checkpoint_path} "
                      f"(clés manquantes : {len(missing)}, inattendues : {len(unexpected)})")
                if missing or unexpected:
                    print("Incohérence classes.")

        self.id2label = self.model.config.id2label

        # Geler le backbone Swin-T si besoin
        if freeze_backbone:
            for param in self.model.model.pixel_level_module.encoder.parameters():
                param.requires_grad = False
            n_train = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            n_total = sum(p.numel() for p in self.model.parameters())
            print(f"[Info] Backbone Swin-T gelé. Paramètres entraînables : {n_train/1e6:.1f}M / {n_total/1e6:.1f}M")

        self.model.to(self.device)

    def train(self, train_dataset, val_dataset=None, output_dir="./results", epochs=5, batch_size=2, lr=1e-4,
              resume_from_checkpoint=None):
        """
        resume_from_checkpoint : dossier de checkpoint pour reprendre exactement le run
        """
        training_args = TrainingArguments(
            output_dir=output_dir,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            num_train_epochs=epochs,
            learning_rate=lr,
            logging_steps=10,
            save_strategy="epoch",
            eval_strategy="epoch" if val_dataset else "no",
            remove_unused_columns=False,  # Indispensable sinon bug avec masques
            fp16=torch.cuda.is_available(),
            report_to="none",
            prediction_loss_only=True
        )

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            data_collator=mask2former_collate_fn,
            processing_class=self.processor,
        )

        print("[Info] Lancement du fine-tuning...")
        trainer.train(resume_from_checkpoint=resume_from_checkpoint)
        print("[Info] Entraînement terminé !")

    def _thresholds_per_label(self, threshold): 
        if isinstance(threshold, (int, float)):
            return {k: float(threshold) for k in self.id2label}
        name2id = {v: k for k, v in self.id2label.items()}
        out = {name2id[key] if key in name2id else key: float(val) for key, val in threshold.items()}
        missing = set(self.id2label) - set(out)
        if missing:
            raise ValueError(f"Seuil manquant pour les classes : {[self.id2label[m] for m in missing]}")
        return out

    def inference(self, image_path_or_pil, confidence_threshold={'fire':0.34,'smoke':0.46}):
        """
        confidence_threshold : float ou dict {nom de classe: seuil}
        Renvoie  un dict avec 'segmentation' (carte d'ids, -1 = fond) et 'segments_info' (classe et score de chaque instance).
        """
        self.model.eval()

        # Charger l'image
        if isinstance(image_path_or_pil, str):
            image = Image.open(image_path_or_pil).convert("RGB")
        else:
            image = image_path_or_pil

        original_size = image.size[::-1]  # (height, width)

        # Prétraitement de l'image pour le modèle
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)

        # Le post-traitement n'accepte qu'un seuil scalaire : on utilise le plus bas des seuils par classe d'abord
        thr_per_label = self._thresholds_per_label(confidence_threshold)
        result = self.processor.post_process_instance_segmentation(
            outputs,
            target_sizes=[original_size],
            threshold=min(thr_per_label.values())
        )[0]

        # on verifie le threshold de tous
        segmentation, kept = result["segmentation"], []
        for seg_info in result["segments_info"]:
            if seg_info["score"] >= thr_per_label[seg_info["label_id"]]:
                kept.append(seg_info)
            else:
                segmentation[segmentation == seg_info["id"]] = -1
        result["segments_info"] = kept

        return result