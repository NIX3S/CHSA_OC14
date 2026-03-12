import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

import torch
import mlflow
from datasets import load_dataset, Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    EarlyStoppingCallback,
)
from peft import LoraConfig, get_peft_model, TaskType
from trl import SFTTrainer, SFTConfig
from rouge_score import rouge_scorer

logging.basicConfig(level=logging.INFO, format="%(asctime)s — %(levelname)s — %(message)s")
log = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════════════
# Configuration
# ══════════════════════════════════════════════════════════════════════════════

BASE_MODEL     = "Qwen/Qwen3-1.7B-Base"
MAX_SEQ_LENGTH = 2048
OUTPUT_DIR     = "checkpoints/sft"

LORA_CONFIG = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
    bias="none",
)

DATA = {
    "train": "data/sft/train.jsonl",
    "val":   "data/sft/validation.jsonl",
    "test":  "data/sft/test.jsonl",
}

MLFLOW_EXPERIMENT = "CHSA-SFT-Qwen3-1.7B"
MLFLOW_URI        = "sqlite:///mlruns.db" 

DTYPE      = torch.float16 if torch.cuda.is_available() else torch.float32
DEVICE_MAP = "auto"        if torch.cuda.is_available() else "cpu"
USE_FP16   = torch.cuda.is_available()

SYSTEM_PROMPT = (
    "Tu es un agent de triage médical expert du Centre Hospitalier Saint-Aurélien (CHSA). "
    "Ton rôle est d'évaluer les symptômes des patients, de déterminer le niveau de priorité "
    "(urgence maximale / modérée / différée) et de fournir des recommandations cliniques claires. "
    "Réponds toujours de manière précise, bienveillante et en conformité avec les protocoles médicaux."
)


# ══════════════════════════════════════════════════════════════════════════════
# 1. Modèle + tokenizer 
# ══════════════════════════════════════════════════════════════════════════════

def load_model_and_tokenizer():
    log.info(f"Tokenizer : {BASE_MODEL}")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token    = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "right" 

    log.info(f"Modèle en {DTYPE} (sans QLoRA — bitsandbytes 0.44.1 incompatible triton 3.1)")
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        dtype=DTYPE,        
        device_map=DEVICE_MAP,
        trust_remote_code=True,
    )
    model.config.use_cache = False

    log.info("Application LoRA…")
    model = get_peft_model(model, LORA_CONFIG)
    trainable, total = model.get_nb_trainable_parameters()
    log.info(f"Paramètres entraînables : {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")
    return model, tokenizer


# ══════════════════════════════════════════════════════════════════════════════
# 2. Dataset SFT
# ══════════════════════════════════════════════════════════════════════════════

def format_chatml(example: Dict) -> str:
    """Format ChatML Qwen : system / user / assistant."""
    return (
        f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
        f"<|im_start|>user\n{example.get('instruction', '')}<|im_end|>\n"
        f"<|im_start|>assistant\n{example.get('response', '')}<|im_end|>"
    )


def load_splits() -> Dict[str, Optional[Dataset]]:
    result: Dict[str, Optional[Dataset]] = {}
    for name, path in DATA.items():
        if not Path(path).exists():
            log.warning(f"Absent : {path} → '{name}' ignoré")
            result[name] = None
            continue
        ds = load_dataset("json", data_files=path, split="train")
        log.info(f"  {name:6s}: {len(ds):,} exemples")
        result[name] = ds
    return result


# ══════════════════════════════════════════════════════════════════════════════
# 3. Évaluation ROUGE
# ══════════════════════════════════════════════════════════════════════════════

def compute_rouge(preds: List[str], refs: List[str]) -> Dict[str, float]:
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    sums = {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0}
    for p, r in zip(preds, refs):
        s = scorer.score(r, p)
        for k in sums:
            sums[k] += s[k].fmeasure
    n = len(preds)
    return {k: round(v / n, 4) for k, v in sums.items()}


def evaluate_model(model, tokenizer, ds: Dataset, n: int = 50) -> Dict:
    log.info(f"Évaluation sur {min(n, len(ds))} exemples…")
    model.eval()
    preds, refs = [], []
    device = next(model.parameters()).device

    for ex in ds.select(range(min(n, len(ds)))):
        prompt = (
            f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
            f"<|im_start|>user\n{ex['instruction']}<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )
        ids = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)
        ids = {k: v.to(device) for k, v in ids.items()}
        with torch.no_grad():
            out = model.generate(**ids, max_new_tokens=256, do_sample=False,
                                 pad_token_id=tokenizer.pad_token_id)
        gen = tokenizer.decode(out[0][ids["input_ids"].shape[1]:], skip_special_tokens=True)
        preds.append(gen.strip())
        refs.append(ex.get("response", ""))

    rouge  = compute_rouge(preds, refs)
    empty  = sum(1 for p in preds if len(p.strip()) < 10) / len(preds)
    return {**rouge, "hallucination_proxy": round(empty, 4), "n_evaluated": len(preds)}


# ══════════════════════════════════════════════════════════════════════════════
# 4. Entraînement
# ══════════════════════════════════════════════════════════════════════════════

def train():
    log.info("═" * 60)
    log.info("  CHSA — Semaine 2 : SFT + LoRA  [TRL 0.29 / fp16]")
    log.info(f"  GPU : {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
    log.info("═" * 60)

    mlflow.set_tracking_uri(MLFLOW_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT)

    with mlflow.start_run(run_name="sft_lora_fp16"):

        mlflow.log_params({
            "model":      BASE_MODEL,
            "lora_r":     LORA_CONFIG.r,
            "lora_alpha": LORA_CONFIG.lora_alpha,
            "epochs":     3,
            "lr":         2e-4,
            "dtype":      str(DTYPE),
        })

        model, tokenizer = load_model_and_tokenizer()

        log.info("Chargement données SFT…")
        splits   = load_splits()
        train_ds = splits.get("train")
        val_ds   = splits.get("val")
        test_ds  = splits.get("test")

        if train_ds is None:
            raise FileNotFoundError(
                "Dataset SFT 'train' introuvable.\n"
                "→ Lancez d'abord : python week1_data_pipeline.py"
            )

        has_val = val_ds is not None

        training_args = SFTConfig(

            output_dir=OUTPUT_DIR,
            num_train_epochs=3,
            per_device_train_batch_size=4,
            per_device_eval_batch_size=4,
            gradient_accumulation_steps=4,
            learning_rate=2e-4,
            lr_scheduler_type="cosine",
            warmup_steps=50,
            weight_decay=0.01,
            max_grad_norm=1.0,
            bf16=False,
            fp16=USE_FP16,
            optim="adamw_torch",
            logging_steps=10,
            eval_strategy="steps" if has_val else "no",
            eval_steps=100        if has_val else None,
            save_strategy="steps",
            save_steps=100,
            save_total_limit=3,
            load_best_model_at_end=has_val,
            metric_for_best_model="eval_loss" if has_val else None,
            greater_is_better=False           if has_val else None,
            report_to="mlflow",
            seed=42,
            dataloader_num_workers=0,
            remove_unused_columns=False,
            # SFT spécifiques 
            max_length=MAX_SEQ_LENGTH, 
            packing=False,
            dataset_text_field="text", 
        )

        callbacks = [EarlyStoppingCallback(early_stopping_patience=3)] if has_val else []


        def add_text_column(example):
            return {"text": format_chatml(example)}

        train_ds_fmt = train_ds.map(add_text_column, desc="Formatage train")
        val_ds_fmt   = val_ds.map(add_text_column, desc="Formatage val") if has_val else None

        callbacks = [EarlyStoppingCallback(early_stopping_patience=3)] if has_val else []

        trainer = SFTTrainer(
            model=model,
            args=training_args,
            train_dataset=train_ds_fmt,
            eval_dataset=val_ds_fmt,
            processing_class=tokenizer,
            callbacks=callbacks,
        )
        log.info("Démarrage SFT…")
        result = trainer.train()

        best = Path(OUTPUT_DIR) / "best_model"
        best.mkdir(parents=True, exist_ok=True)
        trainer.save_model(str(best))
        tokenizer.save_pretrained(str(best))
        log.info(f"Modèle sauvegardé : {best}")

        mlflow.log_metrics({
            "train_loss":      result.training_loss,
            "train_runtime_s": result.metrics.get("train_runtime", 0),
        })

        if test_ds is not None:
            metrics = evaluate_model(model, tokenizer, test_ds)
            mlflow.log_metrics(metrics)
            with open(Path(OUTPUT_DIR) / "test_metrics.json", "w") as f:
                json.dump(metrics, f, indent=2)
            log.info(f"Métriques : {metrics}")

        mlflow.log_artifacts(str(best), artifact_path="model")


if __name__ == "__main__":
    print("\n" + "─" * 60)
    print(f"  GPU    : {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'Aucun'}")
    if torch.cuda.is_available():
        vram = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"  VRAM   : {vram:.1f} GB  |  Requis : ~7 GB fp16")
    print(f"  Mode   : {'fp16 GPU' if torch.cuda.is_available() else 'fp32 CPU'}")
    print("─" * 60 + "\n")
    train()
