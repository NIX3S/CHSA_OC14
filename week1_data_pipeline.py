
# ── Imports ──────────────────────────────────────────────────────────────────
import os
import re
import json
import warnings
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
from tqdm import tqdm

from datasets import load_dataset, concatenate_datasets, Dataset, DatasetDict
from sklearn.model_selection import train_test_split

# Presidio
from presidio_analyzer import AnalyzerEngine, PatternRecognizer, Pattern
from presidio_anonymizer import AnonymizerEngine
from presidio_anonymizer.entities import OperatorConfig
from presidio_analyzer.nlp_engine import NlpEngineProvider
import os
os.environ["SPACY_FORCE_CPU"] = "true"
warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s — %(levelname)s — %(message)s")
log = logging.getLogger(__name__)

# ── Configuration ─────────────────────────────────────────────────────────────
CONFIG = {
    "random_seed": 42,
    "sft_target_size": 5000,
    "val_ratio": 0.10,
    "test_ratio": 0.10,
    "output_dir": "data",
    "audit_dir": "audit",
    "datasets": {
        "FrenchMedMCQA": {
            "path": "../datasets/FrenchMedMCQA/data",
            "format": "parquet",
            "language": "fr",
        },
        "MediQA": {
            "path": "../datasets/MediQAl/mcqm",
            "format": "json",
            "language": "en",
        },
        "UltraMedical": {
            "path": "../datasets/UltraMedical-Preference/data",
            "format": "json",
            "language": "en",
            "role": "dpo",
        },
    },
}

# ── Schéma des métadonnées ────────────────────────────────────────────────────
METADATA_SCHEMA = {
    "id": "str — identifiant unique de l'exemple",
    "instruction": "str — question / symptôme soumis",
    "response": "str — réponse médicale validée (SFT) ou vide (DPO)",
    "chosen": "str — réponse préférée (DPO uniquement)",
    "rejected": "str — réponse rejetée (DPO uniquement)",
    "source": "str — dataset d'origine (FrenchMedMCQA | MediQA | UltraMedical)",
    "language": "str — fr | en",
    "split": "str — train | validation | test",
    "confidence": "float — 1.0 si gold label, 0.8 si inféré",
    "pii_detected": "bool — vrai si des PII ont été détectées et masquées",
    "anonymized_at": "str — ISO 8601 timestamp de l'anonymisation",
    "gdpr_version": "str — version du processus RGPD appliqué",
}

GDPR_VERSION = "1.0.0"


# ══════════════════════════════════════════════════════════════════════════════
# 1. Moteur Presidio
# ══════════════════════════════════════════════════════════════════════════════

def build_presidio_engines() -> Tuple[AnalyzerEngine, AnonymizerEngine]:
    """Instancie les moteurs Presidio avec spaCy (fr + en)."""
    log.info("Chargement des modèles spaCy (fr_core_news_lg + en_core_web_lg)…")
    provider = NlpEngineProvider(nlp_configuration={
        "nlp_engine_name": "spacy",
        "models": [
            {"lang_code": "en", "model_name": "en_core_web_lg"},
            {"lang_code": "fr", "model_name": "fr_core_news_lg"},
        ],
    })
    nlp_engine = provider.create_engine()
    analyzer = AnalyzerEngine(nlp_engine=nlp_engine, supported_languages=["en", "fr"])
    anonymizer = AnonymizerEngine()
    log.info("Presidio prêt ✓")
    return analyzer, anonymizer


ANALYZER, ANONYMIZER = build_presidio_engines()


def get_operators() -> Dict[str, OperatorConfig]:
    return {
        "DEFAULT":        OperatorConfig("replace", {"new_value": "<PII>"}),
        "PERSON":         OperatorConfig("replace", {"new_value": "<PERSON>"}),
        "PHONE_NUMBER":   OperatorConfig("replace", {"new_value": "<PHONE>"}),
        "DATE_TIME":      OperatorConfig("replace", {"new_value": "<DATE>"}),
        "EMAIL_ADDRESS":  OperatorConfig("replace", {"new_value": "<EMAIL>"}),
        "LOCATION":       OperatorConfig("replace", {"new_value": "<LOCATION>"}),
    }


# ══════════════════════════════════════════════════════════════════════════════
# 2. Détection de langue
# ══════════════════════════════════════════════════════════════════════════════

_FR_MARKERS = re.compile(
    r"\b(hôpital|médecin|chu|infirmier|douleur|fièvre|traitement|"
    r"symptôme|antécédent|ordonnance|urgences|consultation)\b",
    re.IGNORECASE,
)
_FR_CHARS = re.compile(r"[àâäéèêëîïôöùûüÿç]")


def detect_language(text: str) -> str:
    """Détecte fr/en par règles rapides (pas de dépendance externe)."""
    if not text:
        return "en"
    score = len(_FR_MARKERS.findall(text)) * 2 + len(_FR_CHARS.findall(text))
    return "fr" if score >= 2 else "en"


# ══════════════════════════════════════════════════════════════════════════════
# 3. Anonymisation unifiée
# ══════════════════════════════════════════════════════════════════════════════

# Patterns Regex couvrant FR + EN
_REGEX_PATTERNS: List[Tuple[str, re.Pattern]] = [
    # Hôpitaux FR
    ("<HOSPITAL>", re.compile(r"(?i)\b(?:Hôpital|CHU|CH|CHRU|AP-HP)\s+[\w\-']+(?:\s+[\w\-']+)?")),
    # Hôpitaux EN
    ("<HOSPITAL>", re.compile(r"(?i)\b(?:Mount Sinai|Mayo Clinic|Johns Hopkins|Hospital|Clinic|Medical Center)\b")),
    # Médecins FR / EN
    ("<DOCTOR>",   re.compile(r"(?i)\b(?:Dr\.?|Docteur|Prof\.?|Pr\.?)\s+[A-ZÀÁÂÃÄÅÆÇÈÉÊËÌÍÎÏÑÒÓÔÕÖÙÚÛÜ][a-zàáâãäåæçèéêëìíîïñòóôõöùúûüý]+")),
    # Téléphones FR
    ("<PHONE>",    re.compile(r"\b0\d(?:[\s\-\.]?\d{2}){4}\b")),
    # Téléphones US
    ("<PHONE>",    re.compile(r"\b(?:\(\d{3}\)\s?|\d{3}[-.])\d{3}[-\.]\d{4}\b")),
    # Emails
    ("<EMAIL>",    re.compile(r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b")),
    # URLs
    ("<URL>",      re.compile(r"https?://\S+")),
    # Numéros SS (FR)
    ("<SS_NUM>",   re.compile(r"\b[12]\s?\d{2}\s?\d{2}\s?\d{2}\s?\d{3}\s?\d{3}\s?\d{2}\b")),
]


class AuditLog:
    """Journal RGPD horodaté pour chaque modification."""

    def __init__(self):
        self.entries: List[Dict] = []

    def record(self, row_id: str, field: str, entity_type: str,
               original_snippet: str, replacement: str):
        self.entries.append({
            "row_id": row_id,
            "field": field,
            "entity_type": entity_type,
            "original_snippet": original_snippet[:80],
            "replacement": replacement,
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "gdpr_version": GDPR_VERSION,
        })

    def save(self, path: str):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.entries, f, ensure_ascii=False, indent=2)
        log.info(f"Audit RGPD sauvegardé : {path} ({len(self.entries)} entrées)")


def anonymize_text(text: str, lang: str, row_id: str,
                   field: str, audit: AuditLog) -> Tuple[str, bool]:
    """
    Anonymise un texte en 2 passes :
      1. Regex (rapide, déterministe)
      2. Presidio (NLP, entités résiduelles)
    Retourne (texte_anonymisé, pii_found).
    """
    if not text or not isinstance(text, str):
        return text, False

    result = text
    pii_found = False

    # Passe 1 — Regex
    for placeholder, pattern in _REGEX_PATTERNS:
        matches = pattern.findall(result)
        if matches:
            for m in matches:
                audit.record(row_id, field, placeholder, m, placeholder)
            result = pattern.sub(placeholder, result)
            pii_found = True

    # Passe 2 — Presidio (PERSON, PHONE, DATE, EMAIL résiduels)
    try:
        detections = ANALYZER.analyze(result, language=lang)
        if detections:
            anon = ANONYMIZER.anonymize(result, detections, get_operators())
            if anon.text != result:
                for d in detections:
                    snippet = result[d.start:d.end]
                    placeholder = f"<{d.entity_type}>"
                    audit.record(row_id, field, d.entity_type, snippet, placeholder)
                result = anon.text
                pii_found = True
    except Exception as e:
        log.warning(f"Presidio error on row {row_id}: {e}")

    return result.strip(), pii_found


# ══════════════════════════════════════════════════════════════════════════════
# 4. Formatage SFT
# ══════════════════════════════════════════════════════════════════════════════

_ANSWER_KEYS = ["answer_a", "answer_b", "answer_c", "answer_d", "answer_e"]


def format_frenchmedmcqa(example: Dict, idx: int) -> Optional[Dict]:
    """Formate un exemple FrenchMedMCQA en paire instruction/réponse."""
    try:
        answers = example.get("correct_answers", [])
        if not answers:
            return None
        correct_idx = answers[0] - 1  # 1-indexed → 0-indexed
        if not (0 <= correct_idx < len(_ANSWER_KEYS)):
            return None
        answer_key = _ANSWER_KEYS[correct_idx]
        response = example.get(answer_key, "")
        if not response:
            return None

        question = example.get("question", "")

        # Prompt de triage médical adaptatif
        instruction = (
            f"En tant qu'agent de triage médical, analysez la question suivante "
            f"et fournissez une réponse clinique précise.\n\n"
            f"Question : {question}"
        )

        return {
            "id": f"frenchmed_{idx}",
            "instruction": instruction,
            "response": response,
            "source": "FrenchMedMCQA",
            "language": "fr",
            "confidence": 1.0,
        }
    except Exception as e:
        log.debug(f"FrenchMedMCQA format error idx={idx}: {e}")
        return None


def format_mediqa(example: Dict, idx: int) -> Optional[Dict]:
    """Formate un exemple MediQA en paire instruction/réponse."""
    try:
        correct_str = example.get("correct_answers", "A")
        correct = correct_str.split(",")[0].strip().lower()
        answer_key = f"answer_{correct}"
        response = example.get(answer_key, "")
        if not response:
            return None

        question = example.get("question", "")
        instruction = (
            f"As a medical triage agent, analyze the following clinical question "
            f"and provide an evidence-based answer.\n\n"
            f"Question: {question}"
        )

        return {
            "id": f"mediqa_{idx}",
            "instruction": instruction,
            "response": response,
            "source": "MediQA",
            "language": "en",
            "confidence": 1.0,
        }
    except Exception as e:
        log.debug(f"MediQA format error idx={idx}: {e}")
        return None


def format_dpo(example: Dict, idx: int) -> Optional[Dict]:
    """Formate un exemple UltraMedical en paire DPO (chosen/rejected)."""
    try:
        chosen_msgs = example.get("chosen", [])
        rejected_msgs = example.get("rejected", [])

        # Cherche le dernier message assistant
        def get_assistant_msg(msgs):
            for m in reversed(msgs):
                if isinstance(m, dict) and m.get("role") == "assistant":
                    return m.get("content", "")
            # fallback : dernier message
            return msgs[-1].get("content", "") if msgs else ""

        chosen = get_assistant_msg(chosen_msgs)
        rejected = get_assistant_msg(rejected_msgs)

        if not chosen or not rejected or chosen == rejected:
            return None

        prompt = example.get("prompt", "")
        if isinstance(prompt, list):
            prompt = prompt[-1].get("content", "") if prompt else ""

        return {
            "id": f"ultramed_{idx}",
            "instruction": prompt,
            "chosen": chosen,
            "rejected": rejected,
            "source": "UltraMedical",
            "language": "en",
            "confidence": 0.9,
        }
    except Exception as e:
        log.debug(f"UltraMedical format error idx={idx}: {e}")
        return None


# ══════════════════════════════════════════════════════════════════════════════
# 5. Chargement des datasets
# ══════════════════════════════════════════════════════════════════════════════

def load_frenchmedmcqa(path: str) -> DatasetDict:
    return load_dataset("parquet", data_files={
        "train":      f"{path}/train-00000-of-00001.parquet",
        "validation": f"{path}/validation-00000-of-00001.parquet",
        "test":       f"{path}/test-00000-of-00001.parquet",
    })


def load_mediqa(path: str) -> DatasetDict:
    return load_dataset("json", data_files={
        "train":      f"{path}/train.json",
        "validation": f"{path}/validation.json",
        "test":       f"{path}/test.json",
    })


def load_ultramedical(path: str) -> DatasetDict:
    return load_dataset("json", data_files={
        "train":      f"{path}/train.json",
        "validation": f"{path}/dev.json",
        "test":       f"{path}/test.json",
    })


# ══════════════════════════════════════════════════════════════════════════════
# 6. Pipeline principal
# ══════════════════════════════════════════════════════════════════════════════

def process_sft_split(
    french_split,
    mediqa_split,
    split_name: str,
    audit: AuditLog,
    output_dir: str,
) -> List[Dict]:
    """Traite un split SFT complet : format + anonymisation + export."""

    records: List[Dict] = []

    # FrenchMedMCQA
    for idx, ex in enumerate(tqdm(french_split, desc=f"  FrenchMed {split_name}", leave=False)):
        formatted = format_frenchmedmcqa(ex, idx)
        if formatted is None:
            continue
        lang = formatted["language"]
        for field in ["instruction", "response"]:
            formatted[field], pii = anonymize_text(
                formatted[field], lang, formatted["id"], field, audit
            )
        formatted["pii_detected"] = pii
        formatted["anonymized_at"] = datetime.utcnow().isoformat() + "Z"
        formatted["gdpr_version"] = GDPR_VERSION
        formatted["split"] = split_name
        records.append(formatted)

    # MediQA
    for idx, ex in enumerate(tqdm(mediqa_split, desc=f"  MediQA {split_name}", leave=False)):
        formatted = format_mediqa(ex, idx)
        if formatted is None:
            continue
        lang = formatted["language"]
        for field in ["instruction", "response"]:
            formatted[field], pii = anonymize_text(
                formatted[field], lang, formatted["id"], field, audit
            )
        formatted["pii_detected"] = pii
        formatted["anonymized_at"] = datetime.utcnow().isoformat() + "Z"
        formatted["gdpr_version"] = GDPR_VERSION
        formatted["split"] = split_name
        records.append(formatted)

    # Export
    out_path = Path(output_dir) / "sft" / f"{split_name}.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    log.info(f"SFT {split_name}: {len(records)} exemples → {out_path}")
    return records


def process_dpo_split(
    ultra_split,
    split_name: str,
    audit: AuditLog,
    output_dir: str,
) -> List[Dict]:
    """Traite un split DPO complet : format + anonymisation + export."""

    records: List[Dict] = []

    for idx, ex in enumerate(tqdm(ultra_split, desc=f"  UltraMed {split_name}", leave=False)):
        formatted = format_dpo(ex, idx)
        if formatted is None:
            continue
        lang = formatted["language"]
        for field in ["instruction", "chosen", "rejected"]:
            formatted[field], pii = anonymize_text(
                formatted[field], lang, formatted["id"], field, audit
            )
        formatted["pii_detected"] = pii
        formatted["anonymized_at"] = datetime.utcnow().isoformat() + "Z"
        formatted["gdpr_version"] = GDPR_VERSION
        formatted["split"] = split_name
        records.append(formatted)

    out_path = Path(output_dir) / "dpo" / f"{split_name}.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    log.info(f"DPO {split_name}: {len(records)} exemples → {out_path}")
    return records


# ══════════════════════════════════════════════════════════════════════════════
# 7. Validation et statistiques finales
# ══════════════════════════════════════════════════════════════════════════════

def validate_and_stats(records: List[Dict], name: str):
    """Affiche des statistiques de validation sur le dataset."""
    if not records:
        log.warning(f"{name}: dataset vide !")
        return

    df = pd.DataFrame(records)
    print(f"\n{'='*60}")
    print(f"  {name} — Statistiques")
    print(f"{'='*60}")
    print(f"  Exemples total      : {len(df)}")

    if "language" in df.columns:
        print(f"  Langues             : {df['language'].value_counts().to_dict()}")

    if "source" in df.columns:
        print(f"  Sources             : {df['source'].value_counts().to_dict()}")

    if "pii_detected" in df.columns:
        n_pii = df["pii_detected"].sum()
        print(f"  PII détectés        : {n_pii} exemples ({100*n_pii/len(df):.1f}%)")

    # Longueurs moyennes
    for col in ["instruction", "response", "chosen"]:
        if col in df.columns:
            avg = df[col].dropna().apply(len).mean()
            print(f"  Longueur moy. [{col}] : {avg:.0f} chars")

    # Exemple aléatoire
    sample = df.sample(1, random_state=42).iloc[0]
    print(f"\n  Exemple aléatoire :")
    print(f"  instruction : {str(sample.get('instruction',''))[:120]}…")
    if "response" in sample:
        print(f"  response    : {str(sample.get('response',''))[:120]}…")
    print(f"{'='*60}\n")


# ══════════════════════════════════════════════════════════════════════════════
# 8. Export du schéma RGPD
# ══════════════════════════════════════════════════════════════════════════════

def export_metadata_schema(output_dir: str):
    schema = {
        "version": GDPR_VERSION,
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "schema": METADATA_SCHEMA,
        "gdpr_compliance": {
            "anonymization_tool": "Presidio + Regex",
            "models_used": ["en_core_web_lg", "fr_core_news_lg"],
            "entities_masked": [
                "PERSON", "PHONE_NUMBER", "DATE_TIME", "EMAIL_ADDRESS",
                "LOCATION", "HOSPITAL", "DOCTOR", "SS_NUM",
            ],
            "strategy": "replace",
            "audit_trail": "audit/sft_audit.json + audit/dpo_audit.json",
            "data_minimization": "Seuls les champs nécessaires au fine-tuning sont conservés",
            "retention_policy": "Données conservées pour la durée du POC (4 semaines)",
        },
    }
    path = Path(output_dir) / "metadata_schema.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(schema, f, ensure_ascii=False, indent=2)
    log.info(f"Schéma RGPD exporté : {path}")


# ══════════════════════════════════════════════════════════════════════════════
# 9. MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    log.info("═" * 60)
    log.info("  CHSA — Semaine 1 : Pipeline de données")
    log.info("═" * 60)

    output_dir = CONFIG["output_dir"]
    audit_dir  = CONFIG["audit_dir"]

    # ── Chargement des datasets ───────────────────────────────────────────────
    log.info("Chargement des datasets…")
    cfg = CONFIG["datasets"]

    french_med  = load_frenchmedmcqa(cfg["FrenchMedMCQA"]["path"])
    mediqa      = load_mediqa(cfg["MediQA"]["path"])
    ultra_pref  = load_ultramedical(cfg["UltraMedical"]["path"])

    log.info(f"FrenchMedMCQA : {len(french_med['train'])} train")
    log.info(f"MediQA        : {len(mediqa['train'])} train")
    log.info(f"UltraMedical  : {len(ultra_pref['train'])} train")

    # ── SFT ──────────────────────────────────────────────────────────────────
    log.info("\n─── SFT Dataset ───")
    audit_sft = AuditLog()
    all_sft: List[Dict] = []

    for split in ["train", "validation", "test"]:
        records = process_sft_split(
            french_med[split],
            mediqa[split],
            split,
            audit_sft,
            output_dir,
        )
        all_sft.extend(records)

    audit_sft.save(f"{audit_dir}/sft_audit.json")
    validate_and_stats(all_sft, "SFT Global")

    # Vérification taille cible
    train_sft = [r for r in all_sft if r["split"] == "train"]
    if len(train_sft) < CONFIG["sft_target_size"]:
        log.warning(
            f"⚠ Dataset SFT train ({len(train_sft)}) < cible ({CONFIG['sft_target_size']}). "
            f"Augmentez les sources ou appliquez de la data augmentation."
        )
    else:
        log.info(f"✓ SFT train : {len(train_sft)} exemples (cible : {CONFIG['sft_target_size']})")

    # ── DPO ──────────────────────────────────────────────────────────────────
    log.info("\n─── DPO Dataset ───")
    audit_dpo = AuditLog()
    all_dpo: List[Dict] = []

    for split in ["train", "validation", "test"]:
        records = process_dpo_split(
            ultra_pref[split],
            split,
            audit_dpo,
            output_dir,
        )
        all_dpo.extend(records)

    audit_dpo.save(f"{audit_dir}/dpo_audit.json")
    validate_and_stats(all_dpo, "DPO Global")

    # ── Schéma RGPD ──────────────────────────────────────────────────────────
    export_metadata_schema(output_dir)

    # ── Résumé final ─────────────────────────────────────────────────────────
    print("\n" + "═" * 60)
    print("  RÉSUMÉ FINAL — Semaine 1")
    print("═" * 60)
    print(f"  SFT exemples  : {len(all_sft)}")
    print(f"  DPO exemples  : {len(all_dpo)}")
    print(f"  Audit SFT     : {len(audit_sft.entries)} modifications RGPD")
    print(f"  Audit DPO     : {len(audit_dpo.entries)} modifications RGPD")
    print(f"  Outputs       : {output_dir}/{{sft,dpo}}/{{train,validation,test}}.jsonl")
    print(f"  Schéma RGPD   : {output_dir}/metadata_schema.json")
    print("═" * 60)
    print("  ✅ Semaine 1 terminée — Données prêtes pour le fine-tuning")
    print("═" * 60 + "\n")


if __name__ == "__main__":
    main()
