#  CHSA – Agent IA de Triage Médical

> ** POC (Proof of Concept) — Usage strictement expérimental**  
> Ce projet est une démonstration technique réalisée dans le cadre d'un POC de 4 semaines.  
> Il ne doit en aucun cas être utilisé en conditions réelles ou cliniques.

---

##  Présentation

Agent IA de triage médical développé pour le **Centre Hospitalier Saint-Aurélien (CHSA)**, basé sur le modèle **Qwen3-1.7B-Base** affiné par SFT+LoRA puis aligné par DPO. L'agent analyse les symptômes décrits en langage naturel et retourne un niveau de priorité clinique :

| Niveau | Critère |
|---|---|
| 🔴 `URGENCE_MAXIMALE` | Risque vital immédiat |
| 🟠 `URGENCE_MODEREE` | Consultation dans les 2h |
| 🟢 `URGENCE_DIFFEREE` | Consultation programmable |

---

##  Déploiement

| Ressource | Lien |
|---|---|
|  **Demo live** | [nix3s/chsa-triage-medical](https://huggingface.co/spaces/nix3s/chsa-triage-medical) |
|  **API Docs (Swagger)** | [nix3s-chsa-triage-medical.hf.space/docs](https://nix3s-chsa-triage-medical.hf.space/docs) |
|  **Modèle** | [nix3s/CHSA_Model](https://huggingface.co/nix3s/CHSA_Model) |
|  **Dataset** | [nix3s/CHSA_DAtaset](https://huggingface.co/datasets/nix3s/CHSA_DAtaset) |

---

##  Structure du projet

```
projet14/
├── notebooks/
│   ├── week1_data_pipeline.py      # Pipeline données SFT + DPO (RGPD)
│   ├── week2_sft_lora.py           # Fine-tuning SFT + LoRA (Qwen3-1.7B)
│   ├── week3_dpo_alignment.py      # Alignement DPO
│   ├── week4_api_fastapi.py        # API FastAPI + vLLM
│   └── week4_evaluation.py         # Benchmark et évaluation clinique
├── checkpoints/
│   ├── sft/best_model/             # Poids LoRA SFT
│   └── dpo/best_model/merged/      # Modèle fusionné DPO
├── data/
│   ├── sft/{train,validation,test}.jsonl
│   └── dpo/{train,validation,test}.jsonl
├── logs/
│   └── interactions.jsonl          # Traçabilité RGPD
├── Dockerfile
├── requirements.txt
└── .github/workflows/
    └── week4_github_actions.yml
```

---

##  Installation

### Prérequis

- Python 3.12+
- CUDA 12.1+ (RTX 4060 minimum — 8 GB VRAM)
- WSL2 ou Linux natif

### Setup

```bash
git clone https://github.com/<votre-repo>/chsa-triage-medical
cd chsa-triage-medical
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Versions clés

```
trl==0.29.0          transformers==4.56.2
torch==2.10.0        peft==0.13.0
vllm==0.17.1         fastapi==0.135.1
pydantic==2.12.5     mlflow==3.10.1
bitsandbytes==0.49.2 triton==3.6.0
```

---

##  Pipeline d'entraînement

### Semaine 1 — Données
```bash
python week1_data_pipeline.py
```
~6 900 paires SFT + ~109 000 paires DPO (FrenchMedMCQA, MediQA, UltraMedical)

### Semaine 2 — SFT + LoRA
```bash
python week2_sft_lora.py
```
Qwen3-1.7B-Base fp16, LoRA r=16/alpha=32 (~17M paramètres)

### Semaine 3 — Alignement DPO
```bash
python week3_dpo_alignment.py
```
beta=0.1 sur modèle SFT fusionné

### Semaine 4 — API + Évaluation
```bash
uvicorn week4_api_fastapi:app --host 0.0.0.0 --port 8000
python week4_evaluation.py --url http://localhost:8000 --n 100
```

---

##  API

### Endpoints

| Méthode | Endpoint | Description |
|---|---|---|
| `POST` | `/triage` | Analyse symptômes |
| `GET` | `/health` | Healthcheck |
| `GET` | `/metrics` | Métriques performance |
| `GET` | `/interactions` | Historique RGPD |

### Exemple d'usage
```bash
curl -X POST http://localhost:8000/triage \
  -H "Content-Type: application/json" \
  -d '{"symptoms": "douleur thoracique bras gauche sueurs", "age": 45}'
```

---

##  Résultats POC (12 Mars 2026)

###  Performance technique (5 requêtes)

| Métrique | Valeur | Cible | Statut |
|----------|--------|-------|--------|
| p95 | **18 387 ms** | < 2 000 ms | KO |
| p50 | **14 000 ms** | - |  |
| Moyenne | **15 116 ms** | - |  |
| Taux erreur | **0%** | 0% | ok |

###  Tests cliniques (10 cas)

| Case | Attendu | Prédit | Correct |
|------|---------|--------|---------|
| MAX_001 | MAX | **MAX** | OK |
| MAX_002 | MAX | **MAX** | OK |
| MAX_003 | MAX | **MAX** | OK |
| MAX_004 | MAX | **MAX** | OK |
| MOD_001 | MOD | **MOD** | OK |
| MOD_002 | MOD | MAX | KO |
| MOD_003 | MOD | MAX | KO |
| DIF_001 | DIF | **DIF** | OK |
| DIF_002 | DIF | **DIF** | OK |
| DIF_003 | DIF | **DIF** | OK |

###  Métriques détaillées

| Métrique | Valeur | Cible | Statut |
|----------|--------|-------|--------|
| **Accuracy** | **80%** (8/10) | ≥ 70% | OK |
| **Sous-triage** | **0%** | ≤ 10% | OK |
| **Sur-triage** | **20%** | ≤ 30% | OK |

**Par niveau d'urgence** :
- **URGENCE_MAXIMALE** : 100% (4/4)
- **URGENCE_MODEREE** : 33% (1/3)   
- **URGENCE_DIFFEREE** : 100% (3/3)

### 🎯 Go/No-Go

| Critère | Cible | Résultat | Statut |
|---------|-------|----------|--------|
| Latence p95 | < 2s | 18.4s | KO |
| Accuracy | ≥ 70% | 80% | OK |
| Sous-triage | ≤ 10% | 0% | OK |
| Traçabilité RGPD | OK | OK | OK |

> **Décision : GO**  
> Modèle **safe by design** (zéro sous-triage critique) mais latence insuffisantes

---

## Corrections techniques appliquées

| Problème | Solution |
|----------|----------|
| `triton.ops` absent | fp16 + `bitsandbytes>=0.45.5` |
| TRL 0.29 `max_seq_length` | `max_length` dans `SFTConfig` |
| Pydantic V2 `@validator` | `@field_validator(mode="before")` |
| MLflow 2026 deprecated | `sqlite:///mlruns.db` |

---

## Docker

```bash
docker build -t chsa-triage .
docker run --gpus all -p 8000:8000 \
  -e MODEL_PATH=nix3s/CHSA_Model \
  chsa-triage
```

---

## Conformité RGPD

Symptômes bruts **jamais persistés**  
Métadonnées anonymisées (latence, urgence, `patient_id` client-side)  
Logs traçables (`interactions.jsonl`)

---

##  Recommandations techniques

1. **Latence** : INT8 quantization + vLLM batching dynamique
2. **Accuracy** : Dataset FR clinique > 50k exemples validés urgentistes
3. **Modèle** : Qwen3-7B-Instruct ou Mistral-7B-Medical
4. **DPO** : Pondération loss par criticité des cas
5. **Validation** : Testset n > 100 cas, métriques par niveau d'urgence

---

##  Licence

POC à usage interne CHSA. Non destiné à la production clinique.

---

*Développé par Paul Lesage — IA Engineer Junior CHSA — Mars 2026*
