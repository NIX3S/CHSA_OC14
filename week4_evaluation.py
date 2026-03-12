import argparse
import asyncio
import json
import logging
import time
from datetime import datetime
from pathlib import Path
from statistics import mean, median, stdev
from typing import Dict, List, Optional

import httpx
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s — %(levelname)s — %(message)s")
log = logging.getLogger(__name__)

# ══════════════════════════════════════════════════════════════════════════════
# Cas de test cliniques (ground truth médicale)
# ══════════════════════════════════════════════════════════════════════════════

CLINICAL_TEST_CASES = [
    # ── Urgences maximales ────────────────────────────────────────────────────
    {
        "id": "MAX_001",
        "symptoms": "Patient de 62 ans, douleur thoracique intense irradiant vers le bras gauche, sueurs profuses, nausées, essoufflement depuis 20 minutes",
        "age": 62,
        "constantes": {"fc": 115, "ta": "160/95", "spo2": 91},
        "expected_urgence": "URGENCE_MAXIMALE",
        "clinical_rationale": "Suspicion SCA (syndrome coronarien aigu) — pronostic vital immédiat",
    },
    {
        "id": "MAX_002",
        "symptoms": "Femme de 35 ans, céphalée brutale en coup de tonnerre, raideur de nuque, fièvre à 40°C",
        "age": 35,
        "constantes": {"fc": 108, "ta": "145/85", "temperature": 40.2},
        "expected_urgence": "URGENCE_MAXIMALE",
        "clinical_rationale": "Suspicion méningite bactérienne / hémorragie sous-arachnoïdienne",
    },
    {
        "id": "MAX_003",
        "symptoms": "Homme de 45 ans, perte de conscience brève, confusion, hémiplégie gauche brutale, trouble du langage",
        "age": 45,
        "expected_urgence": "URGENCE_MAXIMALE",
        "clinical_rationale": "Suspicion AVC ischémique — fenêtre thérapeutique thrombolyse",
    },
    {
        "id": "MAX_004",
        "symptoms": "Enfant de 3 ans, difficultés respiratoires sévères, stridor inspiratoire, cyanose des lèvres",
        "age": 3,
        "constantes": {"fc": 145, "spo2": 88},
        "expected_urgence": "URGENCE_MAXIMALE",
        "clinical_rationale": "Détresse respiratoire sévère pédiatrique",
    },

    # ── Urgences modérées ─────────────────────────────────────────────────────
    {
        "id": "MOD_001",
        "symptoms": "Patient de 28 ans, douleur abdominale droite modérée depuis 6h, légère fièvre à 38.2°C, nausées sans vomissements",
        "age": 28,
        "constantes": {"fc": 88, "ta": "125/75", "temperature": 38.2},
        "expected_urgence": "URGENCE_MODEREE",
        "clinical_rationale": "Suspicion appendicite débutante — surveillance et bilan biologique urgent",
    },
    {
        "id": "MOD_002",
        "symptoms": "Femme de 25 ans, brûlures mictionnelles intenses, pollakiurie, douleurs lombaires, fièvre à 38.8°C",
        "age": 25,
        "constantes": {"temperature": 38.8},
        "expected_urgence": "URGENCE_MODEREE",
        "clinical_rationale": "Pyélonéphrite probable — antibiothérapie urgente après ECBU",
    },
    {
        "id": "MOD_003",
        "symptoms": "Homme de 55 ans, asthme connu, dyspnée modérée au repos, sibilances bilatérales, n'a pas son inhalateur",
        "age": 55,
        "antecedents": "Asthme depuis 20 ans, tabagisme actif",
        "expected_urgence": "URGENCE_MODEREE",
        "clinical_rationale": "Exacerbation asthmatique modérée — bronchodilatateur et surveillance",
    },

    # ── Urgences différées ────────────────────────────────────────────────────
    {
        "id": "DIF_001",
        "symptoms": "Patient de 32 ans, rhume depuis 3 jours, légère rhinorrhée, éternuements, pas de fièvre, état général conservé",
        "age": 32,
        "expected_urgence": "URGENCE_DIFFEREE",
        "clinical_rationale": "Rhinopharyngite virale commune — traitement symptomatique",
    },
    {
        "id": "DIF_002",
        "symptoms": "Femme de 40 ans, douleur lombaire mécanique chronique, sans irradiation, aggravée en fin de journée, antécédent de lombalgie",
        "age": 40,
        "antecedents": "Lombalgie chronique, sans déficit neurologique",
        "expected_urgence": "URGENCE_DIFFEREE",
        "clinical_rationale": "Lombalgie commune — consultation programmée kinésithérapie/médecin traitant",
    },
    {
        "id": "DIF_003",
        "symptoms": "Patient de 22 ans, mal de gorge depuis 2 jours, légère fièvre à 37.8°C, sans dysphagie sévère",
        "age": 22,
        "constantes": {"temperature": 37.8},
        "expected_urgence": "URGENCE_DIFFEREE",
        "clinical_rationale": "Angine probable — test rapide TROD possible en médecine de ville",
    },
]


# ══════════════════════════════════════════════════════════════════════════════
# Tests de latence
# ══════════════════════════════════════════════════════════════════════════════

async def single_request(client: httpx.AsyncClient, base_url: str, payload: Dict) -> Dict:
    """Exécute une requête et mesure la latence."""
    start = time.perf_counter()
    try:
        response = await client.post(f"{base_url}/triage", json=payload, timeout=30.0)
        latency_ms = (time.perf_counter() - start) * 1000
        return {
            "status_code": response.status_code,
            "latency_ms": latency_ms,
            "response": response.json() if response.status_code == 200 else None,
            "error": None,
        }
    except Exception as e:
        latency_ms = (time.perf_counter() - start) * 1000
        return {
            "status_code": 0,
            "latency_ms": latency_ms,
            "response": None,
            "error": str(e),
        }


async def latency_benchmark(base_url: str, n_requests: int = 100, concurrency: int = 5) -> Dict:
    """
    Benchmark de latence avec requêtes concurrentes.
    Retourne p50, p95, p99, min, max, avg.
    """
    log.info(f"Benchmark latence : {n_requests} requêtes, concurrence={concurrency}")

    # Payload de test standard
    payload = {
        "symptoms": "Patient de 45 ans, douleur thoracique modérée, légère dyspnée d'effort",
        "age": 45,
        "constantes": {"fc": 88, "ta": "130/80", "spo2": 97},
    }

    latencies = []
    errors = 0

    async with httpx.AsyncClient() as client:
        # Envoi en vagues de `concurrency` requêtes
        for batch_start in range(0, n_requests, concurrency):
            batch_size = min(concurrency, n_requests - batch_start)
            tasks = [single_request(client, base_url, payload) for _ in range(batch_size)]
            results = await asyncio.gather(*tasks)

            for r in results:
                if r["error"] or r["status_code"] != 200:
                    errors += 1
                else:
                    latencies.append(r["latency_ms"])

            log.info(f"  Progression : {batch_start + batch_size}/{n_requests} requêtes")

    if not latencies:
        log.error("Aucune requête réussie — vérifiez que l'API est démarrée")
        return {}

    latencies_arr = sorted(latencies)
    n = len(latencies_arr)

    metrics = {
        "n_requests": n_requests,
        "n_success": n,
        "n_errors": errors,
        "error_rate": errors / n_requests,
        "latency_ms": {
            "min":  round(min(latencies_arr), 2),
            "p50":  round(latencies_arr[int(n * 0.50)], 2),
            "p75":  round(latencies_arr[int(n * 0.75)], 2),
            "p95":  round(latencies_arr[int(n * 0.95)], 2),
            "p99":  round(latencies_arr[min(int(n * 0.99), n-1)], 2),
            "max":  round(max(latencies_arr), 2),
            "mean": round(mean(latencies_arr), 2),
            "std":  round(stdev(latencies_arr) if n > 1 else 0, 2),
        },
    }

    log.info(f"Latence p50={metrics['latency_ms']['p50']}ms | "
             f"p95={metrics['latency_ms']['p95']}ms | "
             f"p99={metrics['latency_ms']['p99']}ms")

    return metrics


# ══════════════════════════════════════════════════════════════════════════════
# Tests cliniques (pertinence + sécurité)
# ══════════════════════════════════════════════════════════════════════════════

async def run_clinical_tests(base_url: str) -> Dict:
    """
    Exécute tous les cas cliniques et compare avec la ground truth.
    Calcule : accuracy, taux de sur-triage, taux de sous-triage.
    """
    log.info(f"Tests cliniques : {len(CLINICAL_TEST_CASES)} cas")
    results = []

    async with httpx.AsyncClient() as client:
        for case in CLINICAL_TEST_CASES:
            payload = {
                "symptoms": case["symptoms"],
                "age": case.get("age"),
                "antecedents": case.get("antecedents"),
                "constantes": case.get("constantes"),
            }

            result = await single_request(client, base_url, payload)

            if result["response"]:
                predicted = result["response"].get("niveau_urgence", "INDETERMINE")
                expected = case["expected_urgence"]
                correct = predicted == expected

                # Sous-triage = plus grave prédit comme moins grave (dangereux !)
                urgence_rank = {"URGENCE_MAXIMALE": 3, "URGENCE_MODEREE": 2, "URGENCE_DIFFEREE": 1, "INDETERMINE": 0}
                pred_rank = urgence_rank.get(predicted, 0)
                exp_rank = urgence_rank.get(expected, 0)
                under_triage = pred_rank < exp_rank
                over_triage  = pred_rank > exp_rank

                results.append({
                    "case_id": case["id"],
                    "expected": expected,
                    "predicted": predicted,
                    "correct": correct,
                    "under_triage": under_triage,  # Critique !
                    "over_triage": over_triage,
                    "latency_ms": round(result["latency_ms"], 2),
                    "justification": result["response"].get("justification", "")[:200],
                    "clinical_rationale": case["clinical_rationale"],
                })
            else:
                results.append({
                    "case_id": case["id"],
                    "expected": case["expected_urgence"],
                    "predicted": "ERROR",
                    "correct": False,
                    "under_triage": True,   # Erreur = sous-triage par défaut
                    "over_triage": False,
                    "latency_ms": round(result["latency_ms"], 2),
                    "error": result.get("error", "Unknown"),
                    "clinical_rationale": case["clinical_rationale"],
                })

    # Statistiques globales
    n = len(results)
    n_correct     = sum(1 for r in results if r["correct"])
    n_under       = sum(1 for r in results if r.get("under_triage", False))
    n_over        = sum(1 for r in results if r.get("over_triage", False))

    # Par niveau d'urgence
    by_level = {}
    for level in ["URGENCE_MAXIMALE", "URGENCE_MODEREE", "URGENCE_DIFFEREE"]:
        level_cases = [r for r in results if r["expected"] == level]
        correct_level = sum(1 for r in level_cases if r["correct"])
        by_level[level] = {
            "total": len(level_cases),
            "correct": correct_level,
            "accuracy": round(correct_level / len(level_cases), 3) if level_cases else 0,
        }

    summary = {
        "total_cases": n,
        "accuracy": round(n_correct / n, 3),
        "under_triage_rate": round(n_under / n, 3),   # ← Le plus critique
        "over_triage_rate":  round(n_over / n, 3),
        "by_level": by_level,
        "cases": results,
    }

    log.info(f"Accuracy globale : {summary['accuracy']:.1%}")
    log.info(f"Sous-triage (dangereux) : {summary['under_triage_rate']:.1%}")
    log.info(f"Sur-triage : {summary['over_triage_rate']:.1%}")

    return summary


# ══════════════════════════════════════════════════════════════════════════════
# Tests de traçabilité
# ══════════════════════════════════════════════════════════════════════════════

async def test_traceability(base_url: str) -> Dict:
    """Vérifie que le journal de traçabilité est bien alimenté."""
    async with httpx.AsyncClient() as client:
        # Requête de test
        r1 = await client.post(f"{base_url}/triage", json={
            "symptoms": "Test traçabilité — douleur légère",
            "patient_id": "TEST_TRACE_001"
        }, timeout=15.0)

        # Récupère les interactions
        r2 = await client.get(f"{base_url}/interactions?limit=5", timeout=10.0)

        traceability_ok = False
        if r2.status_code == 200:
            interactions = r2.json().get("interactions", [])
            traceability_ok = len(interactions) > 0

        return {
            "triage_status": r1.status_code,
            "traceability_endpoint_status": r2.status_code,
            "interactions_logged": traceability_ok,
            "n_logged": r2.json().get("total", 0) if r2.status_code == 200 else 0,
        }


# ══════════════════════════════════════════════════════════════════════════════
# Génération du rapport final
# ══════════════════════════════════════════════════════════════════════════════

def generate_report(latency: Dict, clinical: Dict, traceability: Dict, output_path: str):
    """Génère le rapport d'évaluation complet en JSON."""

    # Évaluation go/no-go
    go_criteria = {
        "latency_p95_under_2s": latency.get("latency_ms", {}).get("p95", 9999) < 2000,
        "accuracy_above_70pct": clinical.get("accuracy", 0) >= 0.70,
        "under_triage_below_10pct": clinical.get("under_triage_rate", 1.0) <= 0.10,
        "error_rate_below_5pct": latency.get("error_rate", 1.0) <= 0.05,
        "traceability_functional": traceability.get("interactions_logged", False),
    }
    go_decision = all(go_criteria.values())

    report = {
        "metadata": {
            "generated_at": datetime.utcnow().isoformat() + "Z",
            "report_version": "1.0",
            "project": "CHSA – Agent IA Triage Médical POC",
        },
        "go_no_go": {
            "decision": "GO " if go_decision else "NO-GO ",
            "criteria": go_criteria,
        },
        "latency_benchmark": latency,
        "clinical_evaluation": clinical,
        "traceability": traceability,
        "recommendations": _generate_recommendations(go_criteria, latency, clinical),
    }

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    log.info(f"\n{'='*60}")
    log.info(f"  RAPPORT D'ÉVALUATION FINAL")
    log.info(f"{'='*60}")
    log.info(f"  Décision Go/No-Go : {report['go_no_go']['decision']}")
    log.info(f"  Accuracy clinique : {clinical.get('accuracy', 0):.1%}")
    log.info(f"  Sous-triage       : {clinical.get('under_triage_rate', 0):.1%}")
    log.info(f"  Latence p95       : {latency.get('latency_ms', {}).get('p95', 'N/A')} ms")
    log.info(f"  Rapport sauvegardé : {output_path}")
    log.info(f"{'='*60}\n")

    return report


def _generate_recommendations(criteria: Dict, latency: Dict, clinical: Dict) -> List[str]:
    reco = []

    if not criteria.get("latency_p95_under_2s"):
        reco.append("Optimiser la latence : envisager la quantisation INT8 ou l'utilisation de vLLM avec batch dynamique")

    if not criteria.get("accuracy_above_70pct"):
        reco.append("Améliorer l'accuracy : augmenter le dataset SFT, revoir les hyperparamètres LoRA (r=32)")

    if not criteria.get("under_triage_below_10pct"):
        reco.append(" CRITIQUE : Réduire le sous-triage en ajoutant des cas extrêmes dans le dataset DPO")

    if not criteria.get("traceability_functional"):
        reco.append("Corriger le système de traçabilité — requis pour conformité RGPD et audit médical")

    if not reco:
        reco.append(" Tous les critères sont satisfaits. Prévoir une validation clinique avec des médecins experts avant déploiement pilote.")
        reco.append("Prochaines étapes : montée en charge vers Qwen3-7B, extension du dataset à 50 000 paires, validation par le comité médical du CHSA.")

    return reco


# ══════════════════════════════════════════════════════════════════════════════
# Tests pytest (utilisés par CI/CD)
# ══════════════════════════════════════════════════════════════════════════════

import pytest


@pytest.mark.asyncio
async def test_health_endpoint():
    """Test que le /health répond correctement."""
    async with httpx.AsyncClient() as client:
        r = await client.get("http://localhost:8005/health", timeout=5.0)
    assert r.status_code == 200
    data = r.json()
    assert "status" in data
    assert "model_loaded" in data


@pytest.mark.asyncio
async def test_triage_basic():
    """Test basique d'un appel /triage."""
    async with httpx.AsyncClient() as client:
        r = await client.post("http://localhost:8005/triage", json={
            "symptoms": "Patient avec forte douleur thoracique et essoufflement",
            "age": 55,
        }, timeout=30.0)
    assert r.status_code == 200
    data = r.json()
    assert "niveau_urgence" in data
    assert "recommandations" in data
    assert "request_id" in data


@pytest.mark.asyncio
async def test_triage_validation_empty_symptoms():
    """Test que les symptômes vides sont rejetés."""
    async with httpx.AsyncClient() as client:
        r = await client.post("http://localhost:8005/triage", json={"symptoms": "  "})
    assert r.status_code == 422  # Validation error


@pytest.mark.asyncio
async def test_metrics_endpoint():
    """Test que /metrics répond."""
    async with httpx.AsyncClient() as client:
        r = await client.get("http://localhost:8005/metrics", timeout=5.0)
    assert r.status_code == 200


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

async def main(base_url: str, n_requests: int, output: str):
    log.info("═" * 60)
    log.info("  CHSA — Semaine 4 : Évaluation finale")
    log.info("═" * 60)

    # Vérification healthcheck
    async with httpx.AsyncClient() as client:
        try:
            r = await client.get(f"{base_url}/health", timeout=5.0)
            log.info(f"API status : {r.json().get('status', 'unknown')}")
        except Exception as e:
            log.error(f"API inaccessible : {e}")
            log.error("Lancez d'abord : uvicorn week4_api_fastapi:app --host 0.0.0.0 --port 8005")
            return

    # Tests
    log.info("\n── 1. Benchmark latence ──")
    latency_metrics = await latency_benchmark(base_url, n_requests=n_requests)

    log.info("\n── 2. Tests cliniques ──")
    clinical_metrics = await run_clinical_tests(base_url)

    log.info("\n── 3. Tests traçabilité ──")
    trace_metrics = await test_traceability(base_url)

    # Rapport final
    generate_report(latency_metrics, clinical_metrics, trace_metrics, output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Évaluation finale CHSA Triage Agent")
    parser.add_argument("--url", default="http://localhost:8005", help="URL de l'API")
    parser.add_argument("--n", type=int, default=50, help="Nombre de requêtes pour le benchmark")
    parser.add_argument("--output", default="reports/evaluation_report.json", help="Fichier de sortie")
    args = parser.parse_args()

    asyncio.run(main(args.url, args.n, args.output))
