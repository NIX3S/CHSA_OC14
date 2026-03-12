import argparse
import asyncio
import json
import logging
import time
from statistics import mean
from typing import Dict, List

import httpx

# ══════════════════════════════════════════════════════════════════════════════
# LOGGING
logging.basicConfig(level=logging.INFO, format="%(asctime)s — %(levelname)s — %(message)s")
log = logging.getLogger(__name__)

# ══════════════════════════════════════════════════════════════════════════════
# Cas de test cliniques (ground truth médicale) - VOTRE LISTE COMPLÈTE
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
# CLIENT HTTP ASYNCHRONE AVEC TIMEOUT
TIMEOUT = httpx.Timeout(connect=30.0, read=60.0, write=30.0, pool=30.0)

# ══════════════════════════════════════════════════════════════════════════════
# SINGLE REQUEST
async def single_request(client: httpx.AsyncClient, base_url: str, payload: Dict, case_id: str = "") -> Dict:
    """Envoie une requête POST /triage avec gestion timeout et erreurs"""
    start = time.perf_counter()
    log.info(f" {case_id or 'Benchmark'} → En cours...")

    try:
        response = await client.post(f"{base_url}/triage", json=payload, timeout=TIMEOUT)
        latency_ms = (time.perf_counter() - start) * 1000

        if response.status_code == 200:
            data = response.json()
        else:
            data = None
        log.info(f" {case_id or 'Benchmark'} OK: {latency_ms:.0f}ms")
        return {"status_code": response.status_code, "latency_ms": latency_ms, "response": data, "error": None}

    except httpx.TimeoutException:
        latency_ms = (time.perf_counter() - start) * 1000
        log.error(f"⏱ TIMEOUT {case_id or 'Benchmark'}: {latency_ms:.0f}ms")
        return {"status_code": 0, "latency_ms": latency_ms, "response": None, "error": "TIMEOUT"}

    except Exception as e:
        latency_ms = (time.perf_counter() - start) * 1000
        log.error(f" {case_id or 'Benchmark'} ERROR: {latency_ms:.0f}ms - {e}")
        return {"status_code": 0, "latency_ms": latency_ms, "response": None, "error": str(e)}

# ══════════════════════════════════════════════════════════════════════════════
# BENCHMARK SÉQUENTIEL
async def latency_benchmark(base_url: str, n_requests: int = 10) -> Dict:
    log.info(f"⚡ BENCHMARK: {n_requests} reqs SÉQUENTIELLES (15-20s/req)")

    payload = {
        "symptoms": "Patient de 45 ans, douleur thoracique modérée, légère dyspnée d'effort",
        "age": 45,
        "constantes": {"fc": 88, "ta": "130/80", "spo2": 97},
    }

    latencies, errors = [], 0

    async with httpx.AsyncClient(timeout=TIMEOUT) as client:
        for i in range(n_requests):
            result = await single_request(client, base_url, payload, f"B{i+1}")

            if result["error"] or result["status_code"] != 200:
                errors += 1
            else:
                latencies.append(result["latency_ms"])

            # Pause 2s entre requêtes
            if i < n_requests - 1:
                log.info(" Pause 2s...")
                await asyncio.sleep(2.0)

    if not latencies:
        log.error(" AUCUNE RÉPONSE")
        return {}

    latencies.sort()
    n = len(latencies)

    return {
        "n_requests": n_requests,
        "n_success": n,
        "n_errors": errors,
        "error_rate": errors / n_requests,
        "latency_ms": {
            "min": round(min(latencies), 1),
            "p50": round(latencies[n // 2], 1),
            "p95": round(latencies[int(n * 0.95)], 1),
            "max": round(max(latencies), 1),
            "mean": round(mean(latencies), 1),
        },
    }

# ══════════════════════════════════════════════════════════════════════════════
# TESTS CLINIQUES SÉQUENTIELS
async def run_clinical_tests(base_url: str) -> Dict:
    log.info(f" TESTS CLINIQUES: {len(CLINICAL_TEST_CASES)} cas")
    results = []

    async with httpx.AsyncClient(timeout=TIMEOUT) as client:
        for case in CLINICAL_TEST_CASES:
            payload = {
                "symptoms": case["symptoms"],
                "age": case.get("age"),
                "antecedents": case.get("antecedents"),
                "constantes": case.get("constantes"),
            }
            result = await single_request(client, base_url, payload, case["id"])

            if result["response"]:
                predicted = result["response"].get("niveau_urgence", "INDETERMINE")
                expected = case["expected_urgence"]
                correct = predicted == expected

                urgence_rank = {"URGENCE_MAXIMALE": 3, "URGENCE_MODEREE": 2, "URGENCE_DIFFEREE": 1, "INDETERMINE": 0}
                pred_rank = urgence_rank.get(predicted, 0)
                exp_rank = urgence_rank.get(expected, 0)

                results.append({
                    "case_id": case["id"],
                    "expected": expected,
                    "predicted": predicted,
                    "correct": correct,
                    "under_triage": pred_rank < exp_rank,
                    "over_triage": pred_rank > exp_rank,
                    "latency_ms": round(result["latency_ms"], 1),
                    "justification": result["response"].get("justification", "")[:200],
                    "clinical_rationale": case["clinical_rationale"],
                })
            else:
                results.append({
                    "case_id": case["id"],
                    "expected": case["expected_urgence"],
                    "predicted": "ERROR",
                    "correct": False,
                    "under_triage": True,
                    "over_triage": False,
                    "latency_ms": round(result["latency_ms"], 1),
                    "error": result.get("error"),
                    "clinical_rationale": case["clinical_rationale"],
                })

            # Pause 3s entre cas cliniques
            await asyncio.sleep(3.0)

    n_correct = sum(1 for r in results if r["correct"])
    n_under = sum(1 for r in results if r.get("under_triage", False))
    log.info(f" Accuracy: {n_correct/len(results):.1%} | Sous-triage: {n_under/len(results):.1%}")

    return {
        "total_cases": len(results),
        "accuracy": round(n_correct / len(results), 3),
        "under_triage_rate": round(n_under / len(results), 3),
        "cases": results,
    }

# ══════════════════════════════════════════════════════════════════════════════
# TRACEABILITÉ
async def test_traceability(base_url: str) -> Dict:
    async with httpx.AsyncClient(timeout=TIMEOUT) as client:
        try:
            r1 = await client.post(f"{base_url}/triage", json={"symptoms": "Test traçabilité", "patient_id": "TEST_TRACE_001"})
            r2 = await client.get(f"{base_url}/interactions?limit=5")
            interactions_logged = r2.status_code == 200 and len(r2.json().get("interactions", [])) > 0
            return {"triage_status": r1.status_code, "traceability_status": r2.status_code, "interactions_logged": interactions_logged}
        except Exception as e:
            log.error(f" Traceability test failed: {e}")
            return {"triage_status": 0, "traceability_status": 0, "interactions_logged": False}

# ══════════════════════════════════════════════════════════════════════════════
# GENERATE REPORT
def generate_report(latency: Dict, clinical: Dict, trace: Dict, output_file: str):
    report = {"latency": latency, "clinical_tests": clinical, "traceability": trace}
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    log.info(f" Rapport sauvegardé dans {output_file}")

# ══════════════════════════════════════════════════════════════════════════════
# MAIN
async def main(base_url: str, n_requests: int, output: str):
    log.info("═" * 60)
    log.info("  CHSA — Évaluation 15-20s SÉQUENTIELLE")
    log.info("═" * 60)

    # Healthcheck
    async with httpx.AsyncClient(timeout=TIMEOUT) as client:
        try:
            r = await client.get(f"{base_url}/health")
            log.info(f" API Health: {r.json().get('status')}")
        except Exception as e:
            log.error(f" Healthcheck failed: {e}")

    latency = await latency_benchmark(base_url, n_requests)
    clinical = await run_clinical_tests(base_url)
    trace = await test_traceability(base_url)
    generate_report(latency, clinical, trace, output)

# ══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CHSA Évaluation 15-20s Séquentielle")
    parser.add_argument("--url", default="http://localhost:8000")
    parser.add_argument("--n", type=int, default=5, help="Nombre de requêtes benchmark")
    parser.add_argument("--output", default="rapport.json")
    args = parser.parse_args()

    asyncio.run(main(args.url, args.n, args.output))
