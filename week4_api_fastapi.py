import os
import uuid
import time
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Dict, Any
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, BackgroundTasks, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, field_validator

logging.basicConfig(level=logging.INFO, format="%(asctime)s — %(levelname)s — %(message)s")
log = logging.getLogger(__name__)

# ══════════════════════════════════════════════════════════════════════════════
# Configuration
# ══════════════════════════════════════════════════════════════════════════════

#MODEL_PATH = os.getenv("MODEL_PATH", os.path.join(os.path.dirname(__file__), "checkpoints/dpo/best_model/merged"))
#print(MODEL_PATH)
MODEL_PATH = os.getenv("MODEL_PATH", "nix3s/CHSA_Model")
MAX_TOKENS    = int(os.getenv("MAX_TOKENS", "512"))
TEMPERATURE   = float(os.getenv("TEMPERATURE", "0.1"))
INTERACTIONS_LOG = Path(os.getenv("INTERACTIONS_LOG", "logs/interactions.jsonl"))
INTERACTIONS_LOG.parent.mkdir(parents=True, exist_ok=True)

SYSTEM_PROMPT = (
    "Tu es un agent de triage médical expert du Centre Hospitalier Saint-Aurélien (CHSA). "
    "Ton rôle est d'analyser les symptômes décrits par le patient, d'évaluer le niveau d'urgence "
    "selon les critères : URGENCE_MAXIMALE (risque vital immédiat), URGENCE_MODEREE (consultation "
    "dans les 2h), URGENCE_DIFFEREE (consultation programmable), et de fournir des recommandations "
    "cliniques claires. Formate toujours ta réponse en JSON avec les clés : "
    "'niveau_urgence', 'justification', 'recommandations', 'signes_alarme'."
)

# ══════════════════════════════════════════════════════════════════════════════
# Modèles Pydantic (schémas d'API)
# ══════════════════════════════════════════════════════════════════════════════

class TriageRequest(BaseModel):
    """Requête de triage médical."""
    patient_id: Optional[str] = Field(
        default=None,
        description="Identifiant anonymisé du patient (optionnel)"
    )
    symptoms: str = Field(
        ...,
        min_length=10,
        max_length=2000,
        description="Description des symptômes en langage naturel",
        json_schema_extra={"example": "Patient de 45 ans, douleur thoracique irradiant vers le bras gauche, sueurs froides"},
    )
    age: Optional[int] = Field(default=None, ge=0, le=120, description="Âge du patient")
    antecedents: Optional[str] = Field(default=None, description="Antécédents médicaux pertinents")
    constantes: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Constantes vitales (FC, TA, SpO2, T°)",
        json_schema_extra={"example": {"fc": 110, "ta": "145/90", "spo2": 94, "temperature": 37.2}},
    )
    langue: str = Field(default="fr", description="Langue de la réponse (fr/en)")

    @field_validator("symptoms", mode="before")
    @classmethod
    def symptoms_not_empty(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("Les symptômes ne peuvent pas être vides")
        return v.strip()


class TriageLevel(str):
    MAXIMALE = "URGENCE_MAXIMALE"
    MODEREE  = "URGENCE_MODEREE"
    DIFFEREE = "URGENCE_DIFFEREE"


class TriageResponse(BaseModel):
    """Réponse de triage médical."""
    request_id: str
    patient_id: Optional[str]
    timestamp: str
    niveau_urgence: str
    justification: str
    recommandations: List[str]
    signes_alarme: List[str]
    confidence: float
    latency_ms: float
    model_version: str
    disclaimer: str = (
        "Cet outil est une aide au triage et ne remplace pas l'évaluation d'un professionnel de santé. "
        "En cas de doute, appelez le 15 (SAMU) immédiatement."
    )


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    uptime_s: float
    total_requests: int
    model_path: str
    timestamp: str


class MetricsResponse(BaseModel):
    total_requests: int
    avg_latency_ms: float
    p95_latency_ms: float
    error_rate: float
    urgence_distribution: Dict[str, int]
    uptime_s: float


# ══════════════════════════════════════════════════════════════════════════════
# State global de l'application
# ══════════════════════════════════════════════════════════════════════════════

class AppState:
    def __init__(self):
        self.llm = None
        self.model_loaded = False
        self.start_time = time.time()
        self.total_requests = 0
        self.errors = 0
        self.latencies: List[float] = []
        self.urgence_counts = {
            "URGENCE_MAXIMALE": 0,
            "URGENCE_MODEREE": 0,
            "URGENCE_DIFFEREE": 0,
            "INDETERMINE": 0,
        }

    def record_request(self, latency_ms: float, niveau: str, error: bool = False):
        self.total_requests += 1
        self.latencies.append(latency_ms)
        if error:
            self.errors += 1
        self.urgence_counts[niveau] = self.urgence_counts.get(niveau, 0) + 1

        # Garde seulement les 1000 dernières latences
        if len(self.latencies) > 1000:
            self.latencies = self.latencies[-1000:]


STATE = AppState()

# ══════════════════════════════════════════════════════════════════════════════
# Chargement du modèle (vLLM ou transformers fallback)
# ══════════════════════════════════════════════════════════════════════════════

def load_model():
    global STATE
    try:
        import vllm
        from vllm import LLM, SamplingParams
        
        log.info(f"Chargement via vLLM : {MODEL_PATH}")
        STATE.llm = LLM(model=MODEL_PATH, trust_remote_code=True)
        STATE.model_loaded = True
        log.info(" Modèle chargé via vLLM")
        return 
        
    except ImportError:
        log.warning("vLLM non disponible, fallback transformers")
    except Exception as e:
        log.warning(f"vLLM échoue : {e}, fallback transformers")
    
    try:
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
        
        log.info(f"Chargement via TRANSFORMERS : {MODEL_PATH}")
        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_PATH,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
        pipe = pipeline(
            "text-generation", 
            model=model, 
            tokenizer=tokenizer,
            device_map="auto"
        )
        
        class TransformerWrapper:
            def __init__(self, pipe): 
                self.pipe = pipe
            
            def generate(self, prompt: str, max_tokens: int = 512) -> str:
                result = self.pipe(
                    prompt, 
                    max_new_tokens=max_tokens, 
                    temperature=TEMPERATURE,
                    do_sample=TEMPERATURE > 0,
                    pad_token_id=tokenizer.eos_token_id,
                    return_full_text=False
                )
                return result[0]["generated_text"]
        
        STATE.llm = TransformerWrapper(pipe)
        STATE.model_loaded = True
        log.info(" Modèle chargé via TRANSFORMERS")
        
    except Exception as e:
        log.error(f"Erreur transformers : {e}")
        log.warning(" Mode MOCK activé")




def _load_transformers_fallback():
    """Charge le modèle avec transformers (mode développement)."""
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

    class TransformerWrapper:
        def __init__(self, pipe):
            self.pipe = pipe

        def generate(self, prompt: str, max_tokens: int = 512) -> str:
            result = self.pipe(
                prompt,
                max_new_tokens=max_tokens,
                temperature=TEMPERATURE,
                do_sample=TEMPERATURE > 0,
                return_full_text=False,
            )
            return result[0]["generated_text"]

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        trust_remote_code=True,
        dtype=torch.bfloat16,
        device_map="auto",
    )
    pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)
    STATE.llm = TransformerWrapper(pipe)
    STATE.model_loaded = True
    log.info("Modèle chargé via transformers (fallback)")


# ══════════════════════════════════════════════════════════════════════════════
# Logique de triage
# ══════════════════════════════════════════════════════════════════════════════

def build_prompt(req: TriageRequest) -> str:
    """Construit le prompt ChatML à partir de la requête."""
    context_parts = [f"Symptômes : {req.symptoms}"]

    if req.age is not None:
        context_parts.append(f"Âge : {req.age} ans")

    if req.antecedents:
        context_parts.append(f"Antécédents : {req.antecedents}")

    if req.constantes:
        constantes_str = ", ".join(f"{k}={v}" for k, v in req.constantes.items())
        context_parts.append(f"Constantes vitales : {constantes_str}")

    user_content = "\n".join(context_parts)

    return (
        f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
        f"<|im_start|>user\n{user_content}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )


def parse_triage_response(raw: str) -> Dict:
    """
    Parse la réponse JSON du modèle.
    Fallback sur extraction regex si le JSON est malformé.
    """
    # Tentative JSON directe
    try:
        # Cherche le bloc JSON dans la réponse
        json_match = __import__("re").search(r"\{.*\}", raw, __import__("re").DOTALL)
        if json_match:
            return json.loads(json_match.group())
    except (json.JSONDecodeError, AttributeError):
        pass

    # Fallback : extraction par mots-clés
    import re
    niveau = "INDETERMINE"
    for level in ["URGENCE_MAXIMALE", "URGENCE_MODEREE", "URGENCE_DIFFEREE"]:
        if level in raw.upper():
            niveau = level
            break

    return {
        "niveau_urgence": niveau,
        "justification": raw[:500] if raw else "Réponse non structurée",
        "recommandations": ["Consultez un professionnel de santé"],
        "signes_alarme": [],
    }


def mock_triage(symptoms: str) -> Dict:
    """Réponse mock pour le mode développement (sans GPU)."""
    import re
    critical_keywords = ["douleur thoracique", "essoufflement", "perte de conscience",
                         "chest pain", "difficulty breathing", "unconscious"]
    moderate_keywords = ["fièvre", "douleur", "nausée", "fever", "pain", "nausea"]

    symptoms_lower = symptoms.lower()
    if any(kw in symptoms_lower for kw in critical_keywords):
        niveau = "URGENCE_MAXIMALE"
        reco = ["Appeler le 15 (SAMU) immédiatement", "Ne pas laisser le patient seul"]
        alarme = ["Douleur thoracique = signe d'alerte cardiaque potentiel"]
    elif any(kw in symptoms_lower for kw in moderate_keywords):
        niveau = "URGENCE_MODEREE"
        reco = ["Consultation médicale dans les 2 heures", "Surveillance des constantes"]
        alarme = ["Surveillance de l'évolution des symptômes"]
    else:
        niveau = "URGENCE_DIFFEREE"
        reco = ["Consultation médicale programmable", "Surveillance à domicile possible"]
        alarme = []

    return {
        "niveau_urgence": niveau,
        "justification": f"Évaluation basée sur les symptômes : {symptoms[:100]}… [MODE MOCK]",
        "recommandations": reco,
        "signes_alarme": alarme,
    }


def run_inference(req: TriageRequest) -> Dict:
    """Lance l'inférence sur le modèle (vLLM/transformers/mock)."""
    if STATE.llm is None:
        return mock_triage(req.symptoms)

    prompt = build_prompt(req)
    
    try:
        start_time = time.perf_counter()
        
        # vLLM detection and usage
        try:
            import vllm
            if hasattr(STATE.llm, 'generate'):  # vLLM LLM instance
                sampling_params = vllm.SamplingParams(
                    temperature=TEMPERATURE,
                    max_tokens=MAX_TOKENS,
                    stop=["<|im_end|>", "<|im_start|>"]
                )
                outputs = STATE.llm.generate([prompt], sampling_params)
                raw = outputs[0].outputs[0].text
            else:
                raise AttributeError("Not vLLM")
        except (ImportError, AttributeError, Exception):
            # Transformers fallback
            raw = STATE.llm.generate(prompt, max_tokens=MAX_TOKENS)
        
        log.info(f"Inférence réussie ({time.perf_counter() - start_time:.2f}s)")
        return parse_triage_response(raw)
        
    except Exception as e:
        log.error(f" Erreur d'inférence : {e}")
        return mock_triage(req.symptoms)



# ══════════════════════════════════════════════════════════════════════════════
# Traçabilité RGPD
# ══════════════════════════════════════════════════════════════════════════════

def log_interaction(request_id: str, req: TriageRequest, resp: TriageResponse, latency_ms: float):
    """Enregistre l'interaction dans le journal de traçabilité (RGPD)."""
    entry = {
        "request_id": request_id,
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "patient_id": req.patient_id,       # Déjà anonymisé côté client
        "age": req.age,
        "langue": req.langue,
        "niveau_urgence": resp.niveau_urgence,
        "latency_ms": latency_ms,
        "model_version": resp.model_version,
        # NOTE : on ne logue PAS les symptômes bruts (données sensibles)
        "symptoms_length": len(req.symptoms),
        "has_constantes": req.constantes is not None,
    }
    with open(INTERACTIONS_LOG, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


# ══════════════════════════════════════════════════════════════════════════════
# Application FastAPI
# ══════════════════════════════════════════════════════════════════════════════

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Chargement du modèle au démarrage."""
    log.info("Démarrage de l'API CHSA Triage…")
    load_model()
    yield
    log.info("Arrêt de l'API CHSA Triage")


app = FastAPI(
    title="CHSA – Agent IA Triage Médical",
    description=(
        "API de démonstration du POC d'agent IA de triage médical "
        "du Centre Hospitalier Saint-Aurélien. "
        "⚠️ Usage strictement expérimental — ne pas utiliser en conditions réelles."
    ),
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],       # Restreindre en production
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def root():
    return {
        "message": "CHSA Triage API running",
        "docs": "/docs"
    }
@app.post("/triage", response_model=TriageResponse, summary="Analyse de triage médical")
async def triage_endpoint(req: TriageRequest, background_tasks: BackgroundTasks):
    """
    Analyse les symptômes d'un patient et retourne une évaluation de priorité.
    - **symptoms** : description libre des symptômes
    - **age** : âge du patient (améliore la précision)
    - **constantes** : FC, TA, SpO2, Température
    - **antecedents** : antécédents médicaux pertinents
    """
    request_id = str(uuid.uuid4())
    start = time.perf_counter()

    try:
        result = run_inference(req)
        latency_ms = (time.perf_counter() - start) * 1000

        niveau = result.get("niveau_urgence", "INDETERMINE")
        STATE.record_request(latency_ms, niveau)

        response = TriageResponse(
            request_id=request_id,
            patient_id=req.patient_id,
            timestamp=datetime.utcnow().isoformat() + "Z",
            niveau_urgence=niveau,
            justification=result.get("justification", ""),
            recommandations=result.get("recommandations", []),
            signes_alarme=result.get("signes_alarme", []),
            confidence=0.85 if STATE.model_loaded else 0.5,  # Mock = confiance réduite
            latency_ms=round(latency_ms, 2),
            model_version=MODEL_PATH,
        )

        background_tasks.add_task(log_interaction, request_id, req, response, latency_ms)
        return response

    except Exception as e:
        latency_ms = (time.perf_counter() - start) * 1000
        STATE.record_request(latency_ms, "INDETERMINE", error=True)
        log.error(f"Erreur triage {request_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Erreur interne : {str(e)}")


@app.get("/health", response_model=HealthResponse, summary="Healthcheck")
async def health():
    return HealthResponse(
        status="ok" if STATE.model_loaded else "degraded",
        model_loaded=STATE.model_loaded,
        uptime_s=round(time.time() - STATE.start_time, 1),
        total_requests=STATE.total_requests,
        model_path=MODEL_PATH,
        timestamp=datetime.utcnow().isoformat() + "Z",
    )


@app.get("/metrics", response_model=MetricsResponse, summary="Métriques de performance")
async def metrics():
    lats = STATE.latencies
    avg_lat = sum(lats) / len(lats) if lats else 0.0
    p95_lat = sorted(lats)[int(len(lats) * 0.95)] if lats else 0.0
    err_rate = STATE.errors / STATE.total_requests if STATE.total_requests > 0 else 0.0

    return MetricsResponse(
        total_requests=STATE.total_requests,
        avg_latency_ms=round(avg_lat, 2),
        p95_latency_ms=round(p95_lat, 2),
        error_rate=round(err_rate, 4),
        urgence_distribution=STATE.urgence_counts,
        uptime_s=round(time.time() - STATE.start_time, 1),
    )


@app.get("/interactions", summary="Historique des interactions (traçabilité)")
async def interactions(limit: int = 100):
    """Retourne les N dernières interactions (sans données sensibles)."""
    if not INTERACTIONS_LOG.exists():
        return {"interactions": [], "total": 0}

    lines = INTERACTIONS_LOG.read_text(encoding="utf-8").strip().split("\n")
    lines = [l for l in lines if l.strip()]
    records = [json.loads(l) for l in lines[-limit:]]
    return {"interactions": records, "total": len(lines)}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "week4_api_fastapi:app",
        host="0.0.0.0",
        port=int(os.getenv("PORT", "8000")),
        reload=False,
        log_level="info",
    )
