graph TB
    %% Utilisateurs
    Client[ Client/Symptoms] -->|POST /triage| LoadBalancer[ Load Balancer]
    
    %% API Layer
    LoadBalancer --> FastAPI[ FastAPI + Uvicorn]
    
    %% Health & Metrics
    FastAPI -->|GET /health| HealthCheck[ Healthcheck]
    FastAPI -->|GET /metrics| Metrics[ Prometheus Metrics]
    FastAPI -->|GET /interactions| Logs[ RGPD Logs]
    
    %% Inference Pipeline
    FastAPI -->|async| ModelLoader[ vLLM Model Loader]
    ModelLoader -->|GPU RTX4060| QwenModel[Qwen3-1.7B-DPO<br/>fp16 + LoRA r=16]
    
    %% Processing Pipeline
    Client -->|JSON<br/>symptoms, age, constantes| FastAPI
    FastAPI -->|Prompt Engineering| QwenModel
    QwenModel -->|JSON Response| FastAPI
    FastAPI -->|200 OK| Client
    
    %% Training Pipeline (Notebooks)
    subgraph TRAIN [" Training Pipeline"]
        DataPrep[ week1_data_pipeline.py<br/>FrenchMedMCQA + MediQA]
        SFT[ week2_sft_lora.py<br/>SFT + LoRA]
        DPO[ week3_dpo_alignment.py<br/>DPO beta=0.1]
        DataPrep --> SFT --> DPO -->|checkpoints/dpo/best_model/merged| QwenModel
    end
    
    %% Evaluation
    subgraph EVAL [" Évaluation"]
        Benchmark[week4_evaluation.py<br/>10 cas cliniques]
        Benchmark -->|80% accuracy<br/>0% sous-triage| FastAPI
    end
    
    %% Data & Logs
    subgraph STORAGE [" Persistance"]
        Checkpoints[checkpoints/<br/>sft+dpo]
        Dataset[data/<br/>sft+dpo jsonl]
        Logs[logs/interactions.jsonl<br/>RGPD compliant]
    end
    
    %% Docker
    subgraph DOCKER [" Container"]
        FastAPI
        ModelLoader
        QwenModel
    end
    
    %% External
    FastAPI -.->|Swagger UI| Docs[ /docs]
    Logs -.->|Anonymisé| RGPD[ RGPD Compliance]
    
    %% GitHub Actions
    subgraph CI_CD [" CI/CD"]
        GitHub[GitHub Actions<br/>week4_github_actions.yml]
    end
    
    %% Styling
    classDef client fill:#e1f5fe
    classDef api fill:#f3e5f5
    classDef model fill:#e8f5e8
    classDef train fill:#fff3e0
    classDef storage fill:#fce4ec
    classDef success stroke:#4caf50,stroke-width:3px
    
    class Client,LoadBalancer client
    class FastAPI,HealthCheck,Logs,Metrics api
    class QwenModel,ModelLoader model
    class DataPrep,SFT,DPO,EVAL,Benchmark train
    class Checkpoints,Dataset,Logs storage
    
    %% Performance badges
    BADGE_LATENCY[" p95: 18.4s<br/>(cible <2s)"]:::success
    BADGE_ACCURACY[" Accuracy: 80%<br/>(cible ≥70%)"]:::success
    BADGE_SAFE[" Sous-triage: 0%"]:::success
    
    QwenModel ~~~ BADGE_LATENCY
    Benchmark ~~~ BADGE_ACCURACY
    QwenModel ~~~ BADGE_SAFE
