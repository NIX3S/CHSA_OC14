```mermaid
flowchart TD
    %% Styles
    classDef init fill:#e1f5fe
    classDef load fill:#f3e5f5
    classDef format fill:#e8f5e8
    classDef anon1 fill:#fff3e0
    classDef anon2 fill:#fce4ec
    classDef export fill:#e0f2f1
    classDef audit fill:#fff8e1
    
    %% Initialisation
    A[ START<br/>CONFIG + spaCy<br/>fr_core_news_lg + en_core_web_lg] --> B[Presidio<br/>ANALYZER + ANONYMIZER]
    B --> C[AuditLog<br/>RGPD v1.0.0]
    
    %% Chargement datasets
    C --> D1[FrenchMedMCQA<br/>parquet<br/>FR MCQ]
    C --> D2[MediQA<br/>JSON<br/>EN MCQ]
    C --> D3[UltraMedical<br/>JSON<br/>EN DPO]
    
    %% SFT Pipeline
    D1 --> E1[format_frenchmedmcqa<br/>MCQ → instruction/response]
    D2 --> E2[format_mediqa<br/>MCQ → instruction/response]
    E1 --> F[SFT Splits<br/>train/val/test]
    E2 --> F[SFT Splits<br/>train/val/test]
    
    %% DPO Pipeline  
    D3 --> G[format_dpo<br/>prompt → chosen/rejected]
    G --> H[DPO Splits<br/>train/val/test]
    
    %% Anonymisation 2-PASS (COEUR)
    F --> I1[PASSE 1<br/>REGEX rapide<br/>8 patterns FR/EN]
    H --> I1[PASSE 1<br/>REGEX rapide<br/>8 patterns FR/EN]
    I1 --> I2[PASSE 2<br/>Presidio NLP<br/>PERSON/EMAIL/DATE]
    
    %% Métadonnées + Audit
    I2 --> J[Ajout METADATA<br/>id/lang/confidence<br/>pii_detected/timestamp]
    J --> K[AuditLog.record<br/>Chaque <PII> tracé<br/>row_id/field/snippet]
    
    %% Export HF-ready
    K --> L1["data/sft<br/>{train,val,test}.jsonl"]
    K --> L2["data/dpo<br/>{train,val,test}.jsonl"]
    L1 --> M[6900+ paires SFT<br/>5000 train ✓]
    L2 --> N[109k+ paires DPO]
    
    %% Validation & Schema
    M --> O[validate_and_stats<br/>Pandas + console]
    N --> O[validate_and_stats<br/>Pandas + console]
    O --> P[export_metadata_schema<br/>RGPD compliance]
    
    %% Audit final
    K --> Q["audit/sft_audit.json<br/>{dpo_audit.json}"]
    
    %% Fin
    P --> R[ WEEK1 TERMINÉE<br/>Données prêtes SFT/DPO]
    Q --> R
    
    %% Styles
    class A,B,C init
    class D1,D2,D3 load
    class E1,E2,G format
    class I1 anon1
    class I2 anon2
    class J,K audit
    class L1,L2,M,N,P,Q export
    class R init
    
    %% Liens anonymisation détaillés
    I1 -.->|Ex: <HOSPITAL><br/><DOCTOR><br/><PHONE>| I2

    I2 -.->|<PERSON><br/><EMAIL>| J
