# Project Diagrams

## Dependency Graph (Mermaid)

```mermaid
flowchart LR

    subgraph Frontend
        HTML["index.html"]
        CSS["style.css"]
        JS["script.js"]
        IMG["hero.png"]
    end

    subgraph Backend
        APP["app.py (Flask)"]
        VEC["vectorizer.pkl"]
        MODEL["model.pkl"]
        TRAIN["train.py"]
        DATA["dataset.csv"]
    end

    subgraph Docs
        NB["career_prediction_system.ipynb"]
        NOTES["Notes.py"]
        REPORT["AAT_ML_Project_Report.md"]
    end

    REQUIREMENTS["requirements.txt"]

    HTML --> CSS
    HTML --> JS
    HTML --> IMG

    JS -->|"POST /predict"| APP

    APP --> VEC
    APP --> MODEL

    TRAIN --> DATA
    TRAIN --> VEC
    TRAIN --> MODEL

    APP -->|"serves"| HTML

    REQUIREMENTS -.-> APP
    REQUIREMENTS -.-> TRAIN

    NB --> TRAIN
    NOTES --> TRAIN
    REPORT --> TRAIN
```

## Sequence Diagram (Mermaid)

```mermaid
sequenceDiagram
    participant User
    participant Browser
    participant UI as index.html + script.js
    participant API as Flask /predict
    participant V as vectorizer.pkl
    participant M as model.pkl

    User->>UI: Paste skills & click Predict
    UI->>API: POST /predict {text}
    API->>API: Preprocess (tokenize + stem)
    API->>V: transform(text)
    V-->>API: vector
    API->>M: predict(vector)
    M-->>API: label + probabilities
    API-->>UI: JSON response
    UI-->>User: Render prediction results
```

---

### How to View

- **VS Code**: Open `diagram.md` and use the Mermaid Preview extension.
- **GitHub**: Mermaid diagrams render automatically in Markdown files.
- **Online Editor**: Paste the Mermaid blocks into https://mermaid.live