# CampusMate: University FAQ & Intent Assistant

A supervised classical Natural Language Processing (NLP) intent classification and response retrieval system designed for university FAQ assistance. 

CampusMate maps student and applicant inquiries to verified, domain-specific institutional answers using **TF-IDF vectorization**, **multinomial Logistic Regression**, **confidence gating**, **multi-intent disambiguation**, and **adversarial out-of-domain (OOD) protection**.

---

## 1. Overview

CampusMate is engineered as a reliable, explainable, and lightweight FAQ conversational assistant. Unlike generative Large Language Models (LLMs) which are prone to hallucinating critical academic details (such as tuition figures, deadlines, or degree requirements), CampusMate operates strictly on supervised intent classification paired with deterministic response mapping.

### Core Architecture Flow
```
User Query
   │
   ▼
[ 1. Input Validation & Character Limit Check ]
   │
   ▼
[ 2. Compatibility & Adversarial OOD Filter ] ──(Incompatible)──► [ Fallback Response ]
   │
   ▼
[ 3. Multi-Intent Disambiguation Detector ] ──(Multi-Intent)──► [ Multi-Topic Guidance ]
   │
   ▼
[ 4. Deterministic NLP Preprocessing ]
   ├── Contraction & Chat Acronym Expansion
   ├── Domain Spelling & Synonym Normalization
   ├── Stopword Removal (Preserving Question Negations)
   └── WordNet Lemmatization
   │
   ▼
[ 5. WordNet-Protected Typo Correction ]
   │
   ▼
[ 6. Out-of-Domain Vocabulary Coverage Check ] ──(Coverage < 35%)─► [ Fallback Response ]
   │
   ▼
[ 7. Sublinear TF-IDF Feature Extraction ]
   │
   ▼
[ 8. Logistic Regression Probability Estimation (predict_proba) ]
   │
   ▼
[ 9. Confidence & Ambiguity Margin Gating ] ──(Low Conf / Ambiguous)─► [ Fallback Response ]
   │
   ▼
[ 10. Predefined Response Selection & Multi-Turn Context Update ]
   │
   ▼
Final Structured Answer
```

---

## 2. Key Features

- **No Generative AI / No Hallucinations**: Built entirely without LLMs, external APIs, RAG, vector databases, or neural networks.
- **Custom Curated Dataset**: Authoritative dataset (`data/intents.json`) containing 24 distinct university FAQ categories and 487 linguistically diverse training patterns with SHA-256 integrity verification.
- **Multi-Intent Disambiguation**: Detects when queries combine multiple primary topics (e.g., `"What is the admission process and annual fee?"`) and prompts the user to ask about topics separately rather than guessing a single intent.
- **Adversarial & OOD Hardening**: Multi-layer filters protect against creative generative prompts (jokes, poems, stories), language translation, prompt injection attacks, and unrelated domains (weather, stocks, crypto).
- **WordNet-Protected Typo Handling**: Safely corrects student typos (`admision` -> `admission`, `libry` -> `library`) while guaranteeing valid English words (`hotel`, `france`, `hospital`, `replacement`, `weather`, `bitcoin`) are never falsely converted.
- **Context-Aware Multi-Turn Disambiguation**: Short ambiguous follow-ups (e.g., `"where?"`, `"when?"`, `"what about computer engineering?"`) automatically leverage conversation dialog memory from prior turns.
- **Dual User Interfaces**: Interactive modern **Streamlit Web Application** with dark glassmorphic styling and an interactive developer **CLI Console**.

---

## 3. Technology Stack

- **Core Logic & ML**: Python 3.10+, `scikit-learn`, `numpy`, `pandas`
- **NLP & Linguistics**: `nltk` (Tokenizers, WordNet Lemmatizer, English Stopwords)
- **Web UI**: `streamlit`
- **Testing & Quality Assurance**: `pytest`
- **Serialization**: `pickle`, `json`, `pathlib`

---

## 4. Dataset & Intent Taxonomy

The single source of truth is [`data/intents.json`](data/intents.json). It defines 24 distinct university categories:

| Intent Tag | Category Focus | Example User Query |
| :--- | :--- | :--- |
| `greeting` | Conversational greetings | `"Hello! Good morning"` |
| `goodbye` | Conversation termination | `"Goodbye, have a good day"` |
| `name` | Assistant identity | `"What is your name?"` |
| `creator` | Project authorship | `"Who created CampusMate?"` |
| `courses` | Available degree programs & branches | `"What courses are offered?"` |
| `course_details` | Curriculum, subjects & coding content | `"Is programming taught in Computer Engineering?"` |
| `fees` | Tuition & payment schedules | `"What is the annual fee structure?"` |
| `admission` | Eligibility & application steps | `"How do I apply for admission?"` |
| `documents` | Required verification certificates | `"What certificates must be verified for enrollment?"` |
| `hours` | University operational hours | `"What are the college working hours?"` |
| `location` | Campus address & transit directions | `"Where is the university located?"` |
| `hostel` | Student housing availability | `"Is hostel accommodation available for outstation students?"` |
| `hostel_facilities` | Mess, Wi-Fi & dormitory amenities | `"What facilities are available in the hostel?"` |
| `library` | Books, journals & reading rooms | `"Tell me about the central library"` |
| `library_hours` | Library opening & closing schedules | `"What are the library timings?"` |
| `placement` | Campus recruitment & career cell | `"Which companies visit for campus placements?"` |
| `scholarship` | Financial aid & fee concessions | `"Are merit scholarships available?"` |
| `contact` | Telephone & administrative emails | `"How can I contact the university administration?"` |
| `website` | Official university web portal | `"What is the official university website?"` |
| `facilities` | Overall campus infrastructure & labs | `"What facilities are available on campus?"` |
| `events` | Cultural festivals & hackathons | `"What events take place in the university?"` |
| `exams` | Semester examination schedule | `"When are semester exams conducted?"` |
| `canteen` | Cafeteria & food options | `"Does the college have a canteen?"` |
| `sports` | Gymnasium & athletic grounds | `"What sports facilities are available?"` |

---

## 5. NLP Preprocessing Pipeline

The preprocessing module ([`src/preprocess.py`](src/preprocess.py)) is shared deterministically between training and inference:

1. **Character Length Enforcement**: Rejects inputs exceeding 500 characters.
2. **Case Normalization & Artifact Stripping**: Lowercases all text and removes URLs/HTML tags.
3. **Contraction Expansion**: Maps contractions (`"can't"` -> `"cannot"`, `"what's"` -> `"what is"`).
4. **Chat Acronym Normalization**: Standardizes student slang (`"r"` -> `"are"`, `"u"` -> `"you"`, `"pls"` -> `"please"`, `"docs"` -> `"documents"`, `"cse"` -> `"computer science"`).
5. **Domain Typo Correction**: Corrects standard high-frequency typos (`"admision"` -> `"admission"`, `"univercity"` -> `"university"`, `"libry"` -> `"library"`).
6. **Repeated Letter Collapse**: Condenses character elongation (`"heyyyy"` -> `"hey"`, `"hellooo"` -> `"hello"`).
7. **Punctuation & Space Normalization**: Converts special characters to clean whitespace.
8. **Preserved-Keyword Stopword Filtering**: Removes generic stopwords while preserving intent-critical keywords (`what`, `when`, `where`, `why`, `how`, `who`, `which`, `not`, `no`, `cannot`).
9. **WordNet Lemmatization**: Reduces tokens to their root noun and verb lemmas.

---

## 6. Model Training & Evaluation

The training pipeline ([`src/train.py`](src/train.py)) utilizes a stratified train/test split to evaluate generalization performance, and then fits the production artifacts on all curated patterns while recording the dataset's SHA-256 integrity hash.

### Hyperparameters
- **TF-IDF Vectorizer**: `ngram_range=(1, 2)`, `sublinear_tf=True`, `min_df=1`
- **Logistic Regression**: `C=10.0`, `class_weight='balanced'`, `max_iter=300`, `random_state=42`
- **Confidence Threshold**: `0.50`
- **Ambiguity Margin**: `0.08`

### Evaluation Metrics (Independent Test Split)
- **Test Accuracy**: `83.67%`
- **Weighted F1-Score**: `83.41%`
- **Total Intent Classes**: `24`
- **Vocabulary Features**: `1,065` n-grams

---

## 7. Project Structure

```
CampusMate/
│
├── app/
│   ├── __init__.py             # App package initializer
│   ├── streamlit_app.py        # Streamlit Web Application (Modern Dark UI)
│   └── cli.py                  # Interactive Command-Line Interface
│
├── data/
│   └── intents.json            # Single authoritative custom dataset (24 intents, 487 patterns)
│
├── saved_model/
│   ├── intent_classifier.pkl   # Serialized Logistic Regression model
│   ├── vectorizer.pkl          # Serialized TF-IDF feature extractor
│   └── model_metadata.json     # Hyperparameters, hash, and evaluation metrics
│
├── src/
│   ├── __init__.py             # Package initializer
│   ├── config.py               # Central paths and engine hyperparameters
│   ├── data_loader.py          # Schema validation and dataset extraction
│   ├── preprocess.py           # Deterministic NLP cleaning & lemmatization
│   ├── train.py                # Stratified evaluation & production training
│   ├── chatbot.py              # Multi-stage inference and fallback engine
│   ├── response_manager.py     # Response validation and retrieval
│   └── utils.py                # Helper utilities
│
├── tests/
│   ├── __init__.py             # Test package initializer
│   ├── conftest.py             # Shared pytest fixtures
│   ├── test_preprocess.py      # Unit tests for NLP cleaning & tokenization
│   ├── test_data_loader.py     # Unit tests for JSON schema and loaders
│   ├── test_classifier.py      # Tests for model artifacts, hash & metadata
│   ├── test_chatbot.py         # Integration tests for engine responses
│   ├── test_ood.py             # Out-of-domain & security gate tests
│   └── test_hard_cases.py      # 16 mentor adversarial & real-world categories
│
├── .gitignore                  # Git ignore rules
├── pytest.ini                  # Pytest configuration
├── requirements.txt            # Minimal verified dependencies
└── README.md                   # Authoritative project documentation
```

---

## 8. Installation & Setup

### Prerequisites
- Python 3.10, 3.11, 3.12, or 3.13
- Git

### 1. Clone Repository & Setup Environment
```bash
# Clone the repository
git clone https://github.com/your-username/CampusMate.git
cd CampusMate

# Create a virtual environment
python -m venv .venv

# Activate the virtual environment
# On Windows (PowerShell):
.venv\Scripts\Activate.ps1
# On Linux/macOS:
source .venv/bin/activate
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Train Intent Classification Model
```bash
python -m src.train
```

---

## 9. Running the Application

### Option A: Streamlit Web UI
```bash
streamlit run app/streamlit_app.py
```
Open your browser at `http://localhost:8501`.

### Option B: Interactive CLI
```bash
python app/cli.py
```

#### CLI Developer Commands:
- `/help` — Display list of commands
- `/debug` — Toggle NLP pipeline diagnostics
- `/intents` — List all 24 supported categories
- `/stats` — View model parameters and evaluation scores
- `/reset` — Clear multi-turn conversational context
- `/quit` — Exit CLI

---

## 10. Running Tests & Quality Verification

Run the entire test suite covering unit tests, integration tests, OOD filters, multi-intent handling, and adversarial hard cases:

```bash
pytest
```

Run with verbose output:
```bash
pytest -v
```

### Verified Test Categories (50/50 Passed)
1. **Category 1**: Legitimate FAQ queries across all 24 intents
2. **Category 2**: Paraphrased and conversational inquiries
3. **Category 3**: Student typos and chat acronyms
4. **Category 4**: Chat language and student slang (`"bro what branches do u have?"`, `"can u tell me fee?"`, `"abt hostel"`)
5. **Category 5**: Single-word short queries (`"fees"`, `"hostel"`, `"library"`)
6. **Category 6**: Ambiguous queries without context (`"details"`, `"when?"`, `"okay"`)
7. **Category 7**: Context follow-ups and multi-turn conversations
8. **Category 8**: General Out-of-Domain topics (Bitcoin, weather, world leaders)
9. **Category 9**: Creative/generative writing requests (jokes, poems, stories, riddles)
10. **Category 10**: Code generation requests vs legitimate academic inquiries
11. **Category 11**: Prompt injection and jailbreak attempts
12. **Category 12**: False typo correction safety (protecting valid English words)
13. **Category 13**: Multi-intent dual queries (`"What is the admission process and annual fee?"`)
14. **Category 14**: Mixed legitimate + malicious/OOD queries
15. **Category 15**: Input abuse (empty string, emojis, numbers, 600+ chars, HTML, URLs)
16. **Category 16**: Real-world colloquial student phrasing
17. **Model & Response Integrity**: SHA-256 dataset hash verification, missing model handling, response quality checks

---

## 11. Deployment on Streamlit Community Cloud

CampusMate is 100% prepared for instant deployment on [Streamlit Community Cloud](https://streamlit.io/cloud):

1. Push this repository to GitHub.
2. Log into Streamlit Cloud and click **"New app"**.
3. Select your repository, set the Branch to `main`, and set the Main file path to:
   ```
   app/streamlit_app.py
   ```
4. Click **Deploy**. Streamlit Cloud will install dependencies from `requirements.txt`, load the trained model artifacts, and launch the application.

---

## 12. Limitations & Boundaries

1. **Rule-Based & Supervised NLP**: CampusMate is **not** an LLM or generative model. It does not synthesize new text; it classifies user queries into defined intents and returns pre-curated responses.
2. **Closed Domain**: CampusMate only answers queries within its 24 supported university FAQ topics. Questions outside this scope are intentionally routed to fallback.
3. **Vocabulary Dependence**: Queries containing exclusively unseen terms that fail domain similarity will safely trigger fallback rather than guessing incorrect information.

---

## 13. License

This project is open-source and available under the **MIT License**.
