# 🏥 drug-dx-qc-agents
Multi-Agent Drug & Diagnosis Quality Control Pipeline (Google ADK)

![License: Apache-2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)
![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)
![Status](https://img.shields.io/badge/Status-Active-success.svg)

---

## 📖 Overview
This project implements a **multi-agent pipeline** using the **Google Agent Development Kit (ADK)** to process deidentified electronic health records (EHRs), classify medications, summarize drug classes, and evaluate whether diagnoses align with prescribed drugs.  

The pipeline ensures **quality control** by identifying patients with medications that lack corresponding diagnoses in their records.  

---

## 🧠 Agent Architecture
The system runs **4 modular agents** in sequence:

1. **drug-identifier**  
   Extracts drug names from `medication.csv` and normalizes spelling/dosage forms.  

2. **drug-classifier**  
   Maps drugs to ATC hierarchical classes (e.g., bronchodilators, antihypertensives) and to diagnosis (dx).  

3. **stats-summarizer**  
   Summarizes drug usage by patient and class, providing context for diagnosis evaluation.  

4. **qc-evaluator**  
   Compares dx against `diagnosis.csv` and flags patients with prescriptions lacking corresponding diagnoses.  

---

## 📁 Project Structure
```
multi-agent-qc-pipeline/
│── agents/
│   ├── drug_identifier.py
│   ├── drug_classifier.py
│   ├── stats_summarizer.py
│   ├── qc_evaluator.py
│── data/
│   ├── medication.csv   # synthetic sample
│   ├── diagnosis.csv    # synthetic sample
│── tests/
│   ├── test_agents.py
│── main.py
│── requirements.txt
│── README.md
```

---

## 🔧 Installation
Clone the repository and set up a virtual environment:

```bash
git clone https://github.com/<your-username>/drug-dx-qc-agents.git
cd drug-dx-qc-agents
python -m venv venv
venv\Scripts\activate   # Windows
pip install -r requirements.txt
```

---

## 🚀 Usage
Run the pipeline locally with synthetic data:

```bash
python main.py --med data/medication.csv --diag data/diagnosis.csv
```

Expected outputs:
- Patient-level summaries
- QC flags (missing diagnosis, mismatched drug-diagnosis pairs)

---

## 🧪 Testing
Run unit tests with:

```bash
pytest tests/
```

---

## 🌐 Reproducibility
- Synthetic EHR samples (`medication.csv`, `diagnosis.csv`) are included for reproducibility.  
- Real datasets can be applied during development but are not shared publicly.  
- Dependencies are listed in `requirements.txt`.  

---

## 🤝 Contributing
Contributions are welcome!  
- Fork the repo  
- Create a feature branch  
- Submit a pull request  

---

## 📜 License
This project is licensed under the Apache License 2.0. See `LICENSE` for details.

---

## 🙏 Acknowledgements
- Google Agent Development Kit (ADK)  
- ATC Classification System  
- FAIRlyz project ecosystem (future integration)  
```

---



