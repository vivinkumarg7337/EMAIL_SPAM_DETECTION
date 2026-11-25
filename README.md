# MailGuard — email spam detection

## Automatic threshold training

`scripts/train_sweep.py` searches 30 Atlas Engine configurations (Count/TF-IDF,
1–3 word n-grams, five regularization strengths). It selects each threshold on
validation accuracy, breaking ties by fewer false positives and proximity to 0.5.
The selected model's `model/decision.json` controls both single and batch predictions.
Restart the server after training. Prior models and reports are retained in `backups/`.

**Evaluation correction:** the synthetic generator was previously modified after
viewing test errors. Its test results are exploratory, not an untouched benchmark.
The current search freezes the data. Fresh labeled real email is needed to measure
generalization. Atlas internally fits on 70% of the supplied training file.

A working spam classifier trained with [Atlas Engine](https://github.com/Aruneshw/atlas-engine) and served as a small website.

- **Model:** validation-selected Atlas entry #386 or #387; see `reports/sweep.json` for the current winner.
- **Data:** 2,400 synthetic emails, `subject` + `body` → spam / legitimate
- **Website:** FastAPI backend, vanilla HTML/CSS/JS frontend, no build step
- **Results:** see [`reports/report.md`](reports/report.md) and [`reports/sweep.json`](reports/sweep.json)

## Run the site

```bash
cd ~/spam-detector
source .venv/bin/activate          # or: python -m venv .venv && pip install -r requirements.txt
uvicorn app.main:app --port 8000
```

Open <http://127.0.0.1:8000>.

| Tab | What it does |
|---|---|
| **Check an email** | Paste a subject/body → spam probability, verdict, contributing terms, and the automatically learned threshold. |
| **Batch CSV** | Upload a CSV with `text`, or `subject` + `body`. Returns the rows with `prediction` and `spam_probability` appended. |
| **Model** | Test metrics, confusion matrix, full sweep table, and the strongest spam/legitimate terms. |

API: `GET /api/health`, `POST /api/predict`, `POST /api/batch`, `GET /api/metrics`, `GET /api/examples`.

## Project layout

```
spam-detector/
├── app/
│   ├── main.py                 FastAPI app
│   ├── templates/index.html
│   └── static/{style.css,app.js}
├── atlas-engine/               vendored Atlas Engine (training engine)
├── data/
│   ├── emails.csv              full synthetic corpus
│   ├── train.csv               official train split + `text` column
│   ├── validation.csv          official validation split
│   └── test.csv                official test split (used once)
├── model/                      winning Atlas run
│   ├── pipeline.joblib         CountVectorizer + LogisticRegression
│   ├── meta.json               classes + text column
│   ├── metrics.json            engine metrics
│   ├── top_terms.json          coefficients for the UI
│   └── predicted_test.csv
├── reports/
│   ├── sweep.json              every run + official test evaluation
│   └── report.md               human-readable training report
├── runs/                       all raw Atlas run folders
└── scripts/
    ├── generate_dataset.py     build the synthetic corpus
    ├── prepare_data.py         split + combine subject/body into `text`
    ├── train_sweep.py          train via atlas_worker, select, evaluate
    └── make_report.py          render reports/report.md
```

## Reproduce training

```bash
source .venv/bin/activate
python scripts/generate_dataset.py
python scripts/prepare_data.py
python scripts/train_sweep.py
python scripts/make_report.py
```

`train_sweep.py` writes the standard Atlas job file and runs `python -m atlas_worker train <job.json>` for each configuration, then calls `python -m atlas_worker predict` on the official validation and test splits.

## Using the real ChatGPT dataset instead

The ChatGPT library link needs a logged-in session, so this build uses a synthetic stand-in with the same schema and split design. To swap in the real file:

1. Download `email_spam_detection_synthetic.csv` from the conversation.
2. Save it as `data/emails.csv` (columns must include `subject`, `body`, `label`, `split`).
3. Re-run `prepare_data.py`, `train_sweep.py`, `make_report.py`.

## Notes and caveats

- Every writing template appears in exactly one split, so the test templates were never seen in training.
- Model and threshold selection use validation; the previously inspected synthetic test is exploratory.
- A perfect score on synthetic data does **not** mean perfect real-inbox accuracy. Evaluate on labelled real mail before production use.
- The vendored `atlas_worker/nlp.py` moves its `torch` import below the `text_sk` branch, so classic sklearn text models do not require PyTorch.
