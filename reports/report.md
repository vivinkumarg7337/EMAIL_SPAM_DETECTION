# Email spam detection with Atlas Engine

Trained with **Atlas Engine** (`atlas_worker`, text-classification entries #386 and #387) and served by a small FastAPI website.

## Dataset

- 2,400 synthetic English emails, 50/50 spam and legitimate (`data/emails.csv`).
- Columns: `subject`, `body`, `label` (0/1), `split` (train 960+960, validation 120+120, test 120+120).
- Generated stand-in, not the user CSV. Previously revised after test inspection; test scores are exploratory, not independent. Data frozen for this search.
- Shared class-specific greetings and closings make this synthetic task easier than real email.

## Sweep

Models trained through Atlas Engine on training rows; model and threshold selected on validation accuracy, with fewer false alarms as a tie-breaker. The worker fits on 70% of the supplied training rows and uses the rest for internal checks.

Learned deployment threshold: **0.41801422**. Test results below are exploratory because the generator was previously revised after test inspection.

| model | ngram | C | vocabulary | val F1 | val accuracy | ROC-AUC | train |
|---|---|---|---|---|---|---|---|
| CountVectorizer + LogisticRegression | 1 | 0.1 | 20,000 | 100.00% | 100.00% | 1.0000 | 1.2s |
| CountVectorizer + LogisticRegression | 1 | 0.5 | 20,000 | 100.00% | 100.00% | 1.0000 | 1.15s |
| CountVectorizer + LogisticRegression | 1 | 1.0 | 20,000 | 100.00% | 100.00% | 1.0000 | 1.16s |
| CountVectorizer + LogisticRegression | 1 | 5.0 | 20,000 | 100.00% | 100.00% | 1.0000 | 1.23s |
| CountVectorizer + LogisticRegression | 1 | 10.0 | 20,000 | 100.00% | 100.00% | 1.0000 | 1.29s |
| CountVectorizer + LogisticRegression | 2 | 0.1 | 20,000 | 100.00% | 100.00% | 1.0000 | 1.38s |
| CountVectorizer + LogisticRegression | 2 | 0.5 | 20,000 | 100.00% | 100.00% | 1.0000 | 1.36s |
| CountVectorizer + LogisticRegression | 2 | 1.0 | 20,000 | 100.00% | 100.00% | 1.0000 | 1.34s |
| CountVectorizer + LogisticRegression | 2 | 5.0 | 20,000 | 100.00% | 100.00% | 1.0000 | 1.41s |
| CountVectorizer + LogisticRegression | 2 | 10.0 | 20,000 | 100.00% | 100.00% | 1.0000 | 1.49s |
| CountVectorizer + LogisticRegression | 3 | 0.1 | 20,000 | 100.00% | 100.00% | 1.0000 | 3.91s |
| CountVectorizer + LogisticRegression | 3 | 0.5 | 20,000 | 100.00% | 100.00% | 1.0000 | 2.42s |
| CountVectorizer + LogisticRegression | 3 | 1.0 | 20,000 | 100.00% | 100.00% | 1.0000 | 2.56s |
| CountVectorizer + LogisticRegression | 3 | 5.0 | 20,000 | 100.00% | 100.00% | 1.0000 | 2.59s |
| CountVectorizer + LogisticRegression | 3 | 10.0 | 20,000 | 100.00% | 100.00% | 1.0000 | 2.56s |
| TF-IDF + LogisticRegression **← best** | 1 | 0.1 | 20,000 | 100.00% | 100.00% | 1.0000 | 1.63s |
| TF-IDF + LogisticRegression | 1 | 0.5 | 20,000 | 100.00% | 100.00% | 1.0000 | 1.21s |
| TF-IDF + LogisticRegression | 1 | 1.0 | 20,000 | 100.00% | 100.00% | 1.0000 | 1.32s |
| TF-IDF + LogisticRegression | 1 | 5.0 | 20,000 | 100.00% | 100.00% | 1.0000 | 1.25s |
| TF-IDF + LogisticRegression | 1 | 10.0 | 20,000 | 100.00% | 100.00% | 1.0000 | 1.25s |
| TF-IDF + LogisticRegression | 2 | 0.1 | 20,000 | 100.00% | 100.00% | 1.0000 | 1.43s |
| TF-IDF + LogisticRegression | 2 | 0.5 | 20,000 | 100.00% | 100.00% | 1.0000 | 1.39s |
| TF-IDF + LogisticRegression | 2 | 1.0 | 20,000 | 100.00% | 100.00% | 1.0000 | 1.43s |
| TF-IDF + LogisticRegression | 2 | 5.0 | 20,000 | 100.00% | 100.00% | 1.0000 | 1.36s |
| TF-IDF + LogisticRegression | 2 | 10.0 | 20,000 | 100.00% | 100.00% | 1.0000 | 1.41s |
| TF-IDF + LogisticRegression | 3 | 0.1 | 20,000 | 100.00% | 100.00% | 1.0000 | 2.06s |
| TF-IDF + LogisticRegression | 3 | 0.5 | 20,000 | 100.00% | 100.00% | 1.0000 | 2.4s |
| TF-IDF + LogisticRegression | 3 | 1.0 | 20,000 | 100.00% | 100.00% | 1.0000 | 2.72s |
| TF-IDF + LogisticRegression | 3 | 5.0 | 20,000 | 100.00% | 100.00% | 1.0000 | 3.1s |
| TF-IDF + LogisticRegression | 3 | 10.0 | 20,000 | 100.00% | 100.00% | 1.0000 | 3.52s |

## Selected model

`TF-IDF + LogisticRegression` (Atlas entry #387), ngram=1, vocabulary=20,000, C=0.1.

### Exploratory synthetic test (240 emails; not an independent benchmark)

| metric | value |
|---|---|
| Accuracy | 95.00% |
| Balanced accuracy | 95.00% |
| Macro precision | 95.31% |
| Macro recall | 95.00% |
| Macro F1 | 94.99% |
| Spam F1 | 95.20% |
| ROC-AUC | 0.9984 |
| PR-AUC | 0.9985 |

### Confusion matrix

| | predicted legitimate | predicted spam |
|---|---|---|
| **actual legitimate** | 109 | 11 |
| **actual spam** | 1 | 119 |

### Strongest terms

- Spam: `prize`, `http`, `free`, `selected`, `secure`, `address`, `won`, `unpaid`, `received`, `payment`, `invoice`, `details`
- Legitimate: `standup`, `reply`, `next`, `anything`, `please`, `thanks`, `regards`, `help`, `looks`, `wrong`, `message`, `review`

## Reproduce

```bash
python scripts/generate_dataset.py    # build the synthetic corpus
python scripts/prepare_data.py        # split + combine subject/body into text
python scripts/train_sweep.py         # train via atlas_worker, evaluate, save model/
uvicorn app.main:app --port 8000      # serve the website
```

## Caveats

- These scores describe a **synthetic** corpus with template-disjoint splits. A 100% test score does not transfer to a real inbox.
- Before production use, retrain on labelled real mail and review false positives, which are more costly than false negatives for most inboxes.
- The shipped model predicts on `subject + "\n" + body`; the website does this automatically.
- Atlas Engine's sklearn text predictor imports `torch` before its `text_sk` branch; the vendored copy moves that import below the branch so classic text models need no PyTorch install.
