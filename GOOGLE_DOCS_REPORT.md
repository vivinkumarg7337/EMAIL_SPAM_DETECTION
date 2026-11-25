# Email Spam Detection — Final Report

## 1. Project overview

This project trains an email spam classifier with Atlas Engine and serves it through a small FastAPI website. The website supports:

- single-email spam or legitimate classification;
- CSV batch prediction;
- display of the automatically learned decision threshold and contributing terms;
- a model metrics dashboard.

Repository: `https://github.com/vivinkumarg7337/EMAIL_SPAM_DETECTION`
Local project: `/home/vivin-kumar/spam-detector`

## 2. Dataset

The supplied ChatGPT library file could not be downloaded without an authenticated session. The folder supplied instead, `Email Spam Dataset_files`, contained saved webpage assets (CSS, JavaScript and icons), not rows of email data.

A synthetic stand-in with the same stated schema and split design was generated:

| Split | Legitimate | Spam | Total |
|---|---:|---:|---:|
| Training | 960 | 960 | 1,920 |
| Validation | 120 | 120 | 240 |
| Test | 120 | 120 | 240 |
| **Total** | **1,200** | **1,200** | **2,400** |

Columns are `subject`, `body`, `label` (0 = legitimate, 1 = spam) and `split`. The website combines `subject` and `body` into one text field. Each writing template is confined to one split. Ten percent of spam is soft-sell outreach and eight percent of legitimate mail is promotional, adding realistic ambiguity.

**Important limitation:** the corpus generator was revised once after examining test errors. The test split is therefore exploratory and is not an untouched measure of generalization. Real labeled email is required to establish production accuracy.

## 3. Training method

Atlas Engine's `atlas_worker` was invoked directly. Thirty configurations were searched:

- entries 386 (CountVectorizer + Logistic Regression) and 387 (TF-IDF + Logistic Regression);
- word n-grams of length 1, 2 and 3;
- `C` values of 0.1, 0.5, 1.0, 5.0 and 10.0;
- vocabulary size of 20,000.

The threshold for each configuration was selected on validation accuracy. Ties were broken by fewer false positives, then by proximity to 0.5. The configuration with the best validation score was then used to score the exploratory test split. Atlas Engine internally fits on 70% of the supplied training rows and uses the remainder for its own checks.

## 4. Selected model

| Setting | Value |
|---|---|
| Atlas entry | 387 — TF-IDF + Logistic Regression |
| Word n-grams | 1 |
| Vocabulary | 20,000 |
| Regularization `C` | 0.1 |
| Learned decision threshold | **0.41801422** |

Predict spam when `P(spam) >= 0.41801422`; otherwise predict legitimate. Both the single-email API and batch API use this threshold.

## 5. Results

| Metric | Validation | Exploratory synthetic test |
|---|---:|---:|
| Accuracy | 100.00% | 95.00% |
| Macro F1 | 100.00% | 94.99% |
| ROC-AUC | 1.0000 | 0.9984 |
| PR-AUC | 1.0000 | 0.9985 |

### Test confusion matrix

|  | Predicted legitimate | Predicted spam |
|---|---:|---:|
| **Actually legitimate** | 109 | 11 |
| **Actually spam** | 1 | 119 |

The model caught 119 of 120 spam messages and incorrectly flagged 11 of 120 legitimate messages. This reduces false negatives relative to the previous model but increases false positives.

## 6. Mathematical model

The earlier proposed formula is Multinomial Naive Bayes with Laplace smoothing:

\[
P(w\mid c)=\frac{\operatorname{count}(w,c)+1}{\operatorname{total}(c)+|V|},\qquad
\operatorname{weight}(w)=\frac{P(w\mid\mathrm{spam})}{P(w\mid\mathrm{legitimate})}
\]

\[
\operatorname{odds}=
\frac{N_{\mathrm{spam}}}{N_{\mathrm{legitimate}}}
\prod_{w}\operatorname{weight}(w)
\]

For numerical stability, use log odds:

\[
\log\operatorname{odds}=
\log\frac{N_{\mathrm{spam}}}{N_{\mathrm{legitimate}}}
+\sum_w\log\operatorname{weight}(w)
\]

The deployed Atlas model uses TF-IDF and logistic regression rather than this Naive Bayes formula. Its threshold was learned from validation data.

## 7. Website

Start with:

```bash
cd ~/spam-detector
./run.sh
```

Then open `http://127.0.0.1:8000/`.

| Endpoint | Purpose |
|---|---|
| `GET /api/health` | Model status and deployed threshold |
| `POST /api/predict` | Classify one email |
| `POST /api/batch` | Classify an uploaded CSV |
| `GET /api/metrics` | Sweep and final metrics |
| `GET /api/examples` | Sample emails |

## 8. Reproducing the result

```bash
python scripts/generate_dataset.py
python scripts/prepare_data.py
python scripts/train_sweep.py
python scripts/make_report.py
./run.sh
python scripts/check_deployment.py
```

Previous models and reports are retained under `backups/`.

## 9. Conclusion

Thirty Atlas Engine configurations were trained and the decision threshold was learned automatically. The selected model reaches 100% validation accuracy but 95% on the already-inspected synthetic test split. That gap demonstrates why the synthetic result cannot be treated as real-world accuracy. The next step is to obtain the original CSV or collect labeled real emails, freeze the data before evaluation, and repeat the search.
