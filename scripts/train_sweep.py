"""Train Atlas Engine text classifiers, select by the official validation split,
then report on the untouched official test split."""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn import metrics as M

ROOT = Path(__file__).resolve().parents[1]
ENGINE = ROOT / "atlas-engine"
RUNS = ROOT / "runs"
REPORTS = ROOT / "reports"
PY = ROOT / ".venv" / "bin" / "python"
ENTRY = {386: "CountVectorizer + LogisticRegression", 387: "TF-IDF + LogisticRegression"}
STOPWORDS = set("""a about above after again against all am an and any are as at be because been before
being below between both but by can cannot could did do does doing down during each few for from further
had has have having he her here hers herself him himself his how i if in into is it its itself me more most
my myself no nor not now of off on once only or other our ours ourselves out over own same she should so
some such than that the their theirs them themselves then there these they this those through to too under
until up very was we were what when where which while who whom why will with you your yours yourself
yourselves""".split())

OLD_CONFIGS = [
    dict(entry=387, ngram=1, max_features=20000, C=0.5),
    dict(entry=387, ngram=1, max_features=20000, C=1.0),
    dict(entry=387, ngram=1, max_features=20000, C=2.0),
    dict(entry=387, ngram=1, max_features=50000, C=1.0),
    dict(entry=387, ngram=2, max_features=20000, C=1.0),
    dict(entry=387, ngram=2, max_features=20000, C=2.0),
    dict(entry=386, ngram=1, max_features=20000, C=1.0),
    dict(entry=386, ngram=2, max_features=20000, C=1.0),
]


CONFIGS = [dict(entry=e, ngram=n, max_features=20000, C=c)
           for e in (386, 387) for n in (1, 2, 3) for c in (0.1, 0.5, 1.0, 5.0, 10.0)]


def choose_threshold(y, p):
    unique = np.unique(p)
    candidates = np.unique(np.r_[0.0, 0.5, 1.0, (unique[:-1] + unique[1:]) / 2])
    def rank(t):
        pred = p >= t
        return (int((pred == y).sum()), -int(((y == 0) & pred).sum()), -abs(t - 0.5))
    return float(max(candidates, key=rank))


def env():
    e = {k: v for k, v in os.environ.items() if k not in ("PYTHONHOME", "VIRTUAL_ENV", "LD_LIBRARY_PATH")}
    e["PYTHONPATH"] = str(ENGINE)
    e["PYTHONUNBUFFERED"] = "1"
    return e


def run(cmd, tail=20):
    p = subprocess.run(cmd, cwd=ROOT, env=env(), capture_output=True, text=True)
    if p.returncode != 0:
        print("\n".join((p.stdout + p.stderr).splitlines()[-tail:]))
        raise SystemExit(f"command failed ({p.returncode}): {' '.join(map(str, cmd))}")
    return p


def scores(y, pred, proba):
    out = dict(
        accuracy=float(M.accuracy_score(y, pred)),
        balanced_accuracy=float(M.balanced_accuracy_score(y, pred)),
        precision_macro=float(M.precision_score(y, pred, average="macro", zero_division=0)),
        recall_macro=float(M.recall_score(y, pred, average="macro", zero_division=0)),
        f1_macro=float(M.f1_score(y, pred, average="macro", zero_division=0)),
        f1_spam=float(M.f1_score(y, pred, pos_label=1, zero_division=0)),
        roc_auc=float(M.roc_auc_score(y, proba)),
        pr_auc=float(M.average_precision_score(y, proba)),
    )
    cm = M.confusion_matrix(y, pred, labels=[0, 1])
    out["confusion_matrix"] = cm.tolist()
    out["per_class"] = {
        name: dict(
            precision=float(M.precision_score(y == k, pred == k, zero_division=0)),
            recall=float(M.recall_score(y == k, pred == k, zero_division=0)),
            f1=float(M.f1_score(y == k, pred == k, zero_division=0)),
            support=int((y == k).sum()),
        )
        for k, name in ((0, "legitimate"), (1, "spam"))
    }
    return out


def evaluate(run_dir, csv_path, tag):
    pred_path = run_dir / f"pred_{tag}.csv"
    run([PY, "-m", "atlas_worker", "predict", str(run_dir), str(pred_path), "--input", str(csv_path)])
    src = pd.read_csv(csv_path)
    pred = pd.read_csv(pred_path)
    y = src["label"].to_numpy()
    p = pred["proba_1"].to_numpy()
    lab = pred["prediction"].astype(int).to_numpy()
    return scores(y, lab, p), pred


def main():
    REPORTS.mkdir(exist_ok=True)
    RUNS.mkdir(exist_ok=True)
    stamp = datetime.now().strftime('%Y%m%d-%H%M%S')
    backup = ROOT / 'backups' / stamp
    backup.mkdir(parents=True)
    for name in ('model', 'reports'):
        shutil.copytree(ROOT / name, backup / name)
    yval = pd.read_csv(ROOT / 'data' / 'validation.csv')['label'].to_numpy()
    rows, best = [], None
    for i, cfg in enumerate(CONFIGS, 1):
        rid = f"tuned-{stamp}-{i:02d}-e{cfg['entry']}-n{cfg['ngram']}-c{cfg['C']}"
        job = dict(
            run_id=rid,
            run_dir=str(RUNS / rid),
            entry=cfg["entry"],
            name=ENTRY[cfg["entry"]],
            task="text_classification",
            dataset=dict(path=str(ROOT / "data" / "train.csv"), target="label",
                         time_column=None, text_column="text"),
            params=dict(max_features=cfg["max_features"], ngram=cfg["ngram"], C=cfg["C"]),
            split=dict(val=0.15, test=0.15, seed=42),
            device="cpu",
            threads=4,
            batch_size=0,
        )
        job_path = RUNS / f"{rid}.job.json"
        job_path.write_text(json.dumps(job, indent=1))
        print(f"[{i}/{len(CONFIGS)}] {rid}", flush=True)
        run([PY, "-m", "atlas_worker", "train", str(job_path)])
        run_dir = RUNS / rid
        summary = json.loads((run_dir / "metrics.json").read_text())
        default_val, prediction = evaluate(run_dir, ROOT / "data" / "validation.csv", "val")
        p = prediction['proba_1'].to_numpy()
        threshold = choose_threshold(yval, p)
        val = scores(yval, (p >= threshold).astype(int), p)
        rows.append(dict(config=cfg, run_id=rid, run_dir=str(run_dir), engine=summary,
                         val=val, default_val=default_val, threshold=threshold, train_seconds=summary["train_seconds"]))
        print(f"    engine val f1={summary['val']['f1_macro']:.4f} | official val "
              f"f1={val['f1_macro']:.4f} acc={val['accuracy']:.4f} roc={val['roc_auc']:.4f} "
              f"({summary['train_seconds']}s)", flush=True)
        def rank(row):
            v = row['val']
            return (v['accuracy'], -v['confusion_matrix'][0][1], v['f1_macro'], -abs(row['threshold'] - 0.5))
        if best is None or rank(rows[-1]) > rank(best):
            best = rows[-1]

    best_dir = Path(best["run_dir"])
    test, pred = evaluate(best_dir, ROOT / "data" / "test.csv", "test")
    ytest = pd.read_csv(ROOT / 'data' / 'test.csv')['label'].to_numpy()
    test = scores(ytest, (pred.proba_1.to_numpy() >= best['threshold']).astype(int), pred.proba_1.to_numpy())
    pred['prediction'] = (pred.proba_1 >= best['threshold']).astype(int)
    pred.to_csv(best_dir / 'pred_test.csv', index=False)

    # per-config test scores for the report table
    for r in rows:
        if r is best:
            r["test"] = test
            continue
        # only the winner gets a full test pass; the rest are compared on validation
        r["test"] = None

    # copy the winning model next to the app
    model_dir = ROOT / "model"
    if model_dir.exists():
        shutil.rmtree(model_dir)
    model_dir.mkdir(parents=True)
    for f in ("pipeline.joblib", "meta.json"):
        shutil.copy2(best_dir / "model" / f, model_dir / f)
    for f in ("metrics.json", "charts.json", "job.json"):
        shutil.copy2(best_dir / f, model_dir / f)
    shutil.copy2(best_dir / "pred_test.csv", model_dir / "predicted_test.csv")
    (model_dir / 'decision.json').write_text(json.dumps(dict(threshold=best['threshold'],
        objective='validation accuracy; ties prefer fewer false positives, then threshold nearest 0.5'), indent=2))

    # coefficients for the UI's "why" panel
    pipe = joblib.load(model_dir / "pipeline.joblib")
    vec, clf = pipe.named_steps["vec"], pipe.named_steps["model"]
    names = vec.get_feature_names_out()
    coef = clf.coef_[0]
    order = np.argsort(coef)
    def clean(term):
        t = str(term)
        return len(t) >= 3 and t not in STOPWORDS and not t.isdigit()
    spam_terms = [dict(term=str(names[i]), weight=float(coef[i])) for i in order[::-1] if clean(names[i])][:25]
    legit_terms = [dict(term=str(names[i]), weight=float(coef[i])) for i in order if clean(names[i])][:25]
    top_terms = dict(spam=spam_terms, legitimate=legit_terms)
    (model_dir / "top_terms.json").write_text(json.dumps(top_terms, indent=1))

    sweep = dict(
        generated=stamp,
        dataset=dict(
            path="data/emails.csv", rows=2400,
            train=1920, validation=240, test=240,
            note="Generated stand-in, not the user CSV. Previously revised after test inspection; test scores are exploratory, not independent. Data frozen for this search.",
        ),
        primary_metric="validation accuracy, then fewer false positives; threshold selected on validation only",
        best=dict(run_id=best["run_id"], config=best["config"], threshold=best['threshold'], validation=best["val"], test=test),
        runs=rows,
        test_predictions="model/predicted_test.csv",
    )
    (REPORTS / "sweep.json").write_text(json.dumps(sweep, indent=1))
    (REPORTS / "evaluation.json").write_text(json.dumps(dict(best=best["config"], validation=best["val"], test=test), indent=1))

    print("\n=== best ===")
    print(json.dumps(best["config"]))
    print(f"Selected threshold: {best['threshold']:.8f}")
    print(f"official val : f1={best['val']['f1_macro']:.4f} acc={best['val']['accuracy']:.4f} roc={best['val']['roc_auc']:.4f}")
    print(f"official test: f1={test['f1_macro']:.4f} acc={test['accuracy']:.4f} roc={test['roc_auc']:.4f} pr={test['pr_auc']:.4f}")
    print(f"confusion    : {test['confusion_matrix']}")
    print(f"model copied -> {model_dir}")


if __name__ == "__main__":
    main()
