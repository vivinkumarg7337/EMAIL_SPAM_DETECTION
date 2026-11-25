"""Render reports/report.md from the sweep output. Run after train_sweep.py."""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SWEEP = ROOT / "reports" / "sweep.json"
MODEL = ROOT / "model"

NAME = {386: "CountVectorizer + LogisticRegression", 387: "TF-IDF + LogisticRegression"}


def pct(v):
    return f"{v * 100:.2f}%"


def main():
    s = json.loads(SWEEP.read_text())
    best = s["best"]
    t = best["test"]
    terms = json.loads((MODEL / "top_terms.json").read_text())
    cm = t["confusion_matrix"]

    lines = []
    lines.append("# Email spam detection with Atlas Engine\n")
    lines.append("Trained with **Atlas Engine** (`atlas_worker`, text-classification entries "
                 "#386 and #387) and served by a small FastAPI website.\n")
    lines.append("## Dataset\n")
    lines.append(f"- 2,400 synthetic English emails, 50/50 spam and legitimate (`data/emails.csv`).")
    lines.append("- Columns: `subject`, `body`, `label` (0/1), `split` (train 960+960, validation 120+120, test 120+120).")
    lines.append('- ' + s['dataset']['note'])
    lines.append('- Shared class-specific greetings and closings make this synthetic task easier than real email.\n')
    lines.append("## Sweep\n")
    lines.append('Models trained through Atlas Engine on training rows; model and threshold selected on validation accuracy, with fewer false alarms as a tie-breaker. The worker fits on 70% of the supplied training rows and uses the rest for internal checks.\n')
    lines.append(f"Learned deployment threshold: **{best['threshold']:.8f}**. Test results below are exploratory because the generator was previously revised after test inspection.\n")
    lines.append("| model | ngram | C | vocabulary | val F1 | val accuracy | ROC-AUC | train |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for r in s["runs"]:
        c, v = r["config"], r["val"]
        star = " **← best**" if r["run_id"] == best["run_id"] else ""
        lines.append(f"| {NAME[c['entry']]}{star} | {c['ngram']} | {c['C']} | {c['max_features']:,} | "
                     f"{pct(v['f1_macro'])} | {pct(v['accuracy'])} | {v['roc_auc']:.4f} | {r['train_seconds']}s |")
    lines.append("")
    lines.append("## Selected model\n")
    lines.append(f"`{NAME[best['config']['entry']]}` (Atlas entry #{best['config']['entry']}), "
                 f"ngram={best['config']['ngram']}, vocabulary={best['config']['max_features']:,}, C={best['config']['C']}.\n")
    lines.append("### Exploratory synthetic test (240 emails; not an independent benchmark)\n")
    lines.append("| metric | value |")
    lines.append("|---|---|")
    lines.append(f"| Accuracy | {pct(t['accuracy'])} |")
    lines.append(f"| Balanced accuracy | {pct(t['balanced_accuracy'])} |")
    lines.append(f"| Macro precision | {pct(t['precision_macro'])} |")
    lines.append(f"| Macro recall | {pct(t['recall_macro'])} |")
    lines.append(f"| Macro F1 | {pct(t['f1_macro'])} |")
    lines.append(f"| Spam F1 | {pct(t['f1_spam'])} |")
    lines.append(f"| ROC-AUC | {t['roc_auc']:.4f} |")
    lines.append(f"| PR-AUC | {t['pr_auc']:.4f} |")
    lines.append("")
    lines.append("### Confusion matrix\n")
    lines.append("| | predicted legitimate | predicted spam |")
    lines.append("|---|---|---|")
    lines.append(f"| **actual legitimate** | {cm[0][0]} | {cm[0][1]} |")
    lines.append(f"| **actual spam** | {cm[1][0]} | {cm[1][1]} |")
    lines.append("")
    lines.append("### Strongest terms\n")
    lines.append("- Spam: " + ", ".join(f"`{x['term']}`" for x in terms["spam"][:12]))
    lines.append("- Legitimate: " + ", ".join(f"`{x['term']}`" for x in terms["legitimate"][:12]))
    lines.append("")
    lines.append("## Reproduce\n")
    lines.append("```bash")
    lines.append("python scripts/generate_dataset.py    # build the synthetic corpus")
    lines.append("python scripts/prepare_data.py        # split + combine subject/body into text")
    lines.append("python scripts/train_sweep.py         # train via atlas_worker, evaluate, save model/")
    lines.append("uvicorn app.main:app --port 8000      # serve the website")
    lines.append("```")
    lines.append("")
    lines.append("## Caveats\n")
    lines.append("- These scores describe a **synthetic** corpus with template-disjoint splits. A 100% test score "
                 "does not transfer to a real inbox.")
    lines.append("- Before production use, retrain on labelled real mail and review false positives, which are "
                 "more costly than false negatives for most inboxes.")
    lines.append("- The shipped model predicts on `subject + \"\\n\" + body`; the website does this automatically.")
    lines.append("- Atlas Engine's sklearn text predictor imports `torch` before its `text_sk` branch; the vendored "
                 "copy moves that import below the branch so classic text models need no PyTorch install.")
    lines.append("")

    out = ROOT / "reports" / "report.md"
    out.write_text("\n".join(lines))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
