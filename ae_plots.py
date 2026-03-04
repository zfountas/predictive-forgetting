"""
Plot single-run results: Phase 3 accuracy curves and T-sweep analysis.

Usage:
  python ae_plots.py runs/mnist_ponder_s123
"""
import json, csv, os, sys
import matplotlib.pyplot as plt

run = sys.argv[1] if len(sys.argv) > 1 else "./runs/exp1"

def load_json(p):
    with open(p, "r") as f: return json.load(f)

# --- Phase 2 history ---
# Phase 2 summary (if you load it)
p2_json = os.path.join(run, "phase2_summary.json")
if os.path.exists(p2_json):
    with open(p2_json, "r") as f:
        p2 = json.load(f)
    print("Phase 2:", p2)
else:
    print("[plot] phase2_summary.json not found; skipping Phase 2 printout.")

# --- Phase 3 curves ---
# Phase 3 history (optional)
p3_hist_path = os.path.join(run, "phase3_history.json")
if os.path.exists(p3_hist_path):
    p3_hist = json.load(open(p3_hist_path))
    epochs = [h["epoch"] for h in p3_hist]
    train_acc0 = [h["train_acc_t0"] for h in p3_hist]
    train_accR = [h["train_acc_ref"] for h in p3_hist]
    val_acc0   = [h["val_acc_t0"]   for h in p3_hist]
    val_accR   = [h["val_acc_ref"]  for h in p3_hist]

    plt.figure()
    plt.plot(epochs, val_acc0, label="val t=0")
    plt.plot(epochs, val_accR, label="val refined")
    plt.plot(epochs, train_acc0, "--", label="train t=0")
    plt.plot(epochs, train_accR, "--", label="train refined")
    plt.xlabel("Epoch"); plt.ylabel("Accuracy"); plt.title("Phase 3 accuracy"); plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(run, "phase3_acc.png"), dpi=150)
    plt.show()

else:
    print("[plot] phase3_history.json not found; skipping epoch curves.")




# --- T-sweep (if present): plot acc and gap vs T_eval ---
tsweep_p = os.path.join(run, "phase3_Tsweep.csv")
if os.path.exists(tsweep_p):
    import csv
    with open(tsweep_p, "r") as f:
        rd = csv.DictReader(f)
        rows = list(rd)

    T = sorted({int(r["T_eval"]) for r in rows})
    train_acc, test_acc, gap = [], [], []
    for t in T:
        r_tr = next(r for r in rows if int(r["T_eval"])==t and r["split"]=="train")
        r_te = next(r for r in rows if int(r["T_eval"])==t and r["split"]=="test")
        tr_acc = float(r_tr["acc"]); te_acc = float(r_te["acc"])
        train_acc.append(tr_acc); test_acc.append(te_acc)
        # gap = test_err - train_err = (1-te_acc) - (1-tr_acc) = tr_acc - te_acc
        gap.append((1.0 - te_acc) - (1.0 - tr_acc))

    plt.figure()
    plt.subplot(2,1,1)
    plt.plot(T, test_acc, "o-", label="test acc")
    plt.plot(T, train_acc, "o--", label="train acc")
    plt.xlabel("Refinement steps (T_eval)")
    plt.ylabel("Accuracy")
    plt.title("Accuracy vs refinement steps")
    plt.xticks(T); plt.legend(); plt.tight_layout()

    plt.subplot(2,1,2)
    plt.plot(T, gap, "o-")
    plt.axhline(0, linestyle="--", linewidth=1)
    plt.xlabel("Refinement steps (T_eval)")
    plt.ylabel("Generalisation gap (test_err - train_err)")
    plt.title("Gap vs refinement steps")
    plt.xticks(T); plt.tight_layout()
    plt.savefig(os.path.join(run, "phase3_Tsweep_acc_gap.png"), dpi=150)
    plt.show()
else:
    print("[plot] phase3_Tsweep.csv not found; skipping T-sweep plots.")


print("Saved plots in:", run)
