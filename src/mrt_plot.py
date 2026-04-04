# mrt_eval_plot.py — load cached eval results and plot
# python src/mrt_eval_plot.py --domain emea --lambdas 0.1 0.5 0.9

import argparse
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from config import *

parser = argparse.ArgumentParser()
parser.add_argument("--domain",  required=True, choices=["emea", "news_commentary", "opus_books"])
parser.add_argument("--lambdas", nargs="+", type=float, default=[0.1, 0.3, 0.5, 0.7, 0.9])
args = parser.parse_args()

MRT_DIR  = RUN_DIR / "mrt" / args.domain
EVAL_DIR = MRT_DIR / "eval"
EVAL_DIR.mkdir(parents=True, exist_ok=True)

DOMAIN_LABELS = {
    "emea":            "EMEA (Medical)",
    "news_commentary": "News Commentary",
    "opus_books":      "Opus Books (Literary)",
}
label = DOMAIN_LABELS[args.domain]

# ── Load baselines ────────────────────────────────────────────────────────────

raw_front_path  = RUN_DIR / "baseline"  / "results" / f"{args.domain}_af_front.csv"
raw_greedy_path = RUN_DIR / "baseline"  / "data"    / f"{args.domain}_greedy_scored.jsonl"
ft_front_path   = RUN_DIR / "finetuned" / "results" / f"{args.domain}_af_front.csv"
ft_greedy_path  = RUN_DIR / "finetuned" / "data"    / f"{args.domain}_greedy_scored.jsonl"

if raw_front_path.exists() and raw_greedy_path.exists():
    raw_front     = pd.read_csv(raw_front_path)
    raw_greedy_df = pd.read_json(raw_greedy_path, orient="records", lines=True)
    raw_greedy    = {"chrf": raw_greedy_df["chrf"].mean(), "fluency": raw_greedy_df["fluency"].mean()}
    print(f"Raw baseline: chrF={raw_greedy['chrf']:.2f}, fluency={raw_greedy['fluency']:.3f}")
else:
    print("WARNING: Raw baseline files missing — skipping.")
    raw_front = raw_greedy = None

if not ft_front_path.exists() or not ft_greedy_path.exists():
    raise FileNotFoundError(f"Missing fine-tuned baseline files for {args.domain}.")

ft_front     = pd.read_csv(ft_front_path)
ft_greedy_df = pd.read_json(ft_greedy_path, orient="records", lines=True)
ft_greedy    = {"chrf": ft_greedy_df["chrf"].mean(), "fluency": ft_greedy_df["fluency"].mean()}
print(f"Fine-tuned baseline: chrF={ft_greedy['chrf']:.2f}, fluency={ft_greedy['fluency']:.3f}")

# ── Load λ results (whatever's cached) ───────────────────────────────────────

lambda_results = {}
missing = []

for lam in args.lambdas:
    lam_tag       = str(lam).replace(".", "p")
    data_subdir   = EVAL_DIR / f"lambda_{lam_tag}"
    cached_front  = data_subdir / f"{args.domain}_af_front.csv"
    cached_greedy = data_subdir / f"{args.domain}_greedy_scored.jsonl"

    if cached_front.exists() and cached_greedy.exists():
        front_df  = pd.read_csv(cached_front)
        greedy_df = pd.read_json(cached_greedy, orient="records", lines=True)
        greedy    = {"chrf": greedy_df["chrf"].mean(), "fluency": greedy_df["fluency"].mean()}
        lambda_results[lam] = (front_df, greedy)
        print(f"[λ={lam}] Loaded — greedy chrF={greedy['chrf']:.2f}, fluency={greedy['fluency']:.3f}")
    else:
        missing.append(lam)
        print(f"[λ={lam}] Cache incomplete — skipping.")

if not lambda_results:
    print("No lambda results available to plot. Run mrt_eval.py first.")
    exit(0)

if missing:
    print(f"\nNote: results missing for λ={missing} — not included in plot.")

# ── Plot ──────────────────────────────────────────────────────────────────────

fig, ax = plt.subplots(figsize=(10, 6))

if raw_front is not None:
    ax.plot(raw_front["chrf"], raw_front["fluency"],
            color="grey", linewidth=2, linestyle=":", label="Raw baseline oracle front", zorder=2)
    ax.scatter([raw_greedy["chrf"]], [raw_greedy["fluency"]],
               marker="D", s=150, color="grey", zorder=5, label="Raw baseline greedy")

ax.plot(ft_front["chrf"], ft_front["fluency"],
        color="black", linewidth=2.5, linestyle="--", label="Fine-tuned baseline oracle front", zorder=3)
ax.scatter([ft_greedy["chrf"]], [ft_greedy["fluency"]],
           marker="*", s=250, color="black", zorder=6, label="Fine-tuned baseline greedy")

lam_vals = sorted(lambda_results.keys())
cmap     = cm.coolwarm
norm     = plt.Normalize(vmin=min(lam_vals), vmax=max(lam_vals))

for lam in lam_vals:
    front_df, greedy = lambda_results[lam]
    colour = cmap(norm(lam))
    ax.plot(front_df["chrf"], front_df["fluency"],
            color=colour, linewidth=1.5, linestyle="-", alpha=0.6)
    ax.scatter([greedy["chrf"]], [greedy["fluency"]],
               marker="o", s=120, color=colour, zorder=5,
               edgecolors="black", linewidths=0.5, label=f"λ={lam}")

sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])
cbar = plt.colorbar(sm, ax=ax)
cbar.set_label("λ (0=fluency, 1=adequacy)")

ax.set_xlabel("Adequacy (chrF)")
ax.set_ylabel("Fluency (neg-NLL)")
ax.set_title(f"MRT λ Sweep vs Fine-tuned Baseline — {label}\nN={N}, K={K}")
ax.legend(fontsize=8, loc="lower left")
ax.grid(True, alpha=0.3)
plt.tight_layout()

plot_path = EVAL_DIR / f"{args.domain}_mrt_sweep.png"
plt.savefig(str(plot_path), dpi=150)
plt.close()
print(f"\nSaved plot to {plot_path}")

# ── Save greedy scores CSV ────────────────────────────────────────────────────

rows = [
    {"lambda": "baseline_raw", "chrf": raw_greedy["chrf"]    if raw_greedy else None,
                                "fluency": raw_greedy["fluency"] if raw_greedy else None},
    {"lambda": "baseline_ft",  "chrf": ft_greedy["chrf"], "fluency": ft_greedy["fluency"]},
]
for lam, (_, greedy) in sorted(lambda_results.items()):
    rows.append({"lambda": lam, "chrf": greedy["chrf"], "fluency": greedy["fluency"]})

results_df = pd.DataFrame(rows)
results_df.to_csv(EVAL_DIR / f"{args.domain}_greedy_scores.csv", index=False)
print(results_df.to_string(index=False))