"""Mirror the experiment records of the in-pass systems into a local MLflow store (optional viewer).

    uv run --with mlflow==3.18.0 python scripts/log_mlflow.py
    uv run --with mlflow==3.18.0 mlflow ui --backend-store-uri sqlite:///mlflow.db

The record of the experiments is results/inpass/experiments.csv and experiments.json, which are in git.
This script copies them into MLflow so that the runs can be filtered and compared in its interface:
one run per system, set and seed (parameters: what was changed and with which constants; metrics: the
detection rate and what stands next to it in the table; tags: role and hypothesis), and one run per
system and set over all seeds, which also carries the tables, the figures and the per-clip predictions.
The store (mlflow.db, mlruns/) is not tracked: MLflow writes absolute paths into it. Running the
script again replaces the runs. The list of the runs as MLflow holds them (name, tags, parameters, metrics)
is exported to results/inpass/mlflow_runs.csv, which is tracked.
"""

from __future__ import annotations

import argparse
import json

import mlflow

from emoswitch.config import ROOT, load_config
from emoswitch.evaluate import measure_fingerprint
from emoswitch.inpass import CONTINUATION_SEED_OFFSET

ARTIFACTS = ("experiments.md", "experiments.csv", "predictions.csv", "fig_systems.png", "fig_inpass_steps.png",
             "fig_inpass_profiles.png", "fig_pairs.png", "fig_outcomes.png")  # fmt: skip


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/inpass.yaml")
    ap.add_argument("--store", default="sqlite:///mlflow.db")
    args = ap.parse_args()

    cfg = load_config(args.config)
    out = ROOT / "results" / cfg["name"]
    records = json.loads((out / "experiments.json").read_text(encoding="utf-8"))["experiments"]
    mlflow.set_tracking_uri(args.store)
    experiment = mlflow.set_experiment(cfg["name"])
    client = mlflow.MlflowClient()
    for run in client.search_runs([experiment.experiment_id]):  # a second call replaces the runs
        client.delete_run(run.info.run_id)
    ip, dc = cfg["inpass"], cfg["detector"]
    run_names = []
    for e in records:
        spec = cfg["systems"][e["system"]]
        seed = "all" if e["seed"] is None else e["seed"]
        run_names.append(f"{e['system']}/{e['set']}/{'all seeds' if e['seed'] is None else f's{seed}'}")
        with mlflow.start_run(run_name=run_names[-1]):
            mlflow.set_tags({"role": e["role"], "set_role": e["set_role"], "hypothesis": e["hypothesis"]})
            params = {
                "system": e["system"],
                "change": e["change"],
                "set": e["set"],
                "seed": seed,
                "anchor": spec.get("anchor", False),
                "steer": spec.get("steer", False),
                "continuation_seed_offset": CONTINUATION_SEED_OFFSET if spec.get("role") == "inpass" else None,
                "detector_min_abs_delta": dc["min_abs_delta"],
                "detector_max_changepoint_err_s": dc["max_changepoint_err_s"],
                "measure_fingerprint": measure_fingerprint(cfg),
            }
            if spec.get("anchor"):
                params.update(anchor_share=ip["anchor"]["share"], anchor_min_tail=ip["anchor"]["min_tail_frames"])
            if spec.get("steer"):
                params.update(steer_layer=ip["steer"]["layer"], steer_alpha=ip["steer"]["alpha"])
            mlflow.log_params(params)
            metrics = {
                "det": e["det"],
                "detected": e["detected"],
                "n": e["n"],
                "invalid_clips": e["invalid"],
                "wer_mean": e["wer_mean"],
                **{f"outcome_{k}": v for k, v in e["outcomes"].items()},
            }
            if e["commanded_step_mean"] is not None:
                metrics["commanded_step_mean"] = e["commanded_step_mean"]
            if e["seed"] is None:
                metrics.update(det_ci_low=e["ci_phrases"][0], det_ci_high=e["ci_phrases"][1])
                mlflow.log_artifact(str(ROOT / args.config))
                for name in ARTIFACTS:
                    mlflow.log_artifact(str(out / name))
            mlflow.log_metrics(metrics)
    runs = mlflow.search_runs([experiment.experiment_id])
    keep = ["tags.mlflow.runName", *sorted(c for c in runs if c.startswith(("params.", "metrics.")))]
    keep += ["tags.role", "tags.set_role", "tags.hypothesis"]
    order = {name: i for i, name in enumerate(run_names)}
    runs = runs[keep].sort_values("tags.mlflow.runName", key=lambda col: col.map(order))
    runs.to_csv(out / "mlflow_runs.csv", index=False, lineterminator="\n", float_format="%.6g")
    print(f"[mlflow] {len(records)} runs in experiment {cfg['name']!r} of {args.store}; list in mlflow_runs.csv")


if __name__ == "__main__":
    main()
