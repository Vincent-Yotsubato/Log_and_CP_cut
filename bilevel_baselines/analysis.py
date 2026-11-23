"""Tables, a figure and a report for the simple bilevel baselines."""
from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

import numpy as np

from rates_and_inexact.common import trace, csv_write
from .protocol import EXPERIMENTS, H_SOURCE, GAMMAS
from .runner import read, write_json

H_METHODS = ("H-CP", "H-Log")


def median(values):
    values = [v for v in values if v is not None and np.isfinite(v)]
    return float(np.median(values)) if values else None


def fmt(v, digits=3):
    if v is None:
        return "--"
    if isinstance(v, (int, np.integer)) or (abs(v) >= 100 and float(v).is_integer()):
        return f"{v:.0f}"
    return f"{v:.{digits}g}" if abs(v) < 1e4 else f"{v:.0f}"


def collect(output):
    runs = []
    for path in sorted((Path(output)/"instances").glob("*/*/result.json")):
        r = read(path)
        r["path"] = path.parent
        runs.append(r)
    return runs


def h_runs(instances):
    """H-CP and H-Log first hits from the source of Table 3, restricted to the same instances."""
    rows = []
    with open(H_SOURCE, encoding="utf-8-sig") as stream:
        for r in csv.DictReader(stream):
            if r["group"] != "Euclidean" or r["method"] not in H_METHODS or r["instance_id"] not in instances:
                continue
            n = int(float(r["N"])) if r["N"] else None
            joint = int(float(r["joint_first_N"])) if r["joint_first_N"] else None
            rows.append(dict(label=r["method"], source_instance=r["instance_id"], power=int(r["s"]),
                             first_distance_N=n, first_joint_N=joint,
                             distance_success=r["distance_success"] == "True",
                             joint_success=r["joint_success"] == "True",
                             relative_distance=float(r["final_distance"]), f=None, Omega_gap=None,
                             trace=EXPERIMENTS/r["source"].replace("\\", "/").replace("result.json", "trace.csv.gz")))
    return rows


def summarize(rows, key="label"):
    groups = defaultdict(list)
    for r in rows:
        groups[r[key]].append(r)
    out = []
    for name, values in groups.items():
        s = dict(label=name, instances=len(values),
                 distance_success=sum(v["distance_success"] for v in values),
                 joint_success=sum(v["joint_success"] for v in values),
                 median_N_distance=median([v["first_distance_N"] for v in values]),
                 median_N_joint=median([v["first_joint_N"] for v in values]),
                 median_final_relative_distance=median([v["relative_distance"] for v in values]),
                 median_final_f=median([v.get("f") for v in values]),
                 median_final_Omega_gap=median([v.get("Omega_gap") for v in values]))
        for power in (3, 4):
            sub = [v for v in values if v["power"] == power]
            s[f"distance_success_s{power}"] = sum(v["distance_success"] for v in sub)
            s[f"median_N_distance_s{power}"] = median([v["first_distance_N"] for v in sub])
            s[f"median_final_relative_distance_s{power}"] = median([v["relative_distance"] for v in sub])
        out.append(s)
    return out


def best_of(summary, prefix):
    """Most 1% successes, then the smallest median first hit, then the smallest median final distance."""
    candidates = [s for s in summary if s["label"].startswith(prefix)]
    return min(candidates, key=lambda s: (-s["distance_success"],
                                          s["median_N_distance"] if s["median_N_distance"] is not None else np.inf,
                                          s["median_final_relative_distance"]))


def curves(rows, grid):
    """Median relative distance over instances on a common evaluation grid (display only)."""
    values = []
    for r in rows:
        t = trace(r["trace"])
        n = t["N"] if "N" in t else t["n_grad"]
        y = np.maximum(t["relative_distance"], 1e-18)
        # Stopped runs are fixed points; extend their last value.
        values.append(np.interp(np.log(grid), np.log(np.maximum(n, 1)), np.log(y)))
    return np.exp(np.median(np.array(values), axis=0))


def figure(output, selected, by_label):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    grid = np.geomspace(1, 1e5, 300)
    styles = {"H-CP": ("tab:orange", "-"), "H-Log": ("tab:blue", "-"), "AGM-BiO": ("tab:red", "--"),
              "MNG-local": ("tab:green", "-."), "BiG-SAM (ball)": ("tab:purple", ":"),
              "BiG-SAM (clip)": ("tab:brown", ":")}
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4), constrained_layout=True, sharey=True)
    for ax, power in zip(axes, (3, 4)):
        for display, source_label in selected:
            rows = [r for r in by_label[source_label] if r["power"] == power]
            color, style = styles[display]
            ax.loglog(grid, curves(rows, grid), style, color=color, label=display)
        ax.axhline(1e-2, color="0.6", linewidth=.6)
        ax.set(title=f"s = {power}", xlabel="gradient evaluations", ylim=(1e-9, 2))
        ax.grid(alpha=.2)
    axes[0].set_ylabel("median relative distance to $z^\\dagger$")
    axes[1].legend(fontsize=7, loc="lower left")
    dest = Path(output)/"figures"
    dest.mkdir(parents=True, exist_ok=True)
    fig.savefig(dest/"bilevel_baselines.pdf")
    fig.savefig(dest/"bilevel_baselines.png", dpi=160)
    plt.close(fig)


def analyze(output):
    output = Path(output)
    runs = collect(output)
    instances = {r["source_instance"] for r in runs}
    for r in runs:
        r["trace"] = r["path"]/"trace.csv.gz"
    h = h_runs(instances)
    rows = runs+h
    per_run = [{k: (str(v) if isinstance(v, Path) else v) for k, v in r.items()
                if k not in ("method", "parameters", "model", "path")} for r in rows]
    csv_write(output/"tables/baseline_runs.csv", per_run)
    summary = summarize(rows)
    order = {"H-CP": 0, "H-Log": 1, "MNG-local": 2}
    summary.sort(key=lambda s: (order.get(s["label"], 3), s["label"]))
    csv_write(output/"tables/baseline_summary.csv", summary)
    by_label = defaultdict(list)
    for r in rows:
        by_label[r["label"]].append(r)
    ball, clip = best_of(summary, "BiG-SAM(ball,"), best_of(summary, "BiG-SAM(clip,")
    agm_best = best_of(summary, "AGM-BiO")
    selected = [("H-CP", "H-CP"), ("AGM-BiO", agm_best["label"]), ("H-Log", "H-Log"),
                ("MNG-local", "MNG-local"), ("BiG-SAM (ball)", ball["label"]), ("BiG-SAM (clip)", clip["label"])]
    lookup = {s["label"]: s for s in summary}
    paper = []
    for display, source_label in selected:
        s = lookup[source_label]
        paper.append(dict(method=display, configuration=source_label, uses_f_star=display in ("H-CP", "AGM-BiO"),
                          instances=s["instances"],
                          distance_success=s["distance_success"], median_N_distance=s["median_N_distance"],
                          joint_success=s["joint_success"], median_N_joint=s["median_N_joint"],
                          median_final_relative_distance=s["median_final_relative_distance"],
                          distance_success_s3=s["distance_success_s3"], distance_success_s4=s["distance_success_s4"]))
    csv_write(output/"tables/paper_table.csv", paper)
    try:
        figure(output, selected, by_label)
        figure_status = "figures/bilevel_baselines.pdf"
    except Exception as error:  # the report is still useful without the display
        figure_status = f"figure failed: {error}"
    # Paired view: instances where both a baseline and its H counterpart reach 1%.
    paired = {}
    for display, source_label in selected:
        if display in H_METHODS:
            continue
        partner = "H-CP" if display == "AGM-BiO" else "H-Log"
        a = {r["source_instance"]: r for r in by_label[source_label]}
        b = {r["source_instance"]: r for r in by_label[partner]}
        ratios = [a[i]["first_distance_N"]/b[i]["first_distance_N"] for i in a
                  if i in b and a[i]["first_distance_N"] and b[i]["first_distance_N"]]
        paired[display] = dict(partner=partner, both_successful=len(ratios), median_ratio=median(ratios))
    diagnostics = dict(
        mng_max_cut_ratio=max(r["diagnostic"] for r in by_label["MNG-local"]),
        mng_inadmissible=sum(r["diagnostic_count"] for r in by_label["MNG-local"]),
        bigsam_ball_exits=sum(r["diagnostic_count"] for lbl, v in by_label.items() if lbl.startswith("BiG-SAM(ball")
                              for r in v),
        agm_max_violation=max(r["diagnostic"] for lbl, v in by_label.items() if lbl.startswith("AGM-BiO") for r in v),
        L_ball_over_L0_median=median([r["parameters"]["L_ball"]/r["L0"] for r in runs]),
        agm_thm44_gamma_median=median([r["parameters"]["gamma"] for r in by_label.get(
            "AGM-BiO(thm44, radius=1rho)", [])]),
    )
    result = dict(summary=summary, paper_table=paper, paired=paired, diagnostics=diagnostics,
                  selected=selected, gammas=list(GAMMAS), figure=figure_status)
    write_json(output/"analysis.json", result)
    report(output, summary, paper, paired, diagnostics, selected)
    return result


def report(output, summary, paper, paired, diagnostics, selected):
    lines = ["# simple bilevel 基线：40 个欧氏多面体实例", "",
             "由 `bilevel_baselines/analysis.py` 生成。H-CP、H-Log 的数字取自 `results/selection_costs`（论文表 3 的来源）。", "",
             "## 论文用的六行", "",
             "| 方法 | 配置 | 用 f* | 1% 成功 | N 中位数（成功者） | 联合成功 | 最终相对距离中位数 |",
             "|---|---|---|---|---|---|---|"]
    for p in paper:
        lines.append(f"| {p['method']} | {p['configuration']} | {'是' if p['uses_f_star'] else '否'} | "
                     f"{p['distance_success']}/{p['instances']}（s=3: {p['distance_success_s3']}，s=4: {p['distance_success_s4']}） | "
                     f"{fmt(p['median_N_distance'])} | {p['joint_success']}/{p['instances']} | "
                     f"{fmt(p['median_final_relative_distance'])} |")
    lines += ["", "## 全部配置", "",
              "| 配置 | 1% 成功 | N 中位数 | 联合成功 | 最终相对距离中位数 | 最终 f 中位数 | 最终 Ω 间隙中位数 |",
              "|---|---|---|---|---|---|---|"]
    for s in summary:
        lines.append(f"| {s['label']} | {s['distance_success']}/{s['instances']} | {fmt(s['median_N_distance'])} | "
                     f"{s['joint_success']} | {fmt(s['median_final_relative_distance'])} | {fmt(s['median_final_f'])} | "
                     f"{fmt(s['median_final_Omega_gap'])} |")
    lines += ["", "## 配对比较（两者都达到 1% 的实例）", ""]
    for name, v in paired.items():
        lines.append(f"- {name} 对 {v['partner']}：{v['both_successful']} 个实例，首达梯度数之比的中位数 {fmt(v['median_ratio'])}")
    lines += ["", "## 诊断", ""]
    for k, v in diagnostics.items():
        lines.append(f"- {k}: {fmt(v) if isinstance(v, float) else v}")
    Path(output, "REPORT_zh.md").write_text("\n".join(lines)+"\n", encoding="utf-8")
