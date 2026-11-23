"""Tables, figures and a report for the cut comparison."""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import numpy as np

from rates_and_inexact.common import trace, csv_write
from .protocol import LABELS, METHOD_NAMES
from .runner import read, write_json

COLORS = {"C-LBreI": "tab:gray", "Log-LBreI": "tab:blue", "Polyak-1": "tab:purple",
          "Polyak-2": "tab:brown", "CP": "tab:orange"}
STYLES = {"C-LBreI": "-.", "Log-LBreI": "-", "Polyak-1": ":", "Polyak-2": "--", "CP": "-"}


def median(values):
    values = [v for v in values if v is not None and np.isfinite(v)]
    return float(np.median(values)) if values else None


def fmt(v):
    if v is None:
        return "--"
    return f"{v:.0f}" if abs(v) >= 100 else f"{v:.3g}"


def group_of(r):
    m = r["model"]
    kappa = float(m.get("kappa", 1.))
    if m.get("target_r") is None:
        return (kappa, float(m["beta"]), f"p0=0, a={m['amplitude']:g}")
    return (kappa, float(m["beta"]), f"r0={m['target_r']:g}")


def collect(output):
    runs = []
    for path in sorted((Path(output)/"instances").glob("*/*/result.json")):
        r = read(path)
        r["path"] = path.parent
        runs.append(r)
    return runs


def ratio(a, b):
    return a/b if a is not None and b is not None else None


def tables(runs, output):
    by = defaultdict(dict)
    for r in runs:
        # instance_id, not source_instance: new kappa=10 instances have no archived source.
        by[(group_of(r), r["instance_id"])][r["method"]["name"]] = r
    rows, groups = [], defaultdict(list)
    for (group, instance), methods in sorted(by.items()):
        row = dict(kappa=group[0], beta=group[1], setting=group[2], instance=instance)
        for name in METHOD_NAMES:
            r = methods.get(name)
            row[f"N_{LABELS[name]}"] = r["joint_N"] if r else None
            row[f"N1e-4_{LABELS[name]}"] = r["distance_hits"][1] if r else None
            row[f"r0_{LABELS[name]}"] = r["initial_r"] if r else None
        row["Log/clipped"] = ratio(row["N_Log"], row["N_clipped"])
        row["CP/Log"] = ratio(row["N_CP"], row["N_Log"])
        row["CP/Polyak-1"] = ratio(row["N_CP"], row["N_Polyak-1"])
        row["Log/Polyak-2"] = ratio(row["N_Log"], row["N_Polyak-2"])
        rows.append(row)
        groups[group].append(row)
    summary = []
    for group, values in sorted(groups.items()):
        s = dict(kappa=group[0], beta=group[1], setting=group[2], instances=len(values),
                 initial_r_median=median([v["r0_Log"] for v in values]))
        for name in METHOD_NAMES:
            label = LABELS[name]
            s[f"success_{label}"] = sum(v[f"N_{label}"] is not None for v in values)
            s[f"N_{label}"] = median([v[f"N_{label}"] for v in values])
        for key in ("Log/clipped", "CP/Log", "CP/Polyak-1", "Log/Polyak-2"):
            s[key] = median([v[key] for v in values])
        summary.append(s)
    csv_write(Path(output)/"tables/cuts_pairs.csv", rows)
    csv_write(Path(output)/"tables/cuts_summary.csv", summary)
    tex = [r"\begin{tabular}{lllrrrrrrr}\toprule",
           r"$\kappa$ & $\beta$ & Initialization & clipped & Log & Polyak-1 & Polyak-2 & CP & Log/clipped & CP/Log\\\midrule"]
    for s in summary:
        setting = s["setting"].replace("p0=0, ", "$p_0=0$, ").replace("r0=", "$r_0=").replace("a=", "$a=")+"$"
        tex.append(" & ".join([f"{s['kappa']:g}", f"{s['beta']:g}", setting] +
                              [fmt(s[f"N_{LABELS[n]}"]) for n in METHOD_NAMES] +
                              [f"{s['Log/clipped']:.3f}" if s["Log/clipped"] else "--",
                               f"{s['CP/Log']:.3f}" if s["CP/Log"] else "--"]) + r"\\")
    tex.append(r"\bottomrule\end{tabular}")
    (Path(output)/"tables/cuts_table.tex").write_text("\n".join(tex)+"\n", encoding="utf-8")
    return summary


def curves(selected, column, grid, floor=1e-16):
    values = []
    for r in selected:
        t = trace(r["path"]/"trace.csv.gz")
        y = np.maximum(np.abs(t[column]), floor)
        v = np.exp(np.interp(np.log(grid), np.log(t["n_grad"]), np.log(y)))
        # Runs stop at the joint tolerance; the stopped value is held, as stated in the caption.
        values.append(v)
    return np.asarray(values)


def figures(runs, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 10, "pdf.fonttype": 42})
    out = Path(output)/"figures"
    out.mkdir(parents=True, exist_ok=True)
    budget = max(r["N"] for r in runs)
    grid = np.unique(np.geomspace(1, budget, 200).astype(int))
    summary = {}

    def panel(ax, selected, column, title, ylabel):
        summary[title] = {}
        for name in METHOD_NAMES:
            chosen = [r for r in selected if r["method"]["name"] == name]
            if not chosen:
                continue
            v = curves(chosen, column, grid)
            med = np.median(v, axis=0)
            ax.loglog(grid, med, STYLES[name], color=COLORS[name], label=LABELS[name])
            ax.fill_between(grid, np.quantile(v, .25, axis=0), np.quantile(v, .75, axis=0),
                            color=COLORS[name], alpha=.1)
            summary[title][LABELS[name]] = {str(n): float(med[np.searchsorted(grid, n)]) for n in (10, 100, 1000)
                                            if n <= grid[-1]}
        ax.set(title=title, xlabel="Gradient evaluations", ylabel=ylabel)
        ax.grid(alpha=.2)

    hard = [r for r in runs if r["variant"] == "kappa10_calibrated" and r["model"]["beta"] == 4.]
    ls = [r for r in runs if r["variant"] == "kappa10_zero" and r["model"]["beta"] == 0.]
    fig, axs = plt.subplots(1, 3, figsize=(11, 3.6))
    panel(axs[0], hard, "relative_distance", r"(a) $\kappa=10$, $\beta=4$, $r_0=1000$", "Relative reconstruction error")
    panel(axs[1], ls, "relative_distance", r"(b) $\kappa=10$, $\beta=0$ (least squares)", "Relative reconstruction error")
    # (c) relative depth of the four cuts against r=L1 q/L0, over all stored points with beta>0
    pts = defaultdict(list)
    for r in runs:
        # One family only: pooling kappa=1 and kappa=10 mixes different curvature spectra per bin.
        if r["variant"] != "kappa10_calibrated":
            continue
        t = trace(r["path"]/"trace.csv.gz")
        ok = (t["tau"] > 0) & (t["r"] > 0)
        for column in ("rho_C", "rho_log", "rho_delta", "rho_CP"):
            pts[column].append(np.c_[t["r"][ok], t[column][ok]])
    edges = np.geomspace(1e-3, max(np.max(np.vstack(v)[:, 0]) for v in pts.values()), 40)
    summary["(c) cut depth"] = {}
    for column, name in (("rho_C", "C-LBreI"), ("rho_log", "Log-LBreI"), ("rho_delta", "Polyak-1"), ("rho_CP", "CP")):
        data = np.vstack(pts[column])
        idx = np.digitize(data[:, 0], edges)
        centers, meds = [], []
        for i in range(1, len(edges)):
            sel = data[idx == i, 1]
            if sel.size >= 20:
                centers.append(np.sqrt(edges[i-1]*edges[i]))
                meds.append(np.median(sel))
        label = {"Polyak-1": r"Polyak ($\delta$)"}.get(name, LABELS[name])
        axs[2].semilogx(centers, meds, STYLES[name], color=COLORS[name], label=label)
        summary["(c) cut depth"][label] = dict(zip([f"{c:.3g}" for c in centers], [float(m) for m in meds]))
    axs[2].set(title=r"(c) Cut / exact depth ($\kappa=10$, $r_0=1000$)", xlabel=r"$r=L_1q/L_0$",
               ylabel=r"$\sigma/\langle g,x-z^\dagger\rangle$",
               ylim=(0, 1.02))
    axs[2].grid(alpha=.2)
    handles, labels = axs[0].get_legend_handles_labels()
    h2, l2 = axs[2].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=5, fontsize=9, bbox_to_anchor=(.36, 0))
    fig.legend(h2, l2, loc="lower center", ncol=2, fontsize=8, bbox_to_anchor=(.84, 0))
    fig.tight_layout(rect=(0, .12, 1, 1))
    for ext in ("pdf", "png"):
        fig.savefig(out/f"cuts_explicit.{ext}", bbox_inches="tight", dpi=180)
    plt.close(fig)
    # Dai-style three metrics for the hardest group
    fig, axs = plt.subplots(1, 3, figsize=(11, 3.4))
    panel(axs[0], hard, "relative_distance", "Reconstruction error", r"$\|x-z^\dagger\|/\|z^\dagger\|$")
    panel(axs[1], hard, "Omega_gap", "Outer objective gap", r"$|\Omega(z^\dagger)-\Omega(x)|$")
    panel(axs[2], hard, "affine_residual", "Feasibility", r"$\|Bx-b\|/(1+\|b\|)$")
    handles, labels = axs[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=5, fontsize=9)
    fig.tight_layout(rect=(0, .1, 1, 1))
    for ext in ("pdf", "png"):
        fig.savefig(out/f"cuts_dai_metrics.{ext}", bbox_inches="tight", dpi=180)
    plt.close(fig)
    return summary


def analyze(output):
    output = Path(output)
    runs = collect(output)
    summary = tables(runs, output)
    figs = figures(runs, output)
    frozen = read(output/"config.json")
    verification = read(output/"verification.json") if (output/"verification.json").exists() else None
    lines = ["# Log 割与 CP 割：显式更新中的比较", "",
             "**试运行，不是正式结果。**" if frozen["pilot"] else "正式运行。", "",
             "实例：cosh 线性反问题（B 为 128x512，x† 稀疏，弹性网核，解集仿射）。kappa=1 的 100 个取自基线结果；"
             "kappa=10 的 50 个为新生成并带 60 位证书。beta=0 为最小二乘，即 Dai 等人 5.1 的设置；beta=1、4 时 L1>0。"
             "所有方法都是算法 1 的显式更新，只改变割。运行在达到联合精度时停止，预算 50000 次梯度。", "",
             f"完整性核验：{verification['integrity_passed'] if verification else '未执行'}；"
             f"运行数 {verification['runs'] if verification else len(runs)}；状态 {verification['statuses'] if verification else ''}", "",
             "## 达到联合精度所需梯度次数（中位数）与成对比值", "",
             "| kappa | beta | 初值 | 初始 r 中位数 | 截断 | Log | Polyak-1 | Polyak-2 | CP | Log/截断 | CP/Log | CP/Polyak-1 | Log/Polyak-2 | 成功数(截断/Log/P1/P2/CP) |",
             "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|"]
    for s in summary:
        succ = "/".join(str(s[f"success_{LABELS[n]}"]) for n in METHOD_NAMES)
        lines.append(f"| {s['kappa']:g} | {s['beta']:g} | {s['setting']} | {fmt(s['initial_r_median'])} | " +
                     " | ".join(fmt(s[f"N_{LABELS[n]}"]) for n in METHOD_NAMES) + " | " +
                     " | ".join(f"{s[k]:.3f}" if s[k] else "--" for k in ("Log/clipped", "CP/Log", "CP/Polyak-1", "Log/Polyak-2")) +
                     f" | {succ} |")
    lines += ["", "比值为逐实例比值的中位数，只在两种方法都成功的实例上计算。", "",
              "## 图", "",
              "- figures/cuts_explicit.pdf：(a) kappa=10、beta=4、r0=1000 的重构误差；(b) kappa=10、beta=0 最小二乘；"
              "(c) kappa=10、r0=1000（beta=1、4）轨迹的全部存储点上，四种割与精确深度之比随 r 的变化。",
              "- figures/cuts_dai_metrics.pdf：kappa=10、beta=4、r0=1000 下 Dai 等人的三项指标（重构误差、外层目标差、可行性）。",
              "- kappa=1 时 B 的行正交，beta=0 的最小二乘中截断割、Log 割、CP 割和 Polyak-2 都等于精确深度 2*delta，"
              "轨迹相同，这组只作一致性检查。", ""]
    (output/"REPORT_zh.md").write_text("\n".join(lines), encoding="utf-8")
    write_json(output/"analysis.json", dict(summary=summary, figures=figs))
    print("\n".join(lines[8:8+len(summary)+2]), flush=True)
    return summary
