"""Regenerate all tables and publication figures from immutable run records."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import gzip
import hashlib
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from cut_functions import ElasticNet, gamma, rho
from .numerics import COLUMNS
from .runner import write_json

COL = {name: i for i, name in enumerate(COLUMNS)}
ORDER = ["Polyak-1", "Polyak-2", "CP", "C-LBreI(theta=1)", "Log-LBreI(theta=1)", "H-Log", "H-CP"]
COLORS = {name: plt.get_cmap("tab10")(i) for i, name in enumerate(ORDER)}


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def table(path, rows):
    if not rows:
        return
    columns = list(dict.fromkeys(k for row in rows for k in row))
    with Path(path).open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def stats(values, prefix=""):
    values = np.asarray([x for x in values if x is not None and np.isfinite(x)], float)
    if not len(values):
        return {prefix + key: None for key in ("n", "median", "q25", "q75", "min", "max")}
    return {prefix + "n": len(values), prefix + "median": float(np.median(values)),
            prefix + "q25": float(np.quantile(values, .25)), prefix + "q75": float(np.quantile(values, .75)),
            prefix + "min": float(np.min(values)), prefix + "max": float(np.max(values))}


def load(output):
    runs = []
    for result_file in sorted(output.glob("instances/*/*/result.json")):
        r = read(result_file)
        r["path"] = result_file.parent
        r["label"] = r["method"]["label"]
        with gzip.open(result_file.parent / "trace.csv.gz", "rt", encoding="utf-8") as stream:
            r["trace"] = np.loadtxt(stream, delimiter=",", skiprows=1, ndmin=2)
        runs.append(r)
    return runs


def save_figure(output, name, fig):
    fig.savefig(output / "figures" / (name + ".png"), dpi=180, bbox_inches="tight")
    fig.savefig(output / "figures" / (name + ".pdf"), bbox_inches="tight")
    plt.close(fig)


def selection(output, runs):
    selected = [r for r in runs if r["panel"] == 1]
    snapshots, terminal = [], []
    for r in selected:
        md, t = r["metadata"], r["trace"]
        for N in (1000, 10000, 100000):
            indices = np.flatnonzero(t[:, COL["n_grad"]] <= N)
            if not len(indices):
                continue
            row = t[indices[-1]]
            carried = r["status"] == "exact_zero" and r["N"] < N
            available = row[COL["n_grad"]] == N or carried
            snapshots.append({"variant": r["variant"], "m": md["m"], "n": md["n"],
                "s": md["s"], "rho_c": md["rho_c"], "seed": md["seed"], "method": r["label"],
                "N": N, "relative_distance": float(row[COL["relative_distance"]]) if available else None,
                "available": available, "carried_exact_zero": carried,
                "actual_last_N": int(row[COL["n_grad"]]), "status": r["status"]})
        terminal.append({"variant": r["variant"], "m": md["m"], "n": md["n"], "s": md["s"],
             "rho_c": md["rho_c"], "seed": md["seed"], "method": r["label"],
             "relative_distance": r["relative_distance"], "distance_above_1pct": r["relative_distance"] > .01,
             "delta": r["delta"], "feasibility": r["feasibility"],
             "complementarity": r["complementarity"], "status": r["status"]})
    table(output / "tables" / "e1_snapshots.csv", snapshots)
    table(output / "tables" / "e1_terminal.csv", terminal)
    grouped = defaultdict(list)
    for row in snapshots:
        grouped[tuple(row[k] for k in ("variant", "m", "n", "s", "rho_c", "method", "N"))].append(row)
    aggregates = []
    for key, group in grouped.items():
        aggregates.append(dict(zip(("variant", "m", "n", "s", "rho_c", "method", "N"), key)) |
                          stats([r["relative_distance"] for r in group]) |
                          {"expected": len(group), "unavailable": sum(not r["available"] for r in group)})
    table(output / "tables" / "e1_distance_median_range.csv", aggregates)
    groups = defaultdict(list)
    for row in terminal:
        groups[tuple(row[k] for k in ("variant", "m", "n", "s", "rho_c", "method"))].append(row)
    table(output / "tables" / "e1_wrong_selection_fraction.csv", [
        dict(zip(("variant", "m", "n", "s", "rho_c", "method"), key)) |
        {"seeds": len(group), "fraction_distance_above_1pct": float(np.mean([r["distance_above_1pct"] for r in group])),
         "median_feasibility": float(np.median([r["feasibility"] for r in group])),
         "median_complementarity": float(np.median([r["complementarity"] for r in group]))}
        for key, group in groups.items()])
    for m, n in sorted({(r["metadata"]["m"], r["metadata"]["n"]) for r in selected if r["variant"] == "polyhedron"}):
        fig, axes = plt.subplots(2, 4, figsize=(14, 6.5), sharex=True)
        for column, (power, correlation) in enumerate([(3, 0.), (3, .8), (4, 0.), (4, .8)]):
            cell = [r for r in selected if r["variant"] == "polyhedron" and r["metadata"]["m"] == m
                    and r["metadata"]["s"] == power and r["metadata"]["rho_c"] == correlation]
            common_N = sorted({int(record[1]) for r in cell for record in r["trace"]})
            for name in ORDER:
                trajectories = [r for r in cell if r["label"] == name]
                for row, metric in enumerate(("delta", "relative_distance")):
                    by_N = {N: [] for N in common_N}
                    for r in trajectories:
                        for record in r["trace"]:
                            by_N[int(record[COL["n_grad"]])].append(record[COL[metric]])
                    # Explicit exact-zero endpoints are fixed points; extend them
                    # only as such, never carry numerical failures to the budget.
                    for N in list(by_N):
                        for r in trajectories:
                            if r["status"] == "exact_zero" and r["N"] < N:
                                by_N[N].append(r["trace"][-1, COL[metric]])
                    pairs = sorted((N, np.median(v)) for N, v in by_N.items() if len(v) == len(trajectories))
                    if pairs:
                        xs, ys = np.array(pairs).T
                        floor = 1e-18 if metric == "delta" else 1e-16
                        axes[row, column].loglog(xs, np.maximum(ys, floor), label=name,
                                                color=COLORS[name], lw=1.5)
            axes[0, column].set_title(f"s={power}, rho={correlation:g}")
            axes[1, column].set_xlabel("Gradient evaluations N")
        axes[0, 0].set_ylabel("Median objective gap (display floor 1e-18)")
        axes[1, 0].set_ylabel("Median relative distance")
        handles, labels = axes[0, 0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="lower center", ncol=4, fontsize=8)
        seed_count = len({r["metadata"]["seed"] for r in selected if r["variant"] == "polyhedron" and r["metadata"]["m"] == m})
        fig.suptitle(f"Polyhedral selection: {m} x {n}, {seed_count} seeds")
        fig.tight_layout(rect=(0, .1, 1, .96))
        save_figure(output, f"e1_selection_{m}x{n}", fig)
    example = [r for r in selected if r["variant"] == "example32"]
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.7))
    for r in example:
        t = r["trace"]
        for ax, metric in zip(axes, ("delta", "relative_distance", "x1_minus_x2")):
            floor = 1e-18 if metric == "delta" else 1e-16
            ax.loglog(t[:, 1], np.maximum(np.abs(t[:, COL[metric]]), floor), label=r["label"], color=COLORS[r["label"]])
            ax.set_xlabel("Gradient evaluations N")
            ax.set_ylabel(metric)
    axes[0].legend(fontsize=7)
    fig.tight_layout()
    save_figure(output, "e1_example32", fig)
    return aggregates, terminal


def comparison(output, runs):
    selected = [r for r in runs if r["panel"] == 2]
    by_instance = defaultdict(dict)
    for r in selected:
        by_instance[r["instance_id"]][r["label"]] = r
    pairs, theoretical = [], []
    for instance, methods in by_instance.items():
        for baseline in ("Polyak-1", "Polyak-2"):
            for cp_label in ("CP", "CP(c=2)", "CP(c=10)"):
                cp, base = methods[cp_label], methods[baseline]
                md = cp["metadata"]
                for j in (2, 4, 6):
                    numerator, denominator = cp["hits"][j - 1], base["hits"][j - 1]
                    pairs.append({"instance_id": instance, "variant": cp["variant"],
                        "beta": md["beta"], "r0": md["target_r"], "a": md["a"], "seed": md["seed"],
                        "method": cp_label, "baseline": baseline, "j": j,
                        "N_cp": numerator, "N_baseline": denominator,
                        "ratio": numerator / denominator if numerator and denominator else None})
        r = methods["CP"]
        ref = read(r["path"].parent / "reference.json")
        instance_data = np.load(r["path"].parent / "instance.npz")
        kernel = ElasticNet(ref["lambda_en"])
        R2 = 2 * kernel.bregman(instance_data["z"], instance_data["p0"])
        for j in (2, 4, 6):
            epsilon = r["delta0"] * 10.**(-j)
            theoretical.append({"instance_id": instance, "j": j, "epsilon": epsilon, "R_omega_squared": R2,
                "Polyak1_strict_threshold": R2 / gamma(epsilon, ref["L0"], ref["L1"], 1),
                "Polyak2_strict_threshold": R2 / gamma(epsilon, ref["L0"], ref["L1"], 2),
                "CP_strict_threshold": R2 / rho(epsilon, ref["L0"], ref["L1"])**2})
    table(output / "tables" / "e2_paired_ratios.csv", pairs)
    table(output / "tables" / "e2_theory_thresholds.csv", theoretical)
    groups = defaultdict(list)
    keys = ("variant", "beta", "r0", "a", "method", "baseline", "j")
    for row in pairs:
        groups[tuple(row[k] for k in keys)].append(row)
    aggregate = [dict(zip(keys, key)) | stats([r["ratio"] for r in group]) |
                 {"expected_pairs": len(group), "unreached_pairs": sum(r["ratio"] is None for r in group)}
                 for key, group in groups.items()]
    table(output / "tables" / "e2_ratio_median_iqr.csv", aggregate)
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), sharex=True)
    for i, baseline in enumerate(("Polyak-1", "Polyak-2")):
        for col, beta in enumerate((1, 4)):
            ax = axes[i, col]
            for j in (2, 4, 6):
                cells = sorted([r for r in aggregate if r["method"] == "CP" and r["baseline"] == baseline
                                and r["beta"] == beta and r["variant"] == "calibrated" and r["j"] == j], key=lambda r: r["r0"])
                ax.plot([r["r0"] for r in cells], [r["median"] for r in cells], "o-", label=f"j={j}")
                ax.fill_between([r["r0"] for r in cells], [r["q25"] for r in cells], [r["q75"] for r in cells], alpha=.12)
            ax.set_xscale("log")
            ax.axhline(1, color="grey", ls="--", lw=1)
            ax.set_title(f"beta={beta}; CP / {baseline}")
            ax.set_ylabel("Paired gradient-count ratio")
            ax.legend()
    for ax in axes[-1]:
        ax.set_xlabel("Calibrated initial r0")
    fig.tight_layout()
    save_figure(output, "e2_cp_ratios", fig)
    fig, axes = plt.subplots(2, 4, figsize=(14, 6), sharey=True)
    for i, beta in enumerate((1, 4)):
        for col, target in enumerate((1, 10, 100, 1000)):
            for r in selected:
                if r["metadata"]["seed"] == 0 and r["metadata"]["beta"] == beta and r["metadata"]["target_r"] == target:
                    t = r["trace"]
                    axes[i, col].loglog(t[:, 1], np.maximum(t[:, COL["best_delta"]] / r["delta0"], 1e-16), label=r["label"])
            axes[i, col].set_title(f"beta={beta}, r0={target}")
            axes[i, col].set_xlabel("Gradient evaluations N")
        axes[i, 0].set_ylabel("Best gap / initial gap")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, fontsize=8)
    fig.tight_layout(rect=(0, .1, 1, 1))
    save_figure(output, "e2_representative_trajectories", fig)
    return aggregate


def generate(output):
    output = Path(output).resolve()
    (output / "tables").mkdir(exist_ok=True)
    (output / "figures").mkdir(exist_ok=True)
    plt.rcParams.update({"font.size": 9, "axes.grid": True, "grid.alpha": .2,
                         "pdf.fonttype": 42, "ps.fonttype": 42})
    runs = load(output)
    table(output / "tables" / "all_runs.csv", [{k: v for k, v in r.items() if k not in ("trace", "path", "method", "metadata", "hits")} |
          r["metadata"] | {f"N_{j+1}": value for j, value in enumerate(r["hits"])} for r in runs])
    e1, terminal = selection(output, runs)
    e2 = comparison(output, runs)
    statuses = Counter(r["status"] for r in runs)
    references = [read(p) for p in output.glob("instances/*/reference.json")]
    failed = [r for r in runs if r["flag"] >= 5]
    correlated_explicit = [r for r in terminal if r["variant"] == "polyhedron" and r["rho_c"] == .8 and not r["method"].startswith("H-")]
    correlated_hcp = [r for r in terminal if r["variant"] == "polyhedron" and r["rho_c"] == .8 and r["method"] == "H-CP"]
    correlated_hlog = [r for r in terminal if r["variant"] == "polyhedron" and r["rho_c"] == .8 and r["method"] == "H-Log"]
    coarse_ratios = [r["median"] for beta in (1, 4) for r in e2 if r["variant"] == "calibrated" and
                     r["method"] == "CP" and r["baseline"] == "Polyak-2" and r["beta"] == beta and r["r0"] == 1000 and r["j"] == 2]
    small_ratio = next((r["median"] for r in e2 if r["method"] == "CP" and r["baseline"] == "Polyak-2" and
                        r["beta"] == 4 and r["r0"] == 1 and r["j"] == 2), None)
    table(output / "tables" / "numerical_failures.csv", [{"instance_id": r["instance_id"], "method": r["label"], "status": r["status"], "K": r["K"]} for r in failed])
    lines = ["# 基线数值实验", "", f"配置：`{read(output / 'config.json')['profile']}`；已保存 **{len(runs)} 次运行**、{len(references)} 个实例。",
             "本报告完全由当前目录的原始轨迹和终点记录生成。", "",
             "## 完整性与数值检查", "", f"状态计数：`{dict(statuses)}`。",
             f"参考解认证间隙最大值：{max(r['certificate']['gap'] for r in references):.3e}；",
             f"逐步割比值最大值：{max(r['max_cut_ratio'] for r in runs):.12g}。数值算法失败 {len(failed)} 次。", "",
             "`budget` 表示已用完规定预算，不等于数值失败，也不等于已达到选解精度。函数值与选解成功分别记录；表中不会把未达标运行当成成功。", "",
             "## 实验 1：非仿射选解", "", "二维例子末点：", "",
             "| 方法 | 梯度次数 | 函数差 | 相对距离 |", "|---|---:|---:|---:|"]
    for r in sorted([r for r in runs if r["variant"] == "example32"], key=lambda r: ORDER.index(r["label"])):
        lines.append(f"| {r['label']} | {r['N']} | {r['delta']:.3e} | {r['relative_distance']:.3e} |")
    lines += ["", "`e1_distance_median_range.csv` 给出 N=10³、10⁴、10⁵ 时距离的中位数与 [min,max]。",
              "`e1_wrong_selection_fraction.csv` 给出末点距离大于 1% 的种子比例，并同时给出可行性与互补性；有限预算内的距离不能直接称为渐近极限。",
              f"在 rho_c=0.8 的 {len(correlated_hcp)} 个实例中，五种显式方法共有 {sum(r['distance_above_1pct'] for r in correlated_explicit)}/{len(correlated_explicit)} 条轨迹的末点相对距离大于 1%；H-CP 为 {sum(r['distance_above_1pct'] for r in correlated_hcp)}/{len(correlated_hcp)}，H-Log 为 {sum(r['distance_above_1pct'] for r in correlated_hlog)}/{len(correlated_hlog)}。H-Log 在部分四次模型上较慢，本预算内并非所有 H 运行都完成了高精度选解。",
              "rho_c=0 的高斯行不是严格正交；另存 `orthogonal_control` 作为精确机制对照。图中函数差小于 1e-18 的值统一显示于 1e-18；原始数据不截断。仅对 exact_zero 固定点延伸其平台。", "",
              "## 实验 2：CP 与 Polyak", "", "CP(c=1) 相对 Polyak-2 的成对梯度次数比值（中位数 [IQR]）：", "",
              "| beta | r0 | j | 中位数 [Q25,Q75] | 达标配对数 |", "|---:|---:|---:|---:|---:|"]
    for r in sorted([r for r in e2 if r["variant"] == "calibrated" and r["method"] == "CP" and r["baseline"] == "Polyak-2"], key=lambda r: (r["beta"], r["r0"], r["j"])):
        value = f"{r['median']:.3f} [{r['q25']:.3f},{r['q75']:.3f}]" if r["median"] is not None else "未达标"
        lines.append(f"| {r['beta']:g} | {r['r0']:g} | {r['j']} | {value} | {r['n'] or 0}/{r['expected_pairs']} |")
    lines += ["", "标定组统一取 a=0.01，这使最小目标 r0=1 也可达。每个实例固定 B、b 和弹性网阈值，仅改变行空间 p0。二次对照仍用 a=0.25、1。",
              f"r0=1000、j=2 时，beta=1/4 的 CP/Polyak-2 中位数分别为 {', '.join(format(v, '.3f') for v in coarse_ratios if v is not None)}；beta=4、r0=1、j=2 时为 {small_ratio:.3f}。CP 的优势随区域和精度变化，并非在所有区域都更快。",
              "CP(c) 的割修正项逐点严格为 A*/c；这不意味着不同轨迹的达标次数必须单调。优势或劣势均按实测报告，理论阈值单独保存在 `e2_theory_thresholds.csv`。", "",
              "## 文件与复现", "", "- `config.json`：全部网格、预算、版本和代码 SHA-256。",
              "- `instances/`：矩阵、参考证书、每条原始轨迹与终点 p/x。",
              "- `tables/`：可直接处理的 CSV 表。",
              "- `figures/`：同名 PNG 与矢量 PDF。",
              "- `verification.json`：独立终点与文件完整性核验。",
              "",
              "主运行并发时间只用于运行记录。计时包含相同的数值诊断，不包含 JIT 编译、参考解构造与文件写入；计算成本比较优先用 M、W。", ""]
    if (output / "verification.json").exists():
        verification = read(output / "verification.json")
        lines += [f"独立核验：{'通过' if verification['passed'] else '未通过'}；复算 {verification['verified_runs']} 个终点、{verification['verified_references']} 个高精度参考证书，失败项 {len(verification['failures'])}。", ""]
    (output / "REPORT_zh.md").write_text("\n".join(lines), encoding="utf-8")
    write_json(output / "analysis_manifest.json", {"runs": len(runs), "statuses": dict(statuses),
               "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
               "figure_count": len(list((output / "figures").glob("*.png"))),
               "table_count": len(list((output / "tables").glob("*.csv")))})
    print(f"Generated report, tables and figures in {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("output")
    generate(parser.parse_args().output)
