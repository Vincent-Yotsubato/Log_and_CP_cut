"""Summary, paired comparisons, process spread, generated table rows, and the report."""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from rates_and_inexact.common import read, write_json, csv_write
from .protocol import METHODS, TIMED, CONFIGURATIONS
from .runner import results

NEW = tuple(c["name"] for c in CONFIGURATIONS)
ALL = NEW+("SSN-KKT", "SSN-A", "SSN-E")
LEAN = {"SSN-R": "SSN-R", "SSN-R1": "SSN-R1", "SSN-R+": "SSN-R+", "SSN-KKT": "SSN-KKT", "SSN-A": "SSN-A"}


def median(values):
    values = [v for v in values if v is not None and np.isfinite(v)]
    return float(np.median(values)) if values else None


def fmt(value, digits=3):
    return "--" if value is None else f"{value:.{digits}g}"


def spread(values):
    return [median(values), min(values), max(values)] if values else [None, None, None]


def analyze(output):
    output = Path(output)
    found = results(output)
    with (output/"timing/summary.csv").open(encoding="utf-8-sig", newline="") as stream:
        timing = {(r["source_instance"], r["method"], r["timed"]): float(r["median_seconds"]) for r in csv.DictReader(stream)}
    summary = []
    for name in METHODS:
        for cfg in ALL:
            values = {s: r for (s, n, c), (_, r) in found.items() if n == name and c == cfg}
            if not values:
                continue
            ok = {s: r for s, r in values.items() if r.get("first_distance_N") is not None}
            row = dict(method=name, configuration=cfg, runs=len(values), distance_success=len(ok),
                       joint_success=sum(r.get("first_joint_N") is not None for r in values.values()),
                       failures=sum(r["flag"] >= 5 and not (r["flag"] == 8 and cfg.startswith("PG")) for r in values.values()),
                       inner_cap_stops=sum(r["flag"] == 8 for r in values.values()),
                       median_outer_steps_before_stop=median([r["N"] for r in values.values()]),
                       N_median=median([r["first_distance_N"] for r in ok.values()]),
                       W_per_N_median=median([r["first_distance_W"]/r["first_distance_N"] for r in ok.values()]),
                       C_per_N_median=median([r["first_distance_C"]/r["first_distance_N"] for r in ok.values()
                                              if r.get("first_distance_C") is not None]),
                       final_relative_distance_max=max(r["relative_distance"] for r in values.values()),
                       time_ms=median([1000*timing[(s, name, LEAN[cfg])] for s in ok if (s, name, LEAN.get(cfg)) in timing])
                       if cfg in LEAN else None)
            if cfg in NEW:
                steps = sum(r["prefix_accepted_steps"] for r in values.values())
                total = sum(r["K"] for r in values.values())
                row.update(prefix_accepted_steps=steps,
                           prefix_at_kkt_tolerance=sum(r["prefix_at_kkt_tolerance"] for r in values.values())/max(steps, 1),
                           prefix_start_accepted=sum(r["prefix_start_accepted"] for r in values.values())/max(steps, 1),
                           prefix_negative_residual=sum(r["prefix_negative_residual"] for r in values.values())/max(steps, 1),
                           floor_fraction=sum(r["eta_floor_uses"] for r in values.values())/max(total, 1),
                           negative_residual_fraction=sum(r["negative_residual_steps"] for r in values.values())/max(total, 1))
            summary.append(row)
    pairs, paired = [], {}
    for name in METHODS:
        for cfg, base in (("SSN-R", "SSN-KKT"), ("SSN-R", "SSN-A"), ("SSN-R1", "SSN-KKT"),
                          ("SSN-R+", "SSN-KKT"), ("SSN-A", "SSN-KKT")):
            rows = []
            for (s, n, c), (_, r) in sorted(found.items()):
                if n != name or c != cfg or (s, name, base) not in found:
                    continue
                o = found[(s, name, base)][1]
                both = r.get("first_distance_N") is not None and o.get("first_distance_N") is not None
                row = dict(source_instance=s, method=name, configuration=cfg, comparator=base, jointly_successful=both)
                if both:
                    row.update(N_ratio=r["first_distance_N"]/o["first_distance_N"],
                               W_ratio=r["first_distance_W"]/o["first_distance_W"],
                               identical=(r["first_distance_N"], r["first_distance_W"]) ==
                                         (o["first_distance_N"], o["first_distance_W"]))
                    if (s, name, LEAN[cfg]) in timing and (s, name, LEAN[base]) in timing:
                        row["time_ratio"] = timing[(s, name, LEAN[cfg])]/timing[(s, name, LEAN[base])]
                pairs.append(row)
                rows.append(row)
            ok = [r for r in rows if r["jointly_successful"]]
            paired[f"{name} | {cfg} / {base}"] = dict(
                pairs=len(ok), identical=sum(r["identical"] for r in ok),
                only_first=sum(1 for r in rows if not r["jointly_successful"] and found[(r["source_instance"], name, cfg)][1].get("first_distance_N") is not None),
                only_comparator=sum(1 for r in rows if not r["jointly_successful"] and found[(r["source_instance"], name, base)][1].get("first_distance_N") is not None),
                N_ratio=spread([r["N_ratio"] for r in ok]), W_ratio=spread([r["W_ratio"] for r in ok]),
                time_ratio=spread([r["time_ratio"] for r in ok if "time_ratio" in r]))
    process_rows = []
    for path in sorted((output/"timing").glob("process_*.csv")):
        with path.open(encoding="utf-8-sig", newline="") as stream:
            process_rows += list(csv.DictReader(stream))
    by_task = {}
    for r in process_rows:
        by_task.setdefault((int(r["process"]), r["source_instance"], r["method"], r["timed"]), []).append(float(r["seconds"]))
    processes = sorted({k[0] for k in by_task})
    per_process = {}
    for name in METHODS:
        for label in [t[0] for t in TIMED]:
            per_process[f"{name} | {label} (ms)"] = [
                1000*median([np.median(v) for (i, s, n, l), v in by_task.items() if i == pr and n == name and l == label])
                for pr in processes]
        for label, base in (("SSN-R", "SSN-KKT"), ("SSN-A", "SSN-KKT"), ("SSN-R", "SSN-A"),
                            ("SSN-KKT", "SSN-KKT (frozen code)")):
            ratios = []
            for pr in processes:
                values = [np.median(by_task[(pr, s, name, label)])/np.median(by_task[(pr, s, name, base)])
                          for (i, s, n, l) in by_task if i == pr and n == name and l == base
                          and (pr, s, name, label) in by_task]
                ratios.append(median(values))
            per_process[f"{name} | {label} / {base}"] = ratios
    csv_write(output/"tables/residual_summary.csv", summary)
    csv_write(output/"tables/residual_pairs.csv", pairs)
    tex = [r"% Generated rows for the elastic-net part of Table tab:h-cost; exact = SSN-KKT, inexact = SSN-R,"
           r" both timed with the lean harness of inexact_projection."]
    for name in METHODS:
        for cfg, label in (("SSN-KKT", "exact"), ("SSN-R", "inexact")):
            row = next(r for r in summary if r["method"] == name and r["configuration"] == cfg)
            tex.append(" & ".join(["Elastic net", name, label, f"{row['distance_success']}/{row['runs']}",
                                   f"{row['joint_success']}/{row['runs']}", fmt(row["N_median"], 4),
                                   fmt(row["W_per_N_median"]), fmt(row["time_ms"])])+r"\\")
    (output/"tables/h_cost_rows.tex").write_text("\n".join(tex)+"\n", encoding="utf-8")
    verification = read(output/"verification.json") if (output/"verification.json").exists() else {}
    report = ["# 残差接受条件 |c+ - <n+, x+>| <= eta 的实验", "",
              f"核验：{'通过' if verification.get('passed') else '未通过或未执行'}"
              f"（{verification.get('verified_runs')}/{verification.get('expected_runs')} 条新轨迹，"
              f"{len(verification.get('errors', []))} 项完整性错误，{len(verification.get('numerical_failures', []))} 项数值失败）。", "",
              "所有配置使用同一牛顿迭代（起点、单活动约束分支、线搜索相同），只有接受条件不同。"
              "SSN-KKT：KKT 残差 <= 1e-12；SSN-A：修正点证书条件；SSN-R：双侧残差（主配置）；"
              "SSN-R1：双侧残差、容差 ~k^-1；SSN-R+：单侧残差（诊断）；PG-R、PG0-R：双侧残差配投影梯度内层求解（诊断，预期触到内层上限）。"
              "SSN-E 为旧精确代码，只列出不计时；其 flag 1–3 是正常终止，不算失败。", "",
              "| 方法 | 配置 | 1% 成功 | 联合成功 | 失败 | 内层触顶 | N | W/N | C/N | 计时 ms | 最终最大相对距离 |",
              "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for row in summary:
        report.append(f"| {row['method']} | {row['configuration']} | {row['distance_success']}/{row['runs']} | "
                      f"{row['joint_success']}/{row['runs']} | {row['failures']} | {row['inner_cap_stops']} | {fmt(row['N_median'], 4)} | "
                      f"{fmt(row['W_per_N_median'])} | {fmt(row['C_per_N_median'])} | {fmt(row['time_ms'])} | "
                      f"{fmt(row['final_relative_distance_max'])} |")
    report += ["", "新配置的逐步统计（首达 1% 之前的已接受步；比例对全部外层步汇总）：", "",
               "| 方法 | 配置 | 首达前已接受步 | 其中满足 KKT 1e-12 | 其中起点被接受 | 其中残差为负 | 浮点下限启用（全程） | 残差为负（全程） |",
               "|---|---|---:|---:|---:|---:|---:|---:|"]
    for row in summary:
        if row["configuration"] in NEW:
            report.append(f"| {row['method']} | {row['configuration']} | {row['prefix_accepted_steps']} | "
                          f"{row['prefix_at_kkt_tolerance']:.3f} | {row['prefix_start_accepted']:.3f} | "
                          f"{row['prefix_negative_residual']:.3f} | {row['floor_fraction']:.3f} | {row['negative_residual_fraction']:.3f} |")
    report += ["", "成对比较（两者都达到 1% 的实例；[中位数, 最小, 最大]）：", ""]
    for key, v in paired.items():
        report.append(f"- {key}：{v['pairs']} 对，N、W 完全相同 {v['identical']} 对，仅前者成功 {v['only_first']}、仅后者成功 "
                      f"{v['only_comparator']}；N 比值 {[fmt(x) for x in v['N_ratio']]}，W 比值 {[fmt(x) for x in v['W_ratio']]}，"
                      f"计时比值 {[fmt(x) for x in v['time_ratio']]}。")
    report += ["", f"进程间波动（{len(processes)} 个先后启动的独立进程；每个进程内对实例取中位数）：", "",
               "| 项目 | " + " | ".join(f"进程 {i}" for i in processes) + " |", "|---|" + "---:|"*len(processes)]
    for key, values in per_process.items():
        report.append(f"| {key} | " + " | ".join(fmt(v) for v in values) + " |")
    report += ["", "说明：", "",
               "- 计时都用精简外层循环，不含逐步日志。SSN-KKT (frozen code) 用冻结代码计时，用来核对复制的代码在计时上没有差别。",
               "- 每次重放都核对 N、W 和终点的 1% 距离；SSN-KKT 与 SSN-A 的重放还说明复制代码的模式 0、1 与冻结代码逐步一致。",
               "- 每个任务在每个进程内重复三次取中位数，再对进程取中位数，最后对实例取中位数。", ""]
    (output/"REPORT_zh.md").write_text("\n".join(report), encoding="utf-8")
    write_json(output/"analysis.json", dict(summary=summary, paired=paired, process_spread=per_process))
    return summary, paired, per_process
