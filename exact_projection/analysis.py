"""Summary, paired comparison SSN-A vs SSN-KKT, generated table rows, and the report."""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from rates_and_inexact.common import read, write_json, csv_write
from .protocol import METHODS, TIMED
from .runner import results


def median(values):
    values = [v for v in values if v is not None and np.isfinite(v)]
    return float(np.median(values)) if values else None


def fmt(value, digits=3):
    return "--" if value is None else f"{value:.{digits}g}"


def analyze(output):
    output = Path(output)
    found = results(output)
    with (output/"timing/summary.csv").open(encoding="utf-8-sig", newline="") as stream:
        timing = {(r["source_instance"], r["method"], r["timed"]): r for r in csv.DictReader(stream)}
    summary = []
    for name in METHODS:
        for cfg in ("SSN-KKT", "SSN-A", "SSN-E"):
            values = {s: r for (s, n, c), (_, r) in found.items() if n == name and c == cfg}
            ok = {s: r for s, r in values.items() if r.get("first_distance_N") is not None}
            row = dict(method=name, configuration=cfg, runs=len(values), distance_success=len(ok),
                       joint_success=sum(r.get("first_joint_N") is not None for r in values.values()),
                       failures=sum(r["flag"] >= 5 for r in values.values()),
                       N_median=median([r["first_distance_N"] for r in ok.values()]),
                       W_per_N_median=median([r["first_distance_W"]/r["first_distance_N"] for r in ok.values()]),
                       C_per_N_median=median([r["first_distance_C"]/r["first_distance_N"] for r in ok.values()
                                              if r.get("first_distance_C") is not None]),
                       final_relative_distance_max=max(r["relative_distance"] for r in values.values()))
            for label, source_cfg, harness in TIMED:
                if source_cfg == cfg:
                    row[f"time_ms {label}"] = median([1000*float(timing[(s, name, label)]["median_seconds"])
                                                      for s in ok if (s, name, label) in timing])
                    row[f"timed {label}"] = sum((s, name, label) in timing for s in ok)
            summary.append(row)
    pairs, prefix = [], {}
    for name in METHODS:
        for (s, n, c), (_, kkt) in sorted(found.items()):
            if n != name or c != "SSN-KKT":
                continue
            other = found[(s, name, "SSN-A")][1]
            both = kkt.get("first_distance_N") is not None and other.get("first_distance_N") is not None
            row = dict(source_instance=s, method=name, jointly_successful=both)
            if both:
                tk, ta = (float(timing[(s, name, label)]["median_seconds"]) for label in ("SSN-KKT", "SSN-A"))
                row.update(N_ratio=other["first_distance_N"]/kkt["first_distance_N"],
                           W_ratio=other["first_distance_W"]/kkt["first_distance_W"],
                           identical=(other["first_distance_N"], other["first_distance_W"]) ==
                                     (kkt["first_distance_N"], kkt["first_distance_W"]),
                           time_ratio=ta/tk)
            pairs.append(row)
        logged = [r for (s, n, label), r in timing.items() if n == name and label == "SSN-A (logged)"]
        steps = sum(int(r["accepted_steps"]) for r in logged)
        prefix[name] = dict(
            accepted_steps=steps,
            at_kkt_tolerance=sum(int(r["accepted_at_kkt_tolerance"]) for r in logged)/max(steps, 1),
            start_accepted=sum(int(r["start_accepted"]) for r in logged)/max(steps, 1))
    # Spread between fresh processes: per process, the median over instances of the per-task
    # medians, and the median over instances of the paired ratio SSN-A / SSN-KKT.
    process_rows = []
    for path in sorted((output/"timing").glob("process_*.csv")):
        with path.open(encoding="utf-8-sig", newline="") as stream:
            process_rows += list(csv.DictReader(stream))
    by_task = {}
    for r in process_rows:
        by_task.setdefault((int(r["process"]), r["source_instance"], r["method"], r["timed"]), []).append(float(r["seconds"]))
    processes = sorted({k[0] for k in by_task})
    spread = {}
    for name in METHODS:
        for label in [t[0] for t in TIMED]:
            spread[(name, label)] = [1000*median([np.median(v) for (i, s, n, l), v in by_task.items()
                                                  if i == process and n == name and l == label])
                                     for process in processes]
        ratios = []
        for process in processes:
            values = [np.median(by_task[(process, s, name, "SSN-A")])/np.median(by_task[(process, s, name, "SSN-KKT")])
                      for (i, s, n, l) in by_task if i == process and n == name and l == "SSN-KKT"
                      and (process, s, name, "SSN-A") in by_task]
            ratios.append(median(values))
        spread[(name, "paired time ratio")] = ratios
    paired = {}
    for name in METHODS:
        rows = [r for r in pairs if r["method"] == name and r["jointly_successful"]]
        paired[name] = dict(
            pairs=len(rows), identical=sum(r["identical"] for r in rows),
            N_ratio=[median([r["N_ratio"] for r in rows]), min(r["N_ratio"] for r in rows), max(r["N_ratio"] for r in rows)],
            W_ratio=[median([r["W_ratio"] for r in rows]), min(r["W_ratio"] for r in rows), max(r["W_ratio"] for r in rows)],
            time_ratio=[median([r["time_ratio"] for r in rows]), min(r["time_ratio"] for r in rows), max(r["time_ratio"] for r in rows)])
    csv_write(output/"tables/cost_summary.csv", summary)
    csv_write(output/"tables/cost_pairs.csv", pairs)
    tex = [r"% Generated rows for the elastic-net part of Table tab:h-cost; exact = SSN-KKT, inexact = SSN-A,"
           r" both timed with the lean harness."]
    for name in METHODS:
        for cfg, label in (("SSN-KKT", "exact"), ("SSN-A", "inexact")):
            row = next(r for r in summary if r["method"] == name and r["configuration"] == cfg)
            tex.append(" & ".join(["Elastic net", name, label, f"{row['distance_success']}/{row['runs']}",
                                   f"{row['joint_success']}/{row['runs']}", fmt(row["N_median"], 4),
                                   fmt(row["W_per_N_median"]), fmt(row[f"time_ms {cfg}"])])+r"\\")
    (output/"tables/h_cost_rows.tex").write_text("\n".join(tex)+"\n", encoding="utf-8")
    verification = read(output/"verification.json") if (output/"verification.json").exists() else {}
    report = ["# 非精确投影的成本：同一牛顿代码的精确对照与精简计时", "",
              f"核验：{'通过' if verification.get('passed') else '未通过或未执行'}"
              f"（{verification.get('verified_runs')}/{verification.get('expected_runs')} 条 SSN-KKT 轨迹，"
              f"{len(verification.get('errors', []))} 项完整性错误，{len(verification.get('numerical_failures', []))} 项数值失败）。", "",
              "SSN-KKT 与 SSN-A 使用同一牛顿迭代（起点、单活动约束分支、线搜索相同），只是接受条件不同："
              "SSN-KKT 只接受归一化 KKT 残差不超过 1e-12 的候选点。SSN-A、SSN-E 不重跑，按已保存的首达步重放计时。", "",
              "| 方法 | 配置 | 1% 成功 | 联合成功 | N | W/N | C/N | 精简计时 ms | 带日志计时 ms | 旧代码计时 ms |",
              "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for row in summary:
        lean = row.get("time_ms SSN-KKT") if row["configuration"] == "SSN-KKT" else row.get("time_ms SSN-A")
        report.append(f"| {row['method']} | {row['configuration']} | {row['distance_success']}/{row['runs']} | "
                      f"{row['joint_success']}/{row['runs']} | {fmt(row['N_median'], 4)} | {fmt(row['W_per_N_median'])} | "
                      f"{fmt(row['C_per_N_median'])} | {fmt(lean) if row['configuration'] != 'SSN-E' else '--'} | "
                      f"{fmt(row.get('time_ms SSN-A (logged)'))} | {fmt(row.get('time_ms SSN-E'))} |")
    report += ["", "成对比较（SSN-A 相对 SSN-KKT，两者都达到 1% 的实例；括号内为最小值与最大值）：", ""]
    for name, v in paired.items():
        report.append(f"- {name}：{v['pairs']} 对，N 与 W 完全相同 {v['identical']} 对；N 比值中位数 {v['N_ratio'][0]:.3f}"
                      f" [{v['N_ratio'][1]:.2f}, {v['N_ratio'][2]:.2f}]，W 比值 {v['W_ratio'][0]:.3f}"
                      f" [{v['W_ratio'][1]:.2f}, {v['W_ratio'][2]:.2f}]，精简计时比值 {v['time_ratio'][0]:.3f}"
                      f" [{v['time_ratio'][1]:.2f}, {v['time_ratio'][2]:.2f}]。")
    report += ["", f"进程间波动（{len(processes)} 个先后启动的独立进程；每个进程内对实例取中位数，单位 ms；"
               "最后一行是每个进程内 SSN-A/SSN-KKT 成对计时比值的中位数）：", "",
               "| 方法 | 计时 | " + " | ".join(f"进程 {i}" for i in processes) + " |",
               "|---|---|" + "---:|"*len(processes)]
    for (name, label), values in spread.items():
        report.append(f"| {name} | {label} | " + " | ".join(fmt(v) for v in values) + " |")
    report += ["", "SSN-A 在首达 1% 之前的已接受外层步（带日志重放，逐步统计）：", ""]
    for name, v in prefix.items():
        report.append(f"- {name}：{v['accepted_steps']} 步；被接受的乘子已满足 KKT 容差 1e-12 的比例 {v['at_kkt_tolerance']:.3f}；"
                      f"起点即被接受的比例 {v['start_accepted']:.3f}。")
    report += ["", "说明：", "",
               "- 精简计时与带日志计时运行同一算法、得到相同的 N 和 W；差别只在逐步日志与诊断 ||n + A^T w||（每个外层步一次矩阵–向量乘）。",
               "- 旧代码计时重放 elastic_net 的精确实现（SSN-E），只记录两个步，与 rates_and_inexact 的计时口径相同。",
               "- 每次重放都核对 N、W 和终点的 1% 距离。每个任务在每个进程内重复三次取中位数，再对进程取中位数，最后对实例取中位数。",
               "- 不同进程之间的中位数可相差 1.7 倍，且对两个分支的影响不同，因此计时使用多个先后启动的进程。",
               "- 计时只说明本实现在 n=40 时的开销；W/N 才是核映射昂贵时的相关度量。", ""]
    (output/"REPORT_zh.md").write_text("\n".join(report), encoding="utf-8")
    write_json(output/"analysis.json", dict(summary=summary, paired=paired, prefix=prefix,
                                            process_spread={f"{n} | {l}": v for (n, l), v in spread.items()}))
    return summary, paired, prefix
