"""Tables and the status report of the inexact-projection panel and the rate analysis."""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path
import numpy as np
from .common import read, write_json, csv_write
from .protocol import PANELS


def median(values):
    values = [v for v in values if v is not None and np.isfinite(v)]
    return float(np.median(values)) if values else None


def fmt(value):
    return "--" if value is None else f"{value:.4g}"


def inexact_costs(output):
    groups, steps, costs = defaultdict(list), defaultdict(list), []
    for path in sorted((output/"inexact/instances").glob("*/*/result.json")):
        r = read(path)
        r["run_id"] = path.parent.name
        key = (r["method"]["name"], r["method"]["configuration"])
        groups[key].append(r)
        # audit.csv.gz holds only a sample of steps; the histogram covers every attempt.
        for level, count in r.get("inner_iterations_histogram", []):
            steps[key].append(np.full(count, level, dtype=float))
    for (name, cfg), values in sorted(groups.items()):
        successful = [r for r in values if r.get("first_distance_N") is not None]
        costs.append(dict(
            method=name, configuration=cfg, main=cfg in ("PG-A", "SSN-A"), runs=len(values),
            distance_success=len(successful), joint_success=sum(r.get("first_joint_N") is not None for r in values),
            algorithm_failures=sum(r["flag"] >= 5 for r in values),
            N_median=median([r["first_distance_N"] for r in successful]),
            W_per_N_median=median([r["first_distance_W"]/r["first_distance_N"] for r in successful]),
            C_per_N_median=median([r.get("first_distance_C", 0)/r["first_distance_N"] for r in successful
                                   if r.get("first_distance_C") is not None]),
            inner_iterations_median=median(np.concatenate(steps[(name, cfg)])) if steps[(name, cfg)] else None,
            inner_outer_attempts=sum(len(v) for v in steps[(name, cfg)]),
            inner_iterations_max=max((r["inner_iterations_max"] for r in values if "inner_iterations_max" in r), default=None),
            historical_inner_kernel_calls_max=max((r["max_inner"] for r in values if "max_inner" in r), default=None),
            start_accepted_fraction=(sum(r["start_accepts"] for r in values)/max(1, sum(r["K"] for r in values))
                                     if all("start_accepts" in r for r in values) else None),
            floor_uses=sum(r.get("eta_floor_uses", 0) for r in values),
            pg_i_violations=sum(r.get("pg_i_violations", 0) for r in values),
            final_relative_distance_median=median([r["relative_distance"] for r in values])))
    paired = []
    for (name, cfg), values in sorted(groups.items()):
        for comparator in ("SSN-E", "PG-E"):
            if cfg == comparator:
                continue
            reference = {r["instance_id"]: r for r in groups.get((name, comparator), [])}
            for r in values:
                other = reference.get(r["instance_id"])
                if other is None:
                    continue
                success = r.get("first_distance_N") is not None and other.get("first_distance_N") is not None
                paired.append(dict(instance_id=r["instance_id"], method=name, configuration=cfg,
                                   comparator=comparator, jointly_successful=success,
                                   N_ratio=r["first_distance_N"]/other["first_distance_N"] if success else None,
                                   W_ratio=r["first_distance_W"]/other["first_distance_W"] if success else None,
                                   flag=r["flag"], comparator_flag=other["flag"]))
    return costs, paired


def analyze(output):
    output = Path(output)
    costs, paired = inexact_costs(output)
    # Raw counts, first hits and complete pairs; failures stay in the denominators.
    csv_write(output/"tables/inexact_summary.csv", costs)
    csv_write(output/"tables/inexact_pairs.csv", paired)
    frozen = read(output/"config.json")
    report = ["# 非精确两半空间投影与收敛速率分析", "",
              "**这是 500 次梯度、种子 0 的试运行，不是正式结果。**" if frozen["pilot"] else "正式配置；完成程度见 verification.json 的 coverage。",
              "", "原始模型、轨迹和参考证书只读。", "",
              "## 执行状态", "", "| 面板 | 实际运行数 | 预定运行数 |", "|---|---:|---:|"]
    for panel in PANELS:
        actual = len(list((output/panel/"instances").glob("*/*/result.json")))
        expected = sum(len(j["methods"]) for j in frozen["jobs"][panel])
        report.append(f"| {panel} | {actual} | {expected} |")
    report += ["", "## 非精确 H 投影", "",
               "主配置预先固定为 PG-A 与 SSN-A，(eta_rel,p)=(1e-2,2)。PG-* 在归一化坐标 (lambda||g||, nu||n||) 上做投影梯度，步长 mu/(1+|cos(g,n)|)；"
               "PG0-A 按原始坐标与步长 1/L_Psi 执行，仅作诊断（q_k 远小于 ||n_k|| 时极慢）。SSN-E 调用 elastic_net 的高精度算法，是复现基准。", "",
               "| 方法 | 配置 | 1% 成功 | 联合成功 | 算法失败 | N 中位数 | W/N 中位数 | 浮点下限启用 |", "|---|---|---:|---:|---:|---:|---:|---:|"]
    for row in costs:
        report.append(f"| {row['method']} | {row['configuration']} | {row['distance_success']}/{row['runs']} | "
                      f"{row['joint_success']}/{row['runs']} | {row['algorithm_failures']} | {fmt(row['N_median'])} | "
                      f"{fmt(row['W_per_N_median'])} | {row['floor_uses']} |")
    report += ["", "内层上限失败不接受候选点；失败轨迹仍计入分母。C 单独记录证书内积成本。",
               "浮点下限或进展检查的浮点容差都是对精确算术条件的数值实现偏离，启用次数见表及逐步 audit.csv.gz。",
               "仅在两种方法都成功的同一批实例上比较成对成本。", "", "## 速率（rates/）", ""]
    path = output/"rates/summary.json"
    if path.exists():
        b = read(path)
        report += [f"- {b['trajectories']} 条精确 H 轨迹；rates_summary.csv 同时报可拟合和不可拟合的数量。",
                   "- 全局常数的代入与有限窗口斜率分别报告；斜率不能证明渐近阶，也不能证明理论界尖锐。",
                   "- rho_c=0 的成功数和联合成功数见 rates/uncorrelated_success.csv。"]
    else:
        report.append("- rates 尚未执行。")
    report.append("")
    (output/"REPORT_zh.md").write_text("\n".join(report), encoding="utf-8")
    write_json(output/"analysis.json", dict(inexact=costs, pilot=frozen["pilot"]))
    return costs
