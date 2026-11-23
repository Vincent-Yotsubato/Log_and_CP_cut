"""Rate analysis of the exact H runs: frozen tail windows, explicit non-fits, and rho_c=0 controls."""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path
import numpy as np
from .common import records, load_problem, reference_interval, trace, csv_write, write_json
from .protocol import SETTINGS


def fit(k, value, mask):
    mask = mask & np.isfinite(value) & (value > 0.) & (k > 0.)
    count = int(np.sum(mask))
    if count < SETTINGS["rate_minimum_points"]:
        return None, count
    return float(np.polyfit(np.log(k[mask]), np.log(value[mask]), 1)[0]), count


def quantile(values, q=.5):
    values = [v for v in values if v is not None and np.isfinite(v)]
    return float(np.quantile(values, q)) if values else None


def analyze(output):
    output = Path(output)
    rows, groups, control = [], defaultdict(list), defaultdict(list)
    displays = defaultdict(list)
    for rec in records():
        r = rec.result
        model = rec.job["model"]
        name = r["method"]["name"]
        if rec.source == "baseline" and rec.job["variant"] == "polyhedron" and model["correlation"] == 0.:
            control[name].append(r)
        correlated = (rec.source == "baseline" and rec.job["variant"] == "polyhedron" and model["correlation"] == .8)
        if not (name in ("H-CP", "H-Log") and (correlated or rec.source in ("nonsmooth", "hlog"))):
            continue
        p = load_problem(rec.directory)
        bounds = reference_interval(str(rec.directory))
        t = trace(rec.result_path.parent/"trace.csv.gz")
        k, E, best = t["k"], t["Omega_gap"], t["best_delta"]
        E0, K, s = float(E[0]), float(k[-1]), int(model["power"])
        mask = ((k >= K*SETTINGS["rate_window_fraction"]) & (k <= K)
                & (E >= SETTINGS["rate_gap_floor"]*bounds["reference_gap"])
                & (E <= SETTINGS["rate_initial_ceiling"]*E0))
        slope, count = fit(k, E, mask)
        delta_slope, delta_count = fit(k, best, mask)
        distance_slope, distance_count = fit(k, t["dist_to_zdag"], mask)
        singular = np.linalg.svd(p.A, compute_uv=False)
        growth = singular[-1]**s*p.A.shape[0]**(1-s/2)/s
        chi = growth/singular[0]**s
        zeta_ref = np.linalg.norm(p.A.T@p.dual)
        # A reference dual is approximate. Provide both the requested plug-in
        # multiplier and a conservative norm upper bound valid for any elastic
        # net subgradient at the true reference. Sparse q.max is NOT sup_k q_k.
        zeta_upper = np.linalg.norm(p.p0)+np.linalg.norm(p.z)+bounds["reference_distance_bound"]+p.lam*np.sqrt(p.z.size)
        radius = np.sqrt(bounds["radius2_upper"])+bounds["reference_distance_bound"]
        v_bound = np.linalg.norm(np.maximum(p.b-p.A@p.z, 0.))+singular[0]*radius
        q_upper = singular[0]*v_bound**(s-1)
        Lbd = p.L0+p.L1*q_upper
        E0_upper = bounds["radius2_upper"]/2
        a = chi**2/(2*zeta_upper**2)
        cH = growth/(Lbd*zeta_upper**s)
        glog = growth**2/(2*Lbd**2*zeta_upper**(2*s-2))
        cp_growth_ratio = (1+(s-1)*cH*K*E0_upper**(s-1))**(-1/(s-1))
        ratio = 1/(1+a*E0_upper*K) if name == "H-CP" else (1+(2*s-3)*glog*K*E0_upper**(2*s-3))**(-1/(2*s-3))
        # Also show the reference-multiplier/sample-q illustration, labelled as such:
        # a sampled maximum is not a certificate.
        sample_Lbd = p.L0+p.L1*float(np.max(t["q"]))
        plug_a = chi**2/(2*zeta_ref**2)
        plug_g = growth**2/(2*sample_Lbd**2*zeta_ref**(2*s-2))
        plugin_ratio = 1/(1+plug_a*E0*K) if name == "H-CP" else (1+(2*s-3)*plug_g*K*E0**(2*s-3))**(-1/(2*s-3))
        group = "extended" if rec.source == "hlog" else "euclidean" if correlated else "elastic_net"
        row = dict(group=group, instance_id=rec.job["id"], method=name, power=s, lambda_en=p.lam,
                   K=int(K), E0=E0, E_final=float(E[-1]), E_slope=slope, E_fit_points=count,
                   delta_slope=delta_slope, delta_fit_points=delta_count,
                   distance_slope=distance_slope, distance_fit_points=distance_count,
                   E_order=-1. if name == "H-CP" else -1/(2*s-3),
                   delta_order=-2. if name == "H-CP" else -s/(2*s-3),
                   cp_growth_only_order=-1/(s-1), chi_poly=float(chi), growth_c=float(growth),
                   zeta_reference=float(zeta_ref), zeta_upper=float(zeta_upper), Lbd_upper=float(Lbd),
                   bound_final_over_E0=float(ratio), cp_growth_bound_final_over_E0=float(cp_growth_ratio),
                   plugin_bound_final_over_E0=float(plugin_ratio),
                   plugin_scope="reference multiplier and sampled q maximum; illustrative, not a certified sup")
        rows.append(row)
        groups[(group, s, name)].append(row)
        if group != "extended":
            displays[(group, s, name)].append((k.copy(), E/E0))
    summary = []
    for (group, power, name), values in sorted(groups.items()):
        summary.append(dict(group=group, power=power, method=name, trajectories=len(values),
                            fitted_E=sum(v["E_slope"] is not None for v in values),
                            unfitted_E=sum(v["E_slope"] is None for v in values),
                            E_slope_median=quantile([v["E_slope"] for v in values]),
                            E_slope_q25=quantile([v["E_slope"] for v in values], .25),
                            E_slope_q75=quantile([v["E_slope"] for v in values], .75),
                            fitted_delta=sum(v["delta_slope"] is not None for v in values),
                            delta_slope_median=quantile([v["delta_slope"] for v in values]),
                            bound_final_over_E0_median=quantile([v["bound_final_over_E0"] for v in values]),
                            chi_min=min(v["chi_poly"] for v in values), chi_max=max(v["chi_poly"] for v in values)))
    control_rows = [dict(method=name, instances=len(values),
                         distance_success=sum(r["relative_distance"] <= .01 for r in values),
                         joint_success=sum(r["selection_success"] for r in values)) for name, values in sorted(control.items())]
    csv_write(output/"rates/rates_runs.csv", rows)
    csv_write(output/"rates/rates_summary.csv", summary)
    csv_write(output/"rates/uncorrelated_success.csv", control_rows)
    write_json(output/"rates/summary.json", dict(trajectories=len(rows), groups=summary,
               controls=control_rows, window={k: v for k, v in SETTINGS.items() if k.startswith("rate_")},
               interpretation="Finite-window slopes describe observations; they do not establish asymptotic rates or sharpness."))
    draw(output, displays)
    print(f"R2: {len(rows)} H trajectories; R3: {sum(len(v) for v in control.values())} archived trajectories", flush=True)
    return summary


def draw(output, displays):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4), constrained_layout=True)
    for ax, group in zip(axes, ("euclidean", "elastic_net")):
        for (g, s, name), trajectories in sorted(displays.items()):
            if g != group:
                continue
            # Display interpolation only; slope fits use original stored points.
            # Stopped zero-objective exact H states are fixed points. np.interp
            # extends their final value horizontally on the display grid.
            end = max(max(t[0]) for t in trajectories)
            grid = np.geomspace(1, max(end, 1), 200)
            curves = np.array([np.interp(np.log(grid), np.log(np.maximum(k, 1)), np.maximum(y, 1e-18)) for k, y in trajectories])
            q1, med, q3 = np.quantile(curves, [.25, .5, .75], axis=0)
            line, = ax.loglog(grid, med, label=f"{name}, s={s}")
            ax.fill_between(grid, q1, q3, color=line.get_color(), alpha=.12)
            order = -1. if name == "H-CP" else -1/(2*s-3)
            anchor = max(1, len(grid)//2)
            ax.loglog(grid[anchor:], med[anchor]*(grid[anchor:]/grid[anchor])**order,
                      "--", color=line.get_color(), linewidth=.8)
        ax.set(title="Euclidean kernel" if group == "euclidean" else "Elastic-net kernel",
               xlabel="outer updates k", ylabel=r"$E_k/E_0$")
        ax.legend(fontsize=7)
        ax.grid(alpha=.2)
    dest = Path(output)/"rates"
    fig.savefig(dest/"rates_candidate.pdf")
    fig.savefig(dest/"rates_candidate.png", dpi=160)
    plt.close(fig)
