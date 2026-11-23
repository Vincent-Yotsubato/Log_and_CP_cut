"""Read the saved upstream results without regenerating or modifying them."""
from __future__ import annotations

import csv
import gzip
import hashlib
import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import mpmath as mp
import numpy as np

from baseline.runner import write_json
from elastic_net.problems import Problem

EXPERIMENTS = Path(__file__).resolve().parents[1]
SOURCES = {
    "baseline": EXPERIMENTS / "results/baseline",
    "nonsmooth": EXPERIMENTS / "results/elastic_net/nonsmooth",
    "hlog": EXPERIMENTS / "results/elastic_net/hlog",
}
CURRENT_METHODS = {"Polyak-1", "Polyak-2", "CP", "Log-LBreI", "H-CP", "H-Log"}


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def csv_write(path, rows, fields=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = list(rows)
    fields = fields or list(dict.fromkeys(k for row in rows for k in row))
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def save_trace(path, columns, values):
    with gzip.open(path, "wt", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(columns)
        writer.writerows(values)


def trace(path):
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        columns = stream.readline().strip().split(",")
        values = np.loadtxt(stream, delimiter=",", ndmin=2)
    return {name: values[:, i] for i, name in enumerate(columns)}


def source_jobs(source):
    return read(SOURCES[source] / "config.json")["jobs"]


def instance(source, instance_id):
    return SOURCES[source] / "instances" / instance_id


@lru_cache(maxsize=256)
def load_problem(directory):
    directory = Path(directory)
    data = np.load(directory / "instance.npz")
    ref = read(directory / "reference.json")
    job = read(directory / "job.json")
    model = job["model"]
    return Problem(*(data[k] for k in ("A", "b")), int(model["kind"] == "poly"),
                   model.get("beta", 0.), model.get("power", 4), ref["L0"], ref["L1"],
                   ref["lambda_en"], data["p0"], data["z"], data["reference_dual"],
                   ref["certificate"], ref["metadata"])


@lru_cache(maxsize=None)
def reference_interval(directory):
    """Bounds from the saved feasible auxiliary point, not Omega(z_ref).

    All experiment kernels have mu=1. Certificate.distance_bound already includes
    the distance from the stored vector to the auxiliary feasible vector.
    The arithmetic is high precision, but is not outward-rounded interval math.
    """
    directory = Path(directory)
    p = load_problem(directory)
    with mp.workdps(65):
        u = [mp.mpf(v) for v in p.certificate["auxiliary_feasible_point"]]
        p0 = [mp.mpf(float(v)) for v in p.p0]
        lam = mp.mpf(float(p.lam))
        upper = sum(v*v/2 + lam*abs(v) - a*v for v, a in zip(u, p0))
        gap = mp.mpf(float(p.certificate["gap"]))
        x0 = np.sign(p.p0)*np.maximum(np.abs(p.p0)-p.lam, 0.)
        initial = sum(mp.mpf(float(v))**2/2 + lam*abs(mp.mpf(float(v)))
                      - a*mp.mpf(float(v)) for v, a in zip(x0, p0))
        lower = upper-gap
        return dict(omega_lower=float(lower), omega_upper=float(upper),
                    radius2_lower=max(0., float(2*(lower-initial))),
                    radius2_upper=max(0., float(2*(upper-initial))),
                    reference_gap=float(gap),
                    reference_distance_bound=float(p.certificate["distance_bound"]))


@dataclass
class Record:
    source: str
    job: dict
    directory: Path
    result_path: Path

    @property
    def result(self):
        return read(self.result_path)


def records(output=None):
    """Cases used in the paper: the five baseline variants, CP with c = 1 and theta = 1."""
    for source in SOURCES:
        for job in source_jobs(source):
            if source == "baseline" and job["variant"] not in {
                    "example32", "polyhedron", "orthogonal_control", "calibrated", "quadratic_control"}:
                continue
            d = instance(source, job["id"])
            for path in sorted(d.glob("*/result.json")):
                method = read(path)["method"]
                if (method["name"] in CURRENT_METHODS and method.get("c", 1.) == 1.
                        and method.get("theta", 1.) == 1.):
                    yield Record(source, job, d, path)
    if output is not None:
        for panel in ("inexact",):
            for path in sorted((Path(output)/panel/"instances").glob("*/*/result.json")):
                d = path.parent.parent
                yield Record(panel, read(d/"job.json"), d, path)
