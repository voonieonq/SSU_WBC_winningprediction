"""STEP 1 — 리그 계수 (MLB=1.00 기준, 지표별)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

from models import Batter, Pitcher, Team


@dataclass
class LeagueCoeffs:
    ops: float = 1.0
    era: float = 1.0
    k_rate: float = 1.0
    bb_rate: float = 1.0
    hr_rate: float = 1.0
    salary_ratio: float = 1.0  # 보조 검증용


def _mean(xs: List[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def compute_league_coefficients(teams: List[Team]) -> Dict[str, LeagueCoeffs]:
    """8강 참가 선수 풀에서 리그별 평균 → MLB 대비 비율."""
    ops: Dict[str, List[float]] = {}
    era: Dict[str, List[float]] = {}
    k_b: Dict[str, List[float]] = {}
    bb_p: Dict[str, List[float]] = {}
    hr_b: Dict[str, List[float]] = {}
    salary: Dict[str, List[float]] = {}

    for team in teams:
        if team.avg_salary_usd_m:
            salary.setdefault("MLB", []).append(team.avg_salary_usd_m)
        for b in team.batters:
            ops.setdefault(b.league, []).append(b.ops)
            if b.g > 0:
                k_b.setdefault(b.league, []).append(b.k / b.g)
                hr_b.setdefault(b.league, []).append(b.hr / b.g)
        for p in team.pitchers:
            era.setdefault(p.league, []).append(p.era)
            if p.ip > 0:
                bb_p.setdefault(p.league, []).append(p.bb / p.ip)

    mlb_ops = _mean(ops.get("MLB", [0.82]))
    mlb_era = _mean(era.get("MLB", [3.2]))
    mlb_k = _mean(k_b.get("MLB", [0.65]))
    mlb_bb = _mean(bb_p.get("MLB", [0.28]))
    mlb_hr = _mean(hr_b.get("MLB", [0.18]))

    leagues = set(list(ops.keys()) + list(era.keys()))
    out: Dict[str, LeagueCoeffs] = {}
    for lg in leagues:
        sal = _mean(salary.get(lg, [1.0]))
        mlb_sal = _mean(salary.get("MLB", [5.0]))
        out[lg] = LeagueCoeffs(
            ops=_mean(ops.get(lg, [])) / mlb_ops if mlb_ops else 1.0,
            era=_mean(era.get(lg, [])) / mlb_era if mlb_era else 1.0,
            k_rate=_mean(k_b.get(lg, [])) / mlb_k if mlb_k else 1.0,
            bb_rate=_mean(bb_p.get(lg, [])) / mlb_bb if mlb_bb else 1.0,
            hr_rate=_mean(hr_b.get(lg, [])) / mlb_hr if mlb_hr else 1.0,
            salary_ratio=sal / mlb_sal if mlb_sal else 1.0,
        )
    out["MLB"] = LeagueCoeffs(
        ops=1.0, era=1.0, k_rate=1.0, bb_rate=1.0, hr_rate=1.0, salary_ratio=1.0
    )
    return out
