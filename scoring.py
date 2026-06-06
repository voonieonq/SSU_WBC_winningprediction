"""STEP 2 — 선수 점수 (0~100, Min-Max, 좌/우 split)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from league_coeff import LeagueCoeffs
from models import Batter, Pitcher, Team


def _clamp(x: float, lo: float = 0.0, hi: float = 100.0) -> float:
    return max(lo, min(hi, x))


def minmax_norm(value: float, vmin: float, vmax: float) -> float:
    if vmax <= vmin:
        return 50.0
    return _clamp((value - vmin) / (vmax - vmin) * 100.0)


@dataclass
class StatBounds:
    vmin: float
    vmax: float


@dataclass
class ScoringContext:
    bounds: Dict[str, StatBounds]
    coeffs: Dict[str, LeagueCoeffs]
    rookie_g_threshold: int = 30
    rookie_shrink: float = 0.35  # 신인: 리그 평균으로 수축


def _league_adj_ops(ops: float, league: str, coeffs: Dict[str, LeagueCoeffs]) -> float:
    c = coeffs.get(league)
    if not c or c.ops == 0:
        return ops
    return ops / c.ops


def _league_adj_era(era: float, league: str, coeffs: Dict[str, LeagueCoeffs]) -> float:
    c = coeffs.get(league)
    if not c or c.era == 0:
        return era
    return era / c.era


def _rookie_blend(raw: float, league_mean: float, g: int, ctx: ScoringContext, is_rookie: bool) -> float:
    if is_rookie or g < ctx.rookie_g_threshold:
        w = ctx.rookie_shrink
        return raw * (1 - w) + league_mean * w
    return raw


def build_scoring_context(teams: List[Team], coeffs: Dict[str, LeagueCoeffs]) -> ScoringContext:
    """8강 전체 선수 풀에서 Min-Max 범위 산출."""
    ops_vals, hr_vals, k_vals, g_vals = [], [], [], []
    era_inv, k_p, bb_p, ip_p, opp_ops = [], [], [], [], []

    for team in teams:
        for b in team.batters:
            o = _league_adj_ops(b.ops, b.league, coeffs)
            ops_vals.append(o)
            hr_vals.append(float(b.hr))
            k_vals.append(b.k / max(b.g, 1))
            g_vals.append(float(b.g))
        for p in team.pitchers:
            era_inv.append(1.0 / max(_league_adj_era(p.era, p.league, coeffs), 0.5))
            k_p.append(p.k / max(p.ip, 1))
            bb_p.append(p.bb / max(p.ip, 1))
            ip_p.append(p.ip)
            opp_ops.append(p.opp_ops)

    def b(xs: List[float]) -> StatBounds:
        return StatBounds(min(xs), max(xs))

    bounds = {
        "ops": b(ops_vals),
        "hr": b(hr_vals),
        "k_bat": b(k_vals),
        "g": b(g_vals),
        "era_inv": b(era_inv),
        "k_pit": b(k_p),
        "bb_pit": b(bb_p),
        "ip": b(ip_p),
        "opp_ops": b(opp_ops),
    }
    return ScoringContext(bounds=bounds, coeffs=coeffs)


def _bat_score(
    ops: float,
    hr: int,
    k: int,
    g: int,
    ctx: ScoringContext,
    league: str,
    is_rookie: bool,
) -> float:
    c = ctx.coeffs.get(league, LeagueCoeffs())
    ops_m = _mean_ops_league(ctx)
    ops_a = _rookie_blend(_league_adj_ops(ops, league, ctx.coeffs), ops_m, g, ctx, is_rookie)
    hr_a = _rookie_blend(float(hr), _mean_hr(ctx), g, ctx, is_rookie)
    k_rate = k / max(g, 1)
    n_ops = minmax_norm(ops_a, ctx.bounds["ops"].vmin, ctx.bounds["ops"].vmax)
    n_hr = minmax_norm(hr_a, ctx.bounds["hr"].vmin, ctx.bounds["hr"].vmax)
    n_k = minmax_norm(k_rate, ctx.bounds["k_bat"].vmin, ctx.bounds["k_bat"].vmax)
    n_g = minmax_norm(float(g), ctx.bounds["g"].vmin, ctx.bounds["g"].vmax)
    return 0.55 * n_ops + 0.25 * n_hr + 0.10 * (100 - n_k) + 0.10 * n_g


def _mean_ops_league(ctx: ScoringContext) -> float:
    return (ctx.bounds["ops"].vmin + ctx.bounds["ops"].vmax) / 2


def _mean_hr(ctx: ScoringContext) -> float:
    return (ctx.bounds["hr"].vmin + ctx.bounds["hr"].vmax) / 2


def batter_vs_rhp_score(b: Batter, ctx: ScoringContext) -> float:
    ops = b.ops_vs_rhp if b.ops_vs_rhp > 0 else b.ops
    return _bat_score(ops, b.hr, b.k, b.g, ctx, b.league, b.is_rookie)


def batter_vs_lhp_score(b: Batter, ctx: ScoringContext) -> float:
    ops = b.ops_vs_lhp if b.ops_vs_lhp > 0 else b.ops
    return _bat_score(ops, b.hr, b.k, b.g, ctx, b.league, b.is_rookie)


def _pit_score(
    p: Pitcher,
    ctx: ScoringContext,
    vs_hand: str,
) -> float:
    era_a = _league_adj_era(p.era, p.league, ctx.coeffs)
    era_inv = 1.0 / max(era_a, 0.5)
    k_rate = p.k / max(p.ip, 1)
    bb_rate = p.bb / max(p.ip, 1)
    if vs_hand == "R":
        opp = p.opp_ops_vs_rhb if p.opp_ops_vs_rhb > 0 else p.opp_ops
    else:
        opp = p.opp_ops_vs_lhb if p.opp_ops_vs_lhb > 0 else p.opp_ops
    if opp <= 0:
        opp = p.era / 5.0  # 피OPS 없으면 ERA 폴백 (명세)

    n_era = minmax_norm(era_inv, ctx.bounds["era_inv"].vmin, ctx.bounds["era_inv"].vmax)
    n_k = minmax_norm(k_rate, ctx.bounds["k_pit"].vmin, ctx.bounds["k_pit"].vmax)
    n_bb = minmax_norm(bb_rate, ctx.bounds["bb_pit"].vmin, ctx.bounds["bb_pit"].vmax)
    n_opp = minmax_norm(opp, ctx.bounds["opp_ops"].vmin, ctx.bounds["opp_ops"].vmax)
    n_ip = minmax_norm(p.ip, ctx.bounds["ip"].vmin, ctx.bounds["ip"].vmax)

    is_sp = p.gs >= 5
    if is_sp:
        return (
            0.35 * n_era
            + 0.30 * n_k
            + 0.15 * (100 - n_bb)
            + 0.10 * (100 - n_opp)
            + 0.10 * n_ip
        )
    return 0.40 * n_era + 0.35 * n_k + 0.15 * (100 - n_bb) + 0.10 * (100 - n_opp)


def pitcher_vs_rhb_score(p: Pitcher, ctx: ScoringContext) -> float:
    return _pit_score(p, ctx, "R")


def pitcher_vs_lhb_score(p: Pitcher, ctx: ScoringContext) -> float:
    return _pit_score(p, ctx, "L")
