"""STEP 4 — 포아송 기반 경기 승부 예측."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Tuple

from models import MatchContext, Team
from scoring import ScoringContext
from team_power import TeamMatchPower, compute_team_match_power, predominant_bat_hand


def poisson_pmf(k: int, lam: float) -> float:
    if lam <= 0:
        return 1.0 if k == 0 else 0.0
    return math.exp(-lam) * (lam**k) / math.factorial(k)


@dataclass
class MatchPrediction:
    lambda_a: float
    lambda_b: float
    win_prob_a: float
    win_prob_b: float
    tie_prob: float
    expected_runs_a: float
    expected_runs_b: float
    score_matrix_top: List[Tuple[int, int, float]]
    power_a: TeamMatchPower
    power_b: TeamMatchPower
    weather_note: str
    home_away_note: str
    defense_note: str


def _starter_for_round(team: Team, round_idx: int, injured: List[str]) -> Tuple[str, str]:
    from team_power import _filter_injured_pitchers

    pitchers = _filter_injured_pitchers(team, injured)
    sps = [p for p in pitchers if p.gs >= 5] or sorted(pitchers, key=lambda x: x.ip, reverse=True)[:3]
    sps = sorted(sps, key=lambda x: x.ip, reverse=True)[:3]
    sp = sps[round_idx % len(sps)]
    return sp.name, sp.throws


def _lambda_from_matchup(
    offense_lineup: float,
    opponent_pitch: float,
    opponent_defense: float,
    run_mult: float,
) -> float:
    base = 4.15
    off = max(offense_lineup, 15.0) / 50.0
    pit = max(opponent_pitch, 15.0) / 50.0
    lam = base * (off**1.15) * (1.0 / pit) ** 0.85
    lam *= 1.0 - (opponent_defense - 50.0) / 280.0
    lam *= run_mult
    return max(0.8, min(12.0, lam))


def poisson_win_probs(lam_a: float, lam_b: float, max_runs: int = 14) -> Tuple[float, float, float]:
    p_a = p_b = p_tie = 0.0
    for a in range(max_runs + 1):
        pa = poisson_pmf(a, lam_a)
        for b in range(max_runs + 1):
            pb = poisson_pmf(b, lam_b)
            p = pa * pb
            if a > b:
                p_a += p
            elif b > a:
                p_b += p
            else:
                p_tie += p
    p_a += p_tie * 0.5
    p_b += p_tie * 0.5
    tail = max(0.0, 1.0 - p_a - p_b)
    if tail > 0:
        p_a += tail * 0.5
        p_b += tail * 0.5
    return p_a, p_b, p_tie


def predict_match(
    team_a: Team,
    team_b: Team,
    ctx: ScoringContext,
    match_ctx: MatchContext,
    round_idx: int = 0,
    weather_mult: float = 1.0,
    weather_note: str = "",
) -> MatchPrediction:
    inj_bat = match_ctx.injured_batter_names or []
    inj_pit = match_ctx.injured_pitcher_names or []

    _, throws_b = _starter_for_round(team_b, round_idx, inj_pit)
    pa = compute_team_match_power(
        team_a,
        throws_b,
        "R",
        ctx,
        round_idx,
        inj_bat,
        inj_pit,
    )
    _, throws_a = _starter_for_round(team_a, round_idx, inj_pit)
    pb = compute_team_match_power(
        team_b,
        throws_a,
        predominant_bat_hand(pa.lineup, team_a),
        ctx,
        round_idx,
        inj_bat,
        inj_pit,
    )
    pa = compute_team_match_power(
        team_a,
        throws_b,
        predominant_bat_hand(pb.lineup, team_b),
        ctx,
        round_idx,
        inj_bat,
        inj_pit,
    )

    home = match_ctx.home_team_id
    mult_a = weather_mult
    mult_b = weather_mult
    ha_note = "중립 구장 (홈/어웨이 미적용)"
    if home not in (team_a.id, team_b.id):
        pass
    elif home == team_a.id:
        mult_a *= team_a.home_run_factor
        mult_b *= team_b.away_run_factor
        ha_note = f"홈 {team_a.name_ko} / 어웨이 {team_b.name_ko}"
    elif home == team_b.id:
        mult_b *= team_b.home_run_factor
        mult_a *= team_a.away_run_factor
        ha_note = f"홈 {team_b.name_ko} / 어웨이 {team_a.name_ko}"

    lam_a = _lambda_from_matchup(pa.lineup_strength, pb.pitching_strength, pb.defense_index, mult_a)
    lam_b = _lambda_from_matchup(pb.lineup_strength, pa.pitching_strength, pa.defense_index, mult_b)

    wp_a, wp_b, p_tie = poisson_win_probs(lam_a, lam_b)
    top_scores = []
    for a in range(9):
        for b in range(9):
            pr = poisson_pmf(a, lam_a) * poisson_pmf(b, lam_b)
            if pr > 0.008:
                top_scores.append((a, b, pr))
    top_scores.sort(key=lambda x: x[2], reverse=True)

    def_note = f"수비 A={pa.defense_index:.1f} B={pb.defense_index:.1f} (DER→λ 억제)"

    return MatchPrediction(
        lambda_a=lam_a,
        lambda_b=lam_b,
        win_prob_a=wp_a,
        win_prob_b=wp_b,
        tie_prob=p_tie,
        expected_runs_a=lam_a,
        expected_runs_b=lam_b,
        score_matrix_top=top_scores[:8],
        power_a=pa,
        power_b=pb,
        weather_note=weather_note or f"날씨 배율 {weather_mult:.3f}",
        home_away_note=ha_note,
        defense_note=def_note,
    )
