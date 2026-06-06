"""STEP 5 — 8강 몬테카를로 시뮬레이션."""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Dict, List, Tuple

from models import MatchContext, Team
from poisson_match import predict_match
from rosters import BRACKET_QF
from scoring import ScoringContext


@dataclass
class TournamentSimResult:
    n_sims: int
    champion_probs: Dict[str, float]
    finalist_probs: Dict[str, float]
    semifinal_probs: Dict[str, float]
    qf_win_counts: Dict[str, int]


def _bernoulli(p: float, rng: random.Random) -> bool:
    return rng.random() < p


def simulate_bracket_once(
    teams: Dict[str, Team],
    ctx: ScoringContext,
    weather_mult: float,
    home_qf: str,
    rng: random.Random,
) -> Tuple[str, Dict[str, int]]:
    """한 번의 토너먼트. 반환: 우승팀 id, 라운드별 진출 기록 depth."""
    round_idx = 0
    qf_winners = []
    for a_id, b_id in BRACKET_QF:
        ta, tb = teams[a_id], teams[b_id]
        mc = MatchContext(home_team_id=home_qf, away_team_id=b_id)
        pred = predict_match(ta, tb, ctx, mc, round_idx, weather_mult)
        qf_winners.append(a_id if _bernoulli(pred.win_prob_a, rng) else b_id)
    round_idx = 1
    sf_winners = []
    for i in range(0, 4, 2):
        ta, tb = teams[qf_winners[i]], teams[qf_winners[i + 1]]
        mc = MatchContext(home_team_id=ta.id, away_team_id=tb.id)
        pred = predict_match(ta, tb, ctx, mc, round_idx, weather_mult)
        sf_winners.append(ta.id if _bernoulli(pred.win_prob_a, rng) else tb.id)
    round_idx = 2
    ta, tb = teams[sf_winners[0]], teams[sf_winners[1]]
    mc = MatchContext(home_team_id=ta.id, away_team_id=tb.id)
    pred = predict_match(ta, tb, ctx, mc, round_idx, weather_mult)
    champ = ta.id if _bernoulli(pred.win_prob_a, rng) else tb.id
    depth = {champ: 3}
    for t in sf_winners:
        depth[t] = max(depth.get(t, 0), 2)
    for t in qf_winners:
        depth[t] = max(depth.get(t, 0), 1)
    return champ, depth


def run_monte_carlo(
    teams: Dict[str, Team],
    ctx: ScoringContext,
    n_sims: int = 10_000,
    weather_mult: float = 1.0,
    home_qf: str = "usa",
    seed: int = 42,
) -> TournamentSimResult:
    rng = random.Random(seed)
    champ_c: Dict[str, int] = {}
    final_c: Dict[str, int] = {}
    semi_c: Dict[str, int] = {}
    qf_c: Dict[str, int] = {}

    for _ in range(n_sims):
        champ, depth = simulate_bracket_once(teams, ctx, weather_mult, home_qf, rng)
        champ_c[champ] = champ_c.get(champ, 0) + 1
        for tid, d in depth.items():
            if d >= 1:
                qf_c[tid] = qf_c.get(tid, 0) + 1
            if d >= 2:
                semi_c[tid] = semi_c.get(tid, 0) + 1
            if d >= 3:
                final_c[tid] = final_c.get(tid, 0) + 1

    def norm(d: Dict[str, int]) -> Dict[str, float]:
        s = sum(d.values()) or 1
        return {k: v / s for k, v in d.items()}

    return TournamentSimResult(
        n_sims=n_sims,
        champion_probs=norm(champ_c),
        finalist_probs=norm(final_c),
        semifinal_probs=norm(semi_c),
        qf_win_counts=qf_c,
    )
