"""STEP 6 — 정확도 검증 & 백테스트."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from models import MatchContext, Team
from poisson_match import predict_match
from scoring import ScoringContext


@dataclass
class BacktestResult:
    n_games: int
    winner_hits: int
    winner_accuracy: float
    brier_score: float
    details: List[Dict[str, Any]]


@dataclass
class Validation2026:
    predicted_champion: Optional[str]
    actual_champion: Optional[str]
    champion_match: Optional[bool]
    qf_hit_rate: Optional[float]
    notes: str


def run_backtest(
    games: List[Dict[str, Any]],
    teams: Dict[str, Team],
    ctx: ScoringContext,
) -> BacktestResult:
    hits = 0
    brier = 0.0
    details = []
    for g in games:
        ta = teams[g["team_a"]]
        tb = teams[g["team_b"]]
        mc = MatchContext(
            home_team_id=g.get("home", g["team_a"]),
            away_team_id=g["team_b"],
        )
        pred = predict_match(
            ta,
            tb,
            ctx,
            mc,
            weather_mult=float(g.get("weather_mult", 1.0)),
        )
        pick = g["team_a"] if pred.win_prob_a >= pred.win_prob_b else g["team_b"]
        actual = g["actual_winner"]
        hit = pick == actual
        if hit:
            hits += 1
        p = pred.win_prob_a if actual == g["team_a"] else pred.win_prob_b
        brier += (1 - p) ** 2 if hit else p**2
        details.append(
            {
                "id": g["id"],
                "match": f"{g['team_a']} vs {g['team_b']}",
                "predicted": pick,
                "actual": actual,
                "hit": hit,
                "p_pick": max(pred.win_prob_a, pred.win_prob_b),
            }
        )
    n = len(games) or 1
    return BacktestResult(
        n_games=len(games),
        winner_hits=hits,
        winner_accuracy=hits / n,
        brier_score=brier / n,
        details=details,
    )


def validate_2026(
    champion_probs: Dict[str, float],
    actual: Dict[str, Any],
    qf_predictions: List[Dict[str, str]],
) -> Validation2026:
    pred_champ = max(champion_probs, key=champion_probs.get) if champion_probs else None
    actual_champ = actual.get("champion")
    champ_ok = None
    if actual_champ and pred_champ:
        champ_ok = pred_champ == actual_champ

    qf_hits = 0
    qf_total = 0
    actual_qf = actual.get("qf_results") or []
    for i, row in enumerate(actual_qf):
        if i < len(qf_predictions):
            qf_total += 1
            if qf_predictions[i].get("predicted") == row.get("winner"):
                qf_hits += 1
    qf_rate = qf_hits / qf_total if qf_total else None

    return Validation2026(
        predicted_champion=pred_champ,
        actual_champion=actual_champ,
        champion_match=champ_ok,
        qf_hit_rate=qf_rate,
        notes="actual_2026 데이터 미입력 시 champion/qf 항목은 None",
    )
