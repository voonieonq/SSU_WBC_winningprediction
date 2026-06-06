"""모델 파이프라인 초기화."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

from league_coeff import LeagueCoeffs, compute_league_coefficients
from rosters import TEAM_BY_ID, TEAMS_QF8
from scoring import ScoringContext, build_scoring_context


@dataclass
class ModelBundle:
    teams: Dict[str, object]
    coeffs: Dict[str, LeagueCoeffs]
    scoring_ctx: ScoringContext


def build_model() -> ModelBundle:
    coeffs = compute_league_coefficients(TEAMS_QF8)
    ctx = build_scoring_context(TEAMS_QF8, coeffs)
    return ModelBundle(teams=TEAM_BY_ID, coeffs=coeffs, scoring_ctx=ctx)
