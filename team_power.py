"""STEP 3 — 동적 라인업 & 팀 전력."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

from models import Batter, MatchContext, Pitcher, Team
from scoring import ScoringContext, batter_vs_lhp_score, batter_vs_rhp_score
from scoring import pitcher_vs_lhb_score, pitcher_vs_rhb_score


@dataclass
class LineupSlot:
    order: int
    name: str
    score: float


@dataclass
class TeamMatchPower:
    lineup: List[LineupSlot]
    lineup_strength: float
    starter_name: str
    starter_score: float
    bullpen_avg: float
    pitching_strength: float
    overall: float
    defense_index: float


def _filter_injured_batters(team: Team, injured: List[str]) -> List[Batter]:
    names = set(injured or [])
    return [b for b in team.batters if b.name not in names]


def _filter_injured_pitchers(team: Team, injured: List[str]) -> List[Pitcher]:
    names = set(injured or [])
    return [p for p in team.pitchers if p.name not in names]


def build_lineup(
    team: Team,
    opponent_starter_throws: str,
    ctx: ScoringContext,
    injured: Optional[List[str]] = None,
) -> List[LineupSlot]:
    """상대 선발 손에 맞는 점수로 9명 정렬 → 타순."""
    batters = _filter_injured_batters(team, injured or [])
    if opponent_starter_throws == "L":
        scored = [(b, batter_vs_lhp_score(b, ctx)) for b in batters]
    else:
        scored = [(b, batter_vs_rhp_score(b, ctx)) for b in batters]
    scored.sort(key=lambda x: x[1], reverse=True)
    top9 = scored[:9]
    return [LineupSlot(i + 1, b.name, s) for i, (b, s) in enumerate(top9)]


def _lineup_strength(slots: List[LineupSlot]) -> float:
    if not slots:
        return 0.0
    top = [s.score for s in slots if s.order <= 3]
    mid = [s.score for s in slots if 4 <= s.order <= 6]
    bot = [s.score for s in slots if s.order >= 7]
    if len(top) < 3 or len(mid) < 3 or len(bot) < 3:
        all_s = [s.score for s in slots]
        return sum(all_s) / len(all_s)
    return 0.40 * (sum(top) / 3) + 0.35 * (sum(mid) / 3) + 0.25 * (sum(bot) / 3)


def _pick_rotation(
    pitchers: List[Pitcher],
    round_idx: int,
    opponent_bats_hand: str,
    ctx: ScoringContext,
) -> Tuple[Pitcher, float, List[float]]:
    """선발 3 로테이션 + 불펜 상위 5."""
    sps = [p for p in pitchers if p.gs >= 5]
    rps = [p for p in pitchers if p.gs < 5]
    if not sps:
        sps = sorted(pitchers, key=lambda x: x.ip, reverse=True)[:3]
    sps = sorted(sps, key=lambda x: x.ip, reverse=True)[:3]
    starter = sps[round_idx % len(sps)]
    if opponent_bats_hand == "L":
        st_score = pitcher_vs_lhb_score(starter, ctx)
        rp_scores = [pitcher_vs_lhb_score(p, ctx) for p in rps[:5]]
    else:
        st_score = pitcher_vs_rhb_score(starter, ctx)
        rp_scores = [pitcher_vs_rhb_score(p, ctx) for p in rps[:5]]
    if not rp_scores:
        rp_scores = [st_score * 0.9]
    return starter, st_score, rp_scores[:5]


def defense_index(team: Team) -> float:
    """수비 → DER 근사 (높을수록 좋음, 0~100)."""
    # MLB 평균 DER ~0.700; 팀 DER가 높으면 수비 우수
    der = team.team_der
    return max(0.0, min(100.0, (der - 0.660) / (0.740 - 0.660) * 100.0))


def compute_team_match_power(
    team: Team,
    opponent_starter_throws: str,
    opponent_bats_predominantly: str,
    ctx: ScoringContext,
    round_idx: int = 0,
    injured_batters: Optional[List[str]] = None,
    injured_pitchers: Optional[List[str]] = None,
) -> TeamMatchPower:
    pitchers = _filter_injured_pitchers(team, injured_pitchers or [])
    lineup = build_lineup(team, opponent_starter_throws, ctx, injured_batters)
    ls = _lineup_strength(lineup)
    starter, st_sc, rp_sc = _pick_rotation(pitchers, round_idx, opponent_bats_predominantly, ctx)
    bp_avg = sum(rp_sc) / len(rp_sc)
    pitch = 0.60 * st_sc + 0.40 * bp_avg
    overall = 0.55 * ls + 0.45 * pitch
    return TeamMatchPower(
        lineup=lineup,
        lineup_strength=ls,
        starter_name=starter.name,
        starter_score=st_sc,
        bullpen_avg=bp_avg,
        pitching_strength=pitch,
        overall=overall,
        defense_index=defense_index(team),
    )


def predominant_bat_hand(lineup: List[LineupSlot], team: Team) -> str:
    """라인업 좌타 비중 → 투수 vs 손 판단용."""
    names = {s.name for s in lineup}
    left = sum(1 for b in team.batters if b.name in names and b.bats in ("L", "S"))
    return "L" if left >= 5 else "R"
