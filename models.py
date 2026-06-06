"""데이터 모델 — 2026 WBC 8강 명세 필드."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Literal, Optional

League = Literal["MLB", "NPB", "KBO", "CPBL", "OTHER"]
Hand = Literal["L", "R", "S"]


@dataclass
class Batter:
    name: str
    league: League
    ops: float
    ops_vs_rhp: float
    ops_vs_lhp: float
    hr: int
    k: int
    g: int
    bats: Hand
    # 선택: 수비·주루 확장용
    errors: int = 0
    sb: int = 0
    is_rookie: bool = False


@dataclass
class Pitcher:
    name: str
    league: League
    era: float
    k: int
    bb: int
    h: int
    ip: float
    gs: int
    opp_ops: float
    opp_ops_vs_rhb: float
    opp_ops_vs_lhb: float
    throws: Hand
    is_rookie: bool = False


@dataclass
class Team:
    id: str
    name_ko: str
    name_en: str
    batters: List[Batter]
    pitchers: List[Pitcher]
    # 홈/어웨이 (최근 3년 득점 기대 보정, koreabaseball 홈/어웨이 개념 반영)
    home_run_factor: float = 1.04
    away_run_factor: float = 0.97
    # 수비 지표 (팀 DER 근사: 낮을수록 좋음)
    team_der: float = 0.700
    venue_lat: float = 33.0
    venue_lon: float = -117.0
    avg_salary_usd_m: Optional[float] = None


@dataclass
class MatchContext:
    home_team_id: str
    away_team_id: str
    weather_run_mult: float = 1.0
    injured_batter_names: List[str] = field(default_factory=list)
    injured_pitcher_names: List[str] = field(default_factory=list)
