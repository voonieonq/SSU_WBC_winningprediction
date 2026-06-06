"""
8강 샘플 로스터 — 명세 필드 구조.
실제 배포 시: koreabaseball.com / statiz / mlb.com / kbostuff 등에서 CSV 로드로 교체.
"""

from __future__ import annotations

from typing import Dict, List

from models import Batter, Pitcher, Team


def _b(
    name,
    league,
    ops,
    ops_r,
    ops_l,
    hr,
    k,
    g,
    bats,
    rookie=False,
):
    return Batter(
        name=name,
        league=league,
        ops=ops,
        ops_vs_rhp=ops_r,
        ops_vs_lhp=ops_l,
        hr=hr,
        k=k,
        g=g,
        bats=bats,
        is_rookie=rookie,
    )


def _p(
    name,
    league,
    era,
    k,
    bb,
    h,
    ip,
    gs,
    opp_ops,
    opp_r,
    opp_l,
    throws,
    rookie=False,
):
    return Pitcher(
        name=name,
        league=league,
        era=era,
        k=k,
        bb=bb,
        h=h,
        ip=ip,
        gs=gs,
        opp_ops=opp_ops,
        opp_ops_vs_rhb=opp_r,
        opp_ops_vs_lhb=opp_l,
        throws=throws,
        is_rookie=rookie,
    )


def _kor_batters() -> List[Batter]:
    return [
        _b("김하성", "MLB", 0.82, 0.84, 0.78, 18, 85, 140, "R"),
        _b("이정후", "MLB", 0.88, 0.90, 0.84, 15, 72, 130, "L"),
        _b("김현수", "KBO", 0.92, 0.88, 0.95, 28, 95, 135, "L"),
        _b("박건우", "KBO", 0.84, 0.86, 0.80, 12, 78, 128, "R"),
        _b("강백호", "KBO", 0.80, 0.82, 0.76, 24, 110, 125, "L"),
        _b("양의지", "KBO", 0.78, 0.80, 0.74, 16, 88, 120, "R"),
        _b("이주형", "KBO", 0.86, 0.84, 0.88, 20, 92, 132, "L"),
        _b("최정", "KBO", 0.88, 0.90, 0.85, 32, 105, 138, "R"),
        _b("신인타자", "KBO", 0.72, 0.70, 0.74, 8, 45, 18, "R", rookie=True),
    ]


def _kor_pitchers() -> List[Pitcher]:
    return [
        _p("구창모", "KBO", 2.45, 165, 42, 118, 168, 28, 0.62, 0.64, 0.60, "L"),
        _p("원태인", "KBO", 3.10, 142, 48, 155, 175, 30, 0.68, 0.70, 0.66, "R"),
        _p("이의리", "KBO", 3.45, 128, 52, 148, 142, 26, 0.72, 0.74, 0.70, "L"),
        _p("정해영", "KBO", 3.20, 88, 28, 52, 58, 0, 0.65, 0.66, 0.64, "R"),
        _p("고영표", "KBO", 2.90, 72, 22, 48, 62, 0, 0.63, 0.64, 0.62, "L"),
        _p("김재윤", "KBO", 3.05, 68, 24, 50, 55, 0, 0.66, 0.67, 0.65, "R"),
        _p("한현희", "KBO", 3.40, 55, 20, 45, 48, 0, 0.70, 0.71, 0.69, "R"),
        _p("홍건희", "KBO", 3.15, 62, 22, 48, 52, 0, 0.67, 0.68, 0.66, "R"),
    ]


def _usa_batters() -> List[Batter]:
    return [
        _b("M. Betts", "MLB", 0.96, 0.98, 0.92, 26, 75, 145, "R"),
        _b("M. Trout", "MLB", 0.94, 0.96, 0.90, 32, 120, 140, "R"),
        _b("C. Carroll", "MLB", 0.89, 0.91, 0.85, 22, 95, 135, "L"),
        _b("P. Goldschmidt", "MLB", 0.92, 0.94, 0.88, 28, 105, 138, "R"),
        _b("K. Tucker", "MLB", 0.90, 0.92, 0.86, 24, 88, 132, "L"),
        _b("T. Turner", "MLB", 0.88, 0.90, 0.84, 18, 82, 140, "R"),
        _b("W. Smith", "MLB", 0.82, 0.84, 0.78, 18, 92, 125, "R"),
        _b("B. Witt Jr.", "MLB", 0.81, 0.83, 0.77, 15, 85, 130, "R"),
        _b("R. Acuña Jr.", "MLB", 0.99, 1.02, 0.94, 38, 98, 128, "R"),
    ]


def _usa_pitchers() -> List[Pitcher]:
    return [
        _p("G. Cole", "MLB", 2.85, 195, 48, 145, 185, 32, 0.58, 0.60, 0.56, "R"),
        _p("L. Snell", "MLB", 3.05, 178, 62, 128, 162, 28, 0.60, 0.62, 0.58, "L"),
        _p("M. Keller", "MLB", 3.40, 155, 52, 165, 178, 30, 0.65, 0.67, 0.63, "R"),
        _p("J. Hader", "MLB", 2.55, 95, 28, 42, 58, 0, 0.55, 0.56, 0.54, "L"),
        _p("E. Clase", "MLB", 2.70, 88, 18, 48, 62, 0, 0.57, 0.58, 0.56, "R"),
        _p("R. Pressly", "MLB", 3.00, 72, 22, 45, 55, 0, 0.62, 0.63, 0.61, "R"),
        _p("D. Bednar", "MLB", 3.15, 68, 24, 48, 52, 0, 0.64, 0.65, 0.63, "R"),
        _p("C. Kimbrel", "MLB", 3.35, 75, 28, 50, 48, 0, 0.66, 0.67, 0.65, "R"),
    ]


def _jpn_batters() -> List[Batter]:
    return [
        _b("오타니 쇼헤이", "MLB", 1.02, 1.05, 0.96, 42, 115, 142, "L"),
        _b("무라카미 무네타카", "NPB", 0.96, 0.98, 0.92, 38, 125, 140, "L"),
        _b("야마다 데쓰토", "NPB", 0.88, 0.90, 0.84, 22, 95, 135, "S"),
        _b("스즈키 세이야", "MLB", 0.86, 0.88, 0.82, 24, 88, 130, "L"),
        _b("무라카타 구니요시", "NPB", 0.84, 0.86, 0.80, 28, 102, 128, "R"),
        _b("오카모토 가이토", "NPB", 0.79, 0.81, 0.75, 20, 98, 125, "R"),
        _b("사카모토 하야토", "NPB", 0.74, 0.76, 0.70, 14, 88, 120, "R"),
        _b("노무라 유스케", "NPB", 0.68, 0.70, 0.64, 8, 75, 115, "R"),
        _b("무라카미 다이시", "NPB", 0.88, 0.90, 0.84, 26, 108, 132, "L"),
    ]


def _jpn_pitchers() -> List[Pitcher]:
    return [
        _p("야마모토 요시노부", "NPB", 1.75, 188, 32, 118, 164, 24, 0.52, 0.54, 0.50, "R"),
        _p("이마나가 쇼타", "NPB", 2.35, 165, 38, 125, 155, 26, 0.58, 0.60, 0.56, "L"),
        _p("다르빗슈 유", "MLB", 3.20, 158, 45, 155, 172, 29, 0.64, 0.66, 0.62, "R"),
        _p("다케다 나오", "NPB", 2.65, 82, 24, 48, 58, 0, 0.60, 0.61, 0.59, "R"),
        _p("스가 노부아키", "NPB", 2.50, 78, 22, 45, 52, 0, 0.58, 0.59, 0.57, "R"),
        _p("히라노 요시히사", "NPB", 2.90, 72, 22, 48, 52, 0, 0.62, 0.63, 0.61, "R"),
        _p("스에키 고헤이", "NPB", 3.10, 68, 24, 50, 48, 0, 0.65, 0.66, 0.64, "L"),
        _p("우에하라 고지", "NPB", 3.25, 65, 22, 48, 45, 0, 0.66, 0.67, 0.65, "R"),
    ]


def _dom_batters() -> List[Batter]:
    return [
        _b("J. Soto", "MLB", 0.98, 1.00, 0.94, 32, 88, 140, "L"),
        _b("R. Acuña Jr.", "MLB", 0.99, 1.02, 0.94, 38, 98, 128, "R"),
        _b("V. Guerrero Jr.", "MLB", 0.96, 0.98, 0.92, 35, 95, 138, "R"),
        _b("F. Lindor", "MLB", 0.90, 0.92, 0.86, 28, 82, 135, "S"),
        _b("K. Marte", "MLB", 0.88, 0.90, 0.84, 26, 88, 132, "S"),
        _b("J. Ramírez", "MLB", 0.90, 0.92, 0.86, 30, 85, 138, "S"),
        _b("G. Torres", "MLB", 0.82, 0.84, 0.78, 22, 95, 128, "S"),
        _b("W. Contreras", "MLB", 0.80, 0.82, 0.76, 18, 92, 125, "R"),
        _b("J. Chisholm", "MLB", 0.76, 0.78, 0.72, 20, 105, 120, "L"),
    ]


def _dom_pitchers() -> List[Pitcher]:
    return [
        _p("S. Alcantara", "MLB", 2.65, 198, 52, 165, 198, 33, 0.58, 0.60, 0.56, "R"),
        _p("C. Rodón", "MLB", 3.20, 175, 58, 142, 165, 28, 0.62, 0.64, 0.60, "L"),
        _p("L. Severino", "MLB", 3.45, 148, 48, 155, 152, 27, 0.66, 0.68, 0.64, "R"),
        _p("E. Díaz", "MLB", 2.45, 95, 28, 42, 58, 0, 0.55, 0.56, 0.54, "R"),
        _p("D. Bard", "MLB", 3.00, 78, 26, 48, 54, 0, 0.62, 0.63, 0.61, "R"),
        _p("R. Iglesias", "MLB", 2.85, 72, 22, 45, 56, 0, 0.60, 0.61, 0.59, "R"),
        _p("J. Jiménez", "MLB", 3.10, 68, 24, 48, 50, 0, 0.64, 0.65, 0.63, "R"),
        _p("G. Gallegos", "MLB", 3.20, 65, 22, 45, 52, 0, 0.65, 0.66, 0.64, "R"),
    ]


def _generic_team(
    tid: str,
    name_ko: str,
    name_en: str,
    tier: float,
    league_b: str,
    league_p: str,
    home_f: float,
    away_f: float,
    der: float,
    lat: float,
    lon: float,
) -> Team:
    """중위권 팀 — tier 0.75~0.88 스케일."""
    batters = [
        _b(f"{name_ko} 1번", league_b, 0.78 * tier, 0.80 * tier, 0.74 * tier, 18, 95, 130, "R"),
        _b(f"{name_ko} 2번", league_b, 0.82 * tier, 0.84 * tier, 0.78 * tier, 22, 88, 132, "L"),
        _b(f"{name_ko} 3번", league_b, 0.80 * tier, 0.82 * tier, 0.76 * tier, 20, 92, 128, "R"),
        _b(f"{name_ko} 4번", league_b, 0.76 * tier, 0.78 * tier, 0.72 * tier, 16, 100, 125, "S"),
        _b(f"{name_ko} 5번", league_b, 0.74 * tier, 0.76 * tier, 0.70 * tier, 14, 105, 122, "R"),
        _b(f"{name_ko} 6번", league_b, 0.72 * tier, 0.74 * tier, 0.68 * tier, 12, 98, 120, "L"),
        _b(f"{name_ko} 7번", league_b, 0.70 * tier, 0.72 * tier, 0.66 * tier, 10, 102, 118, "R"),
        _b(f"{name_ko} 8번", league_b, 0.68 * tier, 0.70 * tier, 0.64 * tier, 8, 95, 115, "R"),
        _b(f"{name_ko} 신인", league_b, 0.65 * tier, 0.63 * tier, 0.67 * tier, 5, 55, 22, "R", rookie=True),
    ]
    era = 3.2 / tier
    pitchers = [
        _p(f"{name_ko} SP1", league_p, era, 150, 45, 145, 165, 28, 0.65 / tier, 0.67 / tier, 0.63 / tier, "R"),
        _p(f"{name_ko} SP2", league_p, era + 0.3, 140, 48, 150, 155, 26, 0.68 / tier, 0.70 / tier, 0.66 / tier, "L"),
        _p(f"{name_ko} SP3", league_p, era + 0.5, 130, 50, 155, 145, 24, 0.70 / tier, 0.72 / tier, 0.68 / tier, "R"),
        _p(f"{name_ko} CL", league_p, era + 0.2, 75, 22, 45, 55, 0, 0.66 / tier, 0.67 / tier, 0.65 / tier, "R"),
        _p(f"{name_ko} SU1", league_p, era + 0.4, 68, 24, 48, 50, 0, 0.68 / tier, 0.69 / tier, 0.67 / tier, "R"),
        _p(f"{name_ko} SU2", league_p, era + 0.35, 65, 22, 45, 48, 0, 0.69 / tier, 0.70 / tier, 0.68 / tier, "L"),
        _p(f"{name_ko} SU3", league_p, era + 0.45, 62, 25, 48, 45, 0, 0.70 / tier, 0.71 / tier, 0.69 / tier, "R"),
        _p(f"{name_ko} SU4", league_p, era + 0.5, 58, 26, 50, 42, 0, 0.72 / tier, 0.73 / tier, 0.71 / tier, "R"),
    ]
    return Team(
        id=tid,
        name_ko=name_ko,
        name_en=name_en,
        batters=batters,
        pitchers=pitchers,
        home_run_factor=home_f,
        away_run_factor=away_f,
        team_der=der,
        venue_lat=lat,
        venue_lon=lon,
    )


TEAMS_QF8: List[Team] = [
    Team(
        id="kor",
        name_ko="대한민국",
        name_en="Korea",
        batters=_kor_batters(),
        pitchers=_kor_pitchers(),
        home_run_factor=1.03,
        away_run_factor=0.96,
        team_der=0.705,
        venue_lat=37.57,
        venue_lon=127.00,
        avg_salary_usd_m=1.2,
    ),
    Team(
        id="usa",
        name_ko="미국",
        name_en="USA",
        batters=_usa_batters(),
        pitchers=_usa_pitchers(),
        home_run_factor=1.05,
        away_run_factor=0.98,
        team_der=0.698,
        venue_lat=33.89,
        venue_lon=-117.88,
        avg_salary_usd_m=8.5,
    ),
    Team(
        id="jpn",
        name_ko="일본",
        name_en="Japan",
        batters=_jpn_batters(),
        pitchers=_jpn_pitchers(),
        home_run_factor=1.02,
        away_run_factor=0.97,
        team_der=0.710,
        venue_lat=35.68,
        venue_lon=139.69,
        avg_salary_usd_m=2.8,
    ),
    Team(
        id="dom",
        name_ko="도미니카",
        name_en="Dominican Rep.",
        batters=_dom_batters(),
        pitchers=_dom_pitchers(),
        home_run_factor=1.04,
        away_run_factor=0.97,
        team_der=0.702,
        venue_lat=18.48,
        venue_lon=-69.93,
        avg_salary_usd_m=5.0,
    ),
    _generic_team("mex", "멕시코", "Mexico", 0.82, "MLB", "MLB", 1.02, 0.97, 0.708, 19.43, -99.13),
    _generic_team("ven", "베네수엘라", "Venezuela", 0.84, "MLB", "MLB", 1.03, 0.96, 0.706, 10.48, -66.88),
    _generic_team("pur", "푸에르토리코", "Puerto Rico", 0.83, "MLB", "MLB", 1.04, 0.96, 0.707, 18.47, -66.11),
    _generic_team("aus", "호주", "Australia", 0.72, "OTHER", "OTHER", 1.01, 0.95, 0.715, -33.87, 151.21),
]

TEAM_BY_ID: Dict[str, Team] = {t.id: t for t in TEAMS_QF8}

# 8강 브래킷 (시드 예시)
BRACKET_QF = [
    ("usa", "aus"),
    ("dom", "kor"),
    ("jpn", "pur"),
    ("ven", "mex"),
]
