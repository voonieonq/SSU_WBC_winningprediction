"""WBC 4강 예측 엔진 (JS engine.js 포팅)."""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

from data import LEAGUE_COEFF, TEAMS


def clamp(x: float, a: float, b: float) -> float:
    return max(a, min(b, x))


def norm_linear(x: float, lo: float, hi: float) -> float:
    if hi <= lo:
        return 50.0
    return clamp((x - lo) / (hi - lo) * 100.0, 0.0, 100.0)


def get_coeff(league_key: str) -> Dict[str, float]:
    return LEAGUE_COEFF.get(league_key, LEAGUE_COEFF["OTHER"])


def batter_raw100(b: Dict[str, Any]) -> float:
    L = get_coeff(b["league"])
    ops_blend = b["opsSeason"] * 0.7 + b["opsRecent15"] * 0.3
    ops_adj = ops_blend * L["ops"]
    risp_adj = b["rispOps"] * L["ops"]
    n_ops = norm_linear(ops_adj, 0.58, 1.05)
    n_hr = norm_linear(float(b["hr"]), 6.0, 45.0)
    n_risp = norm_linear(risp_adj, 0.32, 0.42)
    k_avoid = norm_linear(1.0 - b["kRate"], 0.62, 0.88)
    n_bb = norm_linear(b["bbRate"], 0.045, 0.16)
    w_sum = 0.3 + 0.15 + 0.2 + 0.15 + 0.1
    s = (
        0.3 * n_ops + 0.15 * n_hr + 0.2 * n_risp + 0.15 * k_avoid + 0.1 * n_bb
    ) / w_sum
    s *= L["base"]
    return clamp(s, 0.0, 100.0)


def batter_effective_for_starter(b: Dict[str, Any], opponent_starter_throws: str) -> float:
    L = get_coeff(b["league"])
    vs = b["opsVsLhp"] if opponent_starter_throws == "L" else b["opsVsRhp"]
    blend = b["opsSeason"] * 0.7 + b["opsRecent15"] * 0.3
    mixed = vs * 0.55 + blend * 0.45
    ops_adj = mixed * L["ops"]
    risp_adj = b["rispOps"] * L["ops"] * (0.98 if opponent_starter_throws == "L" else 1.0)
    n_ops = norm_linear(ops_adj, 0.55, 1.05)
    n_hr = norm_linear(float(b["hr"]), 6.0, 45.0)
    n_risp = norm_linear(risp_adj, 0.32, 0.42)
    k_avoid = norm_linear(1.0 - b["kRate"], 0.62, 0.88)
    n_bb = norm_linear(b["bbRate"], 0.045, 0.16)
    w_sum = 0.3 + 0.15 + 0.2 + 0.15 + 0.1
    s = (
        0.3 * n_ops + 0.15 * n_hr + 0.2 * n_risp + 0.15 * k_avoid + 0.1 * n_bb
    ) / w_sum
    s *= L["base"]
    return clamp(s, 0.0, 100.0)


def pitcher_raw100(p: Dict[str, Any]) -> float:
    L = get_coeff(p["league"])
    era = max(p["era"] * L["era"], 0.5)
    k9 = p["k9"] * L["k9"]
    bb9 = max(p["bb9"] * L["bb9"], 0.3)
    whip = max(p["whip"] * L["whip"], 0.5)
    era_inv = norm_linear(1.0 / era, 1.0 / 5.5, 1.0 / 1.5)
    k9_n = norm_linear(k9, 6.5, 13.5)
    bb9_inv = norm_linear(1.0 / bb9, 1.0 / 5.5, 1.0 / 1.2)
    whip_inv = norm_linear(1.0 / whip, 1.0 / 1.55, 1.0 / 0.85)
    role_n = 100.0 if p["role"] == "SP" else 93.0
    s = (
        0.3 * era_inv
        + 0.25 * k9_n
        + 0.2 * bb9_inv
        + 0.15 * whip_inv
        + 0.1 * role_n
    )
    s *= L["base"]
    return clamp(s, 0.0, 100.0)


def team_avg_error_rate(team: Dict[str, Any]) -> float:
    xs = [b["errRate"] for b in team["batters"]]
    return sum(xs) / len(xs)


def defense_pitcher_bonus(team: Dict[str, Any]) -> float:
    e = team_avg_error_rate(team)
    return clamp((0.025 - e) * 120.0, -3.0, 3.0)


def bullpen_fatigue_debuff(rp_list: List[Dict[str, Any]]) -> float:
    d = 0.0
    top = sorted(rp_list, key=lambda x: x["ip"], reverse=True)[:2]
    for p in top:
        n = p.get("pitchesLast3Days", 0) or 0
        if n >= 95:
            d += 4.0
        elif n >= 75:
            d += 2.5
        elif n >= 55:
            d += 1.0
    return clamp(d, 0.0, 6.0)


def team_lineup_score(
    team: Dict[str, Any], opponent_starter_throws: str, scenario: str
) -> float:
    throws = opponent_starter_throws or "R"
    rows = []
    for b in team["batters"]:
        s = batter_effective_for_starter(b, throws)
        if scenario == "missing" and b["order"] == 3:
            s *= 0.72
        rows.append({**b, "_s": s})
    top = [x for x in rows if x["order"] <= 3]
    mid = [x for x in rows if 4 <= x["order"] <= 6]
    bot = [x for x in rows if x["order"] >= 7]

    def avg(arr: List[Dict[str, Any]]) -> float:
        return sum(x["_s"] for x in arr) / len(arr) if arr else 0.0

    return 0.4 * avg(top) + 0.35 * avg(mid) + 0.25 * avg(bot)


def team_pitch_score(team: Dict[str, Any]) -> float:
    sps = [pitcher_raw100(p) for p in team["pitchers"]["sp"]]
    rps = [pitcher_raw100(p) for p in team["pitchers"]["rp"]]
    sp_avg = sum(sps) / len(sps)
    rp_avg = sum(rps) / len(rps)
    fat = bullpen_fatigue_debuff(team["pitchers"]["rp"])
    rp_avg = clamp(rp_avg - fat, 0.0, 100.0)
    raw = 0.6 * sp_avg + 0.4 * rp_avg
    def_b = defense_pitcher_bonus(team)
    return clamp(raw + def_b * 0.35, 0.0, 100.0)


def team_overall(lineup: float, pitch: float) -> float:
    return 0.55 * lineup + 0.45 * pitch


def compute_environment(
    team_a: Dict[str, Any], team_b: Dict[str, Any], weather_boost: float
) -> Dict[str, float]:
    wb = float(weather_boost or 0.0)
    avg_park = (team_a["parkFactor"] + team_b["parkFactor"]) / 2.0
    avg_w = (team_a.get("weatherRunAdj", 0) or 0) + (
        team_b.get("weatherRunAdj", 0) or 0
    )
    avg_w /= 2.0
    common_run_env = 1.0 + (avg_park - 1.0) * 0.88 + wb * 0.62 + avg_w * 0.45
    m_lineup_a = clamp(
        common_run_env + (team_a.get("weatherRunAdj", 0) or 0) * 0.42,
        0.88,
        1.18,
    )
    m_lineup_b = clamp(
        common_run_env + (team_b.get("weatherRunAdj", 0) or 0) * 0.42,
        0.88,
        1.18,
    )
    run_env_index = max(0.0, common_run_env - 1.0)
    m_pitch = clamp(1.0 - run_env_index * 0.14 - wb * 0.09, 0.9, 1.05)
    return {
        "avgPark": avg_park,
        "weatherBoost": wb,
        "commonRunEnv": common_run_env,
        "avgWeatherAdj": avg_w,
        "mLineupA": m_lineup_a,
        "mLineupB": m_lineup_b,
        "mPitch": m_pitch,
    }


def baserun_bonus(team: Dict[str, Any]) -> float:
    sb = sum(b["sb"] for b in team["batters"])
    cs = sum(b["cs"] for b in team["batters"])
    den = sb + cs + 1
    rate = (sb - cs) / den
    return clamp(rate * 0.35, -0.35, 0.35)


def logistic_win_prob(sa: float, sb: float, temperature: float = 11.5) -> float:
    raw = 1.0 / (1.0 + math.exp(-(sa - sb) / temperature))
    return clamp(0.5 + (raw - 0.5) * 0.85, 0.02, 0.98)


def confidence_interval(p: float) -> Dict[str, float]:
    spread = 0.035 + abs(p - 0.5) * 0.12
    return {"lo": clamp(p - spread, 0.01, 0.99), "hi": clamp(p + spread, 0.01, 0.99)}


def expected_scores(
    lineup_a: float,
    pitch_b: float,
    lineup_b: float,
    pitch_a: float,
    team_a: Dict[str, Any],
    team_b: Dict[str, Any],
    park_weather_a: float,
    park_weather_b: float,
) -> Dict[str, float]:
    base = 4.35
    diff_lineup = (lineup_a - lineup_b) / 10.0
    sup_a = (pitch_b - 72.0) / 30.0
    sup_b = (pitch_a - 72.0) / 30.0
    a = (
        base
        + diff_lineup * 0.95
        - sup_a * 0.48
        + baserun_bonus(team_a)
        + park_weather_a
    )
    b = (
        base
        - diff_lineup * 0.95
        - sup_b * 0.48
        + baserun_bonus(team_b)
        + park_weather_b
    )
    pf = (team_a["parkFactor"] + team_b["parkFactor"]) / 2.0 - 1.0
    a += pf * 0.45 + (team_a.get("weatherRunAdj", 0) or 0)
    b += pf * 0.45 + (team_b.get("weatherRunAdj", 0) or 0)
    a = clamp(round(a * 2) / 2, 2.0, 9.0)
    b = clamp(round(b * 2) / 2, 2.0, 9.0)
    return {"runsA": a, "runsB": b}


def mvp_score_batter(b: Dict[str, Any]) -> float:
    L = get_coeff(b["league"])
    return b["ops"] * L["ops"] + b["hr"] * 0.08 + b["rispOps"] * L["ops"]


def mvp_score_pitcher(p: Dict[str, Any]) -> float:
    L = get_coeff(p["league"])
    era = max(p["era"] * L["era"], 0.5)
    return (1.0 / era) * 18.0 + p["k9"] * L["k9"] * 0.35 + p["ip"] / 22.0


def pick_mvp_candidate(team: Dict[str, Any]) -> Dict[str, Any]:
    best_bat = max(team["batters"], key=mvp_score_batter)
    pool = team["pitchers"]["sp"] + team["pitchers"]["rp"]
    best_pit = max(pool, key=mvp_score_pitcher)
    sb = mvp_score_batter(best_bat)
    sp = mvp_score_pitcher(best_pit)
    bat_pts = sb * 10.0
    pit_pts = sp * 0.92
    if bat_pts >= pit_pts:
        return {
            "name": best_bat["name"],
            "role": "타자",
            "score": clamp(bat_pts * 0.65, 45.0, 99.0),
            "raw": bat_pts,
            "detail": f"OPS·홈런·득점권 가중 ({best_bat['league']})",
        }
    return {
        "name": best_pit["name"],
        "role": "투수",
        "score": clamp(pit_pts * 0.65, 45.0, 99.0),
        "raw": pit_pts,
        "detail": f"ERA·K/9·이닝 가중 ({best_pit['league']})",
    }


def pick_global_mvp(
    team_a: Dict[str, Any], team_b: Dict[str, Any], cand_a: Dict[str, Any], cand_b: Dict[str, Any]
) -> Dict[str, Any]:
    a = {"team": team_a, **cand_a}
    b = {"team": team_b, **cand_b}
    return a if a["raw"] >= b["raw"] else b


def get_team_by_id(team_id: str) -> Optional[Dict[str, Any]]:
    for t in TEAMS:
        if t["id"] == team_id:
            return t
    return None


def predict_matchup(
    team_a: Dict[str, Any],
    team_b: Dict[str, Any],
    options: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    o = options or {}
    scenario = "missing" if o.get("scenario") == "missing" else "full"
    starter_a = team_a["pitchers"]["sp"][team_a["starterGame1"]["idx"]]
    starter_b = team_b["pitchers"]["sp"][team_b["starterGame1"]["idx"]]
    throws_a = starter_a["throws"]
    throws_b = starter_b["throws"]

    lineup_raw_a = team_lineup_score(team_a, throws_b, scenario)
    lineup_raw_b = team_lineup_score(team_b, throws_a, scenario)
    pitch_raw_a = team_pitch_score(team_a)
    pitch_raw_b = team_pitch_score(team_b)

    env = compute_environment(team_a, team_b, float(o.get("weatherBoost") or 0.0))
    lineup_a = clamp(lineup_raw_a * env["mLineupA"], 0.0, 100.0)
    lineup_b = clamp(lineup_raw_b * env["mLineupB"], 0.0, 100.0)
    pitch_a = clamp(pitch_raw_a * env["mPitch"], 0.0, 100.0)
    pitch_b = clamp(pitch_raw_b * env["mPitch"], 0.0, 100.0)

    overall_a = team_overall(lineup_a, pitch_a)
    overall_b = team_overall(lineup_b, pitch_b)

    p_a = logistic_win_prob(overall_a, overall_b)
    p_b = 1.0 - p_a
    ci_a = confidence_interval(p_a)
    ci_b = confidence_interval(p_b)

    wb = float(o.get("weatherBoost") or 0.0)
    park_w_a = ((team_a["parkFactor"] - 1.0) * 0.55 + wb) * 0.35
    park_w_b = ((team_b["parkFactor"] - 1.0) * 0.55 + wb) * 0.35
    scores = expected_scores(
        lineup_a,
        pitch_b,
        lineup_b,
        pitch_a,
        team_a,
        team_b,
        park_w_a,
        park_w_b,
    )

    cand_a = pick_mvp_candidate(team_a)
    cand_b = pick_mvp_candidate(team_b)
    mvp = pick_global_mvp(team_a, team_b, cand_a, cand_b)

    winner = team_a if p_a >= p_b else team_b
    winner_prob = p_a if p_a >= p_b else p_b

    return {
        "winner": winner,
        "winnerProb": winner_prob,
        "teamA": team_a,
        "teamB": team_b,
        "starterA": starter_a["name"],
        "starterB": starter_b["name"],
        "throwsA": throws_a,
        "throwsB": throws_b,
        "lineupRawA": lineup_raw_a,
        "lineupRawB": lineup_raw_b,
        "lineupA": lineup_a,
        "lineupB": lineup_b,
        "pitchRawA": pitch_raw_a,
        "pitchRawB": pitch_raw_b,
        "pitchA": pitch_a,
        "pitchB": pitch_b,
        "environment": env,
        "overallA": overall_a,
        "overallB": overall_b,
        "pA": p_a,
        "pB": p_b,
        "ciA": ci_a,
        "ciB": ci_b,
        "scores": scores,
        "mvp": mvp,
        "candA": cand_a,
        "candB": cand_b,
        "scenario": scenario,
    }


def _slot_weight(order: int) -> float:
    if order <= 3:
        return 0.4 / 3.0
    if order <= 6:
        return 0.35 / 3.0
    return 0.25 / 3.0


def batter_weighted_contributions(
    team: Dict[str, Any], opp_starter_throws: str, scenario: str
) -> List[Dict[str, Any]]:
    """타순 가중 기여(합 ≈ team_lineup_score)."""
    out = []
    for b in team["batters"]:
        s = batter_effective_for_starter(b, opp_starter_throws)
        if scenario == "missing" and b["order"] == 3:
            s *= 0.72
        w = _slot_weight(b["order"])
        out.append(
            {
                "name": b["name"],
                "order": b["order"],
                "score": s,
                "weighted": w * s,
                "tier": "상단" if b["order"] <= 3 else ("중단" if b["order"] <= 6 else "하단"),
            }
        )
    return sorted(out, key=lambda x: x["order"])


def pitcher_weighted_contributions(team: Dict[str, Any]) -> List[Dict[str, Any]]:
    """선발 60%/불펜 40% 가중 기여(합 ≈ team_pitch_score - 수비보너스는 균등 분배 근사)."""
    sps = team["pitchers"]["sp"]
    rps = team["pitchers"]["rp"]
    fat = bullpen_fatigue_debuff(rps)
    rp_scores = [pitcher_raw100(p) for p in rps]
    rp_avg = sum(rp_scores) / len(rp_scores)
    rp_adj = clamp(rp_avg - fat, 0.0, 100.0)
    def_b = defense_pitcher_bonus(team)
    raw_core = 0.6 * (sum(pitcher_raw100(p) for p in sps) / len(sps)) + 0.4 * rp_adj
    bonus = def_b * 0.35
    n = len(sps) + len(rps)
    per_bonus = bonus / n if n else 0.0
    out = []
    for p in sps:
        pr = pitcher_raw100(p)
        out.append(
            {
                "name": p["name"],
                "role": p["role"],
                "score": pr,
                "weighted": 0.6 * pr / len(sps) + per_bonus,
            }
        )
    for p in rps:
        pr = pitcher_raw100(p)
        out.append(
            {
                "name": p["name"],
                "role": p["role"],
                "score": pr,
                "weighted": 0.4 * pr / len(rps) + per_bonus,
            }
        )
    return out


def win_prob_sweep(
    overall_a: float, overall_b: float, deltas: Optional[List[float]] = None
) -> Tuple[List[float], List[float]]:
    """능력 차 변화에 따른 P(A) 곡선용."""
    if deltas is None:
        deltas = [d * 0.5 for d in range(-20, 21)]
    base_diff = overall_a - overall_b
    xs = [base_diff + d for d in deltas]
    ys = [logistic_win_prob(overall_a + d, overall_b) for d in deltas]
    return xs, ys
