"""과거 경기 백테스트 샘플 (100경기 확장 가능)."""

from __future__ import annotations

from typing import Any, Dict, List

# 실제 2026 WBC 결과 확정 후 actual_winner / score 갱신
ACTUAL_2026_WBC = {
    "champion": None,  # 예: "jpn"
    "qf_results": [
        # {"team_a": "usa", "team_b": "aus", "winner": "usa", "score": "6-2"},
    ],
}

BACKTEST_GAMES: List[Dict[str, Any]] = []


def _gen_backtest_games() -> List[Dict[str, Any]]:
    """8강 팀 조합 기반 100경기 시뮬 레이블 (데모)."""
    pairs = [
        ("usa", "kor"),
        ("usa", "jpn"),
        ("usa", "dom"),
        ("jpn", "kor"),
        ("jpn", "dom"),
        ("dom", "kor"),
        ("mex", "ven"),
        ("pur", "aus"),
        ("usa", "mex"),
        ("jpn", "ven"),
        ("kor", "pur"),
        ("dom", "aus"),
    ]
    games = []
    for i in range(100):
        a, b = pairs[i % len(pairs)]
        home = a if i % 2 == 0 else b
        # 데모 실제 승자: 시드 기반 의사결정 (실데이터로 교체)
        winner = a if (hash((a, b, i)) % 100) > 45 else b
        games.append(
            {
                "id": i + 1,
                "team_a": a,
                "team_b": b,
                "home": home,
                "weather_mult": 1.0 if i % 5 else 0.96,
                "actual_winner": winner,
                "actual_score_a": 4 + (i % 4),
                "actual_score_b": 3 + (i % 3),
                "note": "데모 라벨 — 실제 경기 결과 CSV로 교체",
            }
        )
    return games


BACKTEST_GAMES = _gen_backtest_games()
