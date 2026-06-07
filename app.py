"""
2026 WBC 8강 승부예측 시스템
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from backtest_data import ACTUAL_2026_WBC, BACKTEST_GAMES
from content_docs import (
    AI_USAGE,
    DATA_SOURCES,
    DIFFERENTIATION,
    FUTURE_PARAMS,
    PAPER_VS_PROJECT,
    ROOKIE_INJURY,
    WORLD_CUP_STUB,
)
from models import MatchContext
from pipeline import build_model
from poisson_match import predict_match
from rosters import BRACKET_QF, TEAM_BY_ID, TEAMS_QF8
from tournament import run_monte_carlo
from validation import run_backtest, validate_2026
from weather_api import fetch_temperature_run_multiplier, manual_weather_multiplier

for _font in ("Malgun Gothic", "AppleGothic", "NanumGothic"):
    try:
        plt.rcParams["font.family"] = _font
        break
    except Exception:
        pass
plt.rcParams["axes.unicode_minus"] = False


@st.cache_resource
def get_model():
    return build_model()


def plot_lineup_stack(power, team_name: str, color: str):
    names = [f"{s.order} {s.name}" for s in power.lineup]
    vals = [s.score for s in power.lineup]
    fig, ax = plt.subplots(figsize=(10, 3))
    left = 0.0
    cmap = plt.cm.Blues if "blue" in color else plt.cm.Reds
    cols = cmap(np.linspace(0.35, 0.9, len(vals)))
    for n, v, c in zip(names, vals, cols):
        ax.barh(0, v, left=left, height=0.6, color=c, edgecolor="#333")
        left += v
    ax.set_yticks([0])
    ax.set_yticklabels([team_name])
    ax.set_xlabel("타순별 능력 점수 (누적 스택)")
    ax.set_title(f"{team_name} 추천 라인업")
    return fig


def plot_win_factors(pred):
    fig, ax = plt.subplots(figsize=(8, 4))
    labels = ["타선", "투수", "종합", "λ(득점)"]
    a = [
        pred.power_a.lineup_strength,
        pred.power_a.pitching_strength,
        pred.power_a.overall,
        pred.lambda_a,
    ]
    b = [
        pred.power_b.lineup_strength,
        pred.power_b.pitching_strength,
        pred.power_b.overall,
        pred.lambda_b,
    ]
    x = np.arange(len(labels))
    w = 0.35
    ax.bar(x - w / 2, a, w, label="A", color="#58a6ff")
    ax.bar(x + w / 2, b, w, label="B", color="#f85149")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    ax.set_title("승률 관련 요소 (포아송 λ 포함)")
    return fig


def plot_poisson_heatmap(pred):
    lam_a, lam_b = pred.lambda_a, pred.lambda_b
    grid = np.zeros((8, 8))
    from poisson_match import poisson_pmf

    for i in range(8):
        for j in range(8):
            grid[i, j] = poisson_pmf(i, lam_a) * poisson_pmf(j, lam_b)
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(grid, cmap="viridis", origin="lower")
    ax.set_xlabel("B 득점")
    ax.set_ylabel("A 득점")
    ax.set_title("득점 조합 확률 (포아송)")
    plt.colorbar(im, ax=ax, fraction=0.046)
    return fig


def main():
    st.set_page_config(page_title="2026 WBC 8강 예측", layout="wide")
    st.title("2026 WBC 8강 승부예측 시스템")
    st.caption("명세서 STEP1~6 · 포아송 · MC 10,000 · 백테스트 · (확장) 날씨·홈어웨이·수비")

    bundle = get_model()
    ctx = bundle.scoring_ctx
    teams = bundle.teams

    with st.sidebar:
        st.header("경기 설정")
        ids = list(teams.keys())
        labels = {t.id: f"{t.name_ko}" for t in TEAMS_QF8}
        id_a = st.selectbox("팀 A", ids, format_func=lambda x: labels[x], index=0)
        id_b = st.selectbox("팀 B", ids, format_func=lambda x: labels[x], index=1)
        home = st.selectbox("홈팀", [id_a, id_b, "neutral"], format_func=lambda x: labels.get(x, "중립"))
        st.subheader("날씨 → 득점 → 승패")
        use_api = st.checkbox("Open-Meteo API", value=False)
        if use_api:
            ht = teams[id_a if home != "neutral" else id_a]
            wmult, wnote = fetch_temperature_run_multiplier(ht.venue_lat, ht.venue_lon)
        else:
            temp = st.slider("기온 (°C)", 5, 38, 22)
            wind = st.slider("풍속 (km/h)", 0, 40, 8)
            wmult, wnote = manual_weather_multiplier(temp, wind)
        st.caption(wnote)
        injured_b = st.multiselect(
            "부상 제외 타자",
            [b.name for t in TEAMS_QF8 for b in t.batters],
        )
        injured_p = st.multiselect(
            "부상 제외 투수",
            [p.name for t in TEAMS_QF8 for p in t.pitchers],
        )

    tabs = st.tabs(
        [
            "경기 예측",
            "8강 시뮬",
            "백테스트/검증",
            "AI·논문",
            "월드컵 확장",
        ]
    )

    # ── TAB 1 ──
    with tabs[0]:
        if id_a == id_b:
            st.error("서로 다른 팀을 선택하세요.")
        else:
            ta, tb = teams[id_a], teams[id_b]
            home_id = "neutral" if home == "neutral" else home
            mc = MatchContext(
                home_team_id=home_id,
                away_team_id=id_b,
                injured_batter_names=injured_b,
                injured_pitcher_names=injured_p,
            )
            pred = predict_match(ta, tb, ctx, mc, weather_mult=wmult, weather_note=wnote)

            c1, c2, c3, c4 = st.columns(4)
            win = id_a if pred.win_prob_a >= pred.win_prob_b else id_b
            c1.metric("예상 승자", labels[win])
            c2.metric(labels[id_a] + " 승률", f"{pred.win_prob_a*100:.1f}%")
            c3.metric(labels[id_b] + " 승률", f"{pred.win_prob_b*100:.1f}%")
            c4.metric("예상 λ", f"{pred.lambda_a:.2f} : {pred.lambda_b:.2f}")

            st.info(f"🌤 {pred.weather_note} | 🏟 {pred.home_away_note} | 🧤 {pred.defense_note}")

            with st.expander("League coefficient"):
                st.dataframe(
                    pd.DataFrame(
                        [
                            {
                                "League": k,
                                "OPS coefficient": v.ops,
                                "ERA coefficient": v.era,
                                "Salary Ratio": v.salary_ratio,
                            }
                            for k, v in bundle.coeffs.items()
                        ]
                    ),
                    hide_index=True,
                )

            col_l, col_r = st.columns(2)
            with col_l:
                st.subheader(f"{ta.name_ko} 라인업")
                st.dataframe(
                    pd.DataFrame([{"batting order": s.order, "Player": s.name, "Rate": round(s.score, 1)} for s in pred.power_a.lineup]),
                    hide_index=True,
                )
            with col_r:
                st.subheader(f"{tb.name_ko} 라인업")
                st.dataframe(
                    pd.DataFrame([{"batting order": s.order, "Player": s.name, "Rate": round(s.score, 1)} for s in pred.power_b.lineup]),
                    hide_index=True,
                )

            st.subheader("Visualization")
            f1 = plot_lineup_stack(pred.power_a, ta.name_ko, "blue")
            st.pyplot(f1)
            plt.close(f1)
            f2 = plot_lineup_stack(pred.power_b, tb.name_ko, "red")
            st.pyplot(f2)
            plt.close(f2)
            f3 = plot_win_factors(pred)
            st.pyplot(f3)
            plt.close(f3)
            f4 = plot_poisson_heatmap(pred)
            st.pyplot(f4)
            plt.close(f4)

            st.subheader("Top Scoring Scenarios")
            st.dataframe(
                pd.DataFrame(
                    [
                        {"A": a, "B": b, "Probability %": round(p * 100, 2)}
                        for a, b, p in pred.score_matrix_top
                    ]
                ),
                hide_index=True,
            )

    # ── TAB 2 ──
    with tabs[1]:
        n_sim = st.slider("Number of Simulations", 1000, 20000, 10000, 1000)
        home_qf = st.selectbox("8th Round Home Stadium Criteria Team", ids, format_func=lambda x: labels[x])
        if st.button("Run Monte Carlo"):
            with st.spinner("Simulating..."):
                sim = run_monte_carlo(teams, ctx, n_sim, wmult, home_qf)
            st.success(f"{sim.n_sims} simulations completed")
            champ_df = pd.DataFrame(
                [{"TEAM": labels[k], "Championship Probability %": round(v * 100, 2)} for k, v in sorted(sim.champion_probs.items(), key=lambda x: -x[1])]
            )
            st.subheader("Championship Probability")
            st.bar_chart(champ_df.set_index("TEAM")["Championship Probability %"])
            st.dataframe(champ_df, hide_index=True)
            st.subheader("Round of 16 Match ups")
            for a, b in BRACKET_QF:
                st.write(f"**{labels[a]}** vs **{labels[b]}**")

    # ── TAB 3 ──
    with tabs[2]:
        st.subheader(f"Backtest ({len(BACKTEST_GAMES)} Games)")
        if st.button("Run Backtest"):
            bt = run_backtest(BACKTEST_GAMES, teams, ctx)
            st.metric("Winner Accuracy", f"{bt.winner_accuracy*100:.1f}%")
            st.metric("Brier Score", f"{bt.brier_score:.4f}")
            st.dataframe(pd.DataFrame(bt.details), hide_index=True, use_container_width=True)

        st.subheader("2026 WBC Verification")
        st.json(ACTUAL_2026_WBC)
        if st.button("Run Validation Report"):
            sim = run_monte_carlo(teams, ctx, 5000, wmult, id_a)
            val = validate_2026(sim.champion_probs, ACTUAL_2026_WBC, [])
            st.write(f"Predicted Champion: **{labels.get(val.predicted_champion, '?')}**")
            st.write(f"Actual Champion: **{val.actual_champion or 'Not Entered'}**")
            st.write(f"Champion Match: **{val.champion_match}**")
            st.caption(val.notes)

    # ── TAB 4 ──
    with tabs[3]:
        st.markdown(PAPER_VS_PROJECT)
        st.markdown(AI_USAGE)
        st.markdown(DIFFERENTIATION)
        st.markdown(ROOKIE_INJURY)
        st.markdown(FUTURE_PARAMS)
        st.markdown(DATA_SOURCES)

    # ── TAB 5 ──
    with tabs[4]:
        st.markdown(WORLD_CUP_STUB)
        st.subheader("데모: 4개국 골 λ")
        football = pd.DataFrame(
            {
                "국가": ["브라질", "프랑스", "한국", "일본"],
                "공격": [88, 85, 72, 74],
                "수비": [82, 84, 78, 76],
                "λ(골)": [1.65, 1.58, 1.12, 1.18],
            }
        )
        st.dataframe(football, hide_index=True)
        st.bar_chart(football.set_index("국가")["λ(골)"])
        st.caption("동일 포아송+MC 프레임을 xG 기반으로 교체하면 월드컵 모듈 완성됩니다.")


if __name__ == "__main__":
    main()
