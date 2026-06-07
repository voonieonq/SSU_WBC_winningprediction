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
from rosters import BRACKET_QF, TEAMS_QF8
from tournament import run_monte_carlo
from validation import run_backtest, validate_2026
from weather_api import fetch_temperature_run_multiplier, manual_weather_multiplier

TEAM_COLORS = {
    "kor": "#0046A0",
    "usa": "#BF0A30",
    "jpn": "#BC002D",
    "dom": "#002D62",
    "mex": "#006847",
    "ven": "#FFCC00",
    "aus": "#FFCD00",
    "cub": "#002A8F",
}

for _font in ("Malgun Gothic", "AppleGothic", "NanumGothic"):
    try:
        plt.rcParams["font.family"] = _font
        break
    except Exception:
        pass
plt.rcParams["axes.unicode_minus"] = False


def _theme() -> dict:
    light = st.get_option("theme.base") == "light"
    return {
        "light": light,
        "text": "#262730" if light else "#FAFAFA",
        "muted": "rgba(38,39,48,0.55)" if light else "rgba(250,250,250,0.55)",
        "grid_css": "rgba(38,39,48,0.12)" if light else "rgba(250,250,250,0.10)",
        "grid_mpl": "#d6d6d8" if light else "#3a3b45",
        "card_bg": "rgba(38,39,48,0.04)" if light else "rgba(250,250,250,0.06)",
        "card_border": "rgba(38,39,48,0.14)" if light else "rgba(250,250,250,0.14)",
        "accent": "#1f77b4" if light else "#7dd3fc",
    }


def _inject_global_css() -> None:
    t = _theme()
    st.markdown(
        f"""
        <style>
        [data-testid="stPyplotGlobal"] {{
            background: transparent !important;
        }}
        [data-testid="stPyplotGlobal"] img {{
            background: transparent !important;
        }}
        .wbc-card {{
            border: 1px solid {t["card_border"]};
            border-radius: 12px;
            padding: 1rem 1.25rem;
            background: {t["card_bg"]};
            margin-bottom: 0.75rem;
        }}
        .wbc-winner {{
            font-size: 0.85rem;
            color: {t["accent"]};
            letter-spacing: 0.04em;
            margin-bottom: 0.25rem;
        }}
        .wbc-team {{
            font-size: 1.35rem;
            font-weight: 700;
            margin: 0;
        }}
        .wbc-pct {{
            font-size: 2.6rem;
            font-weight: 800;
            line-height: 1.1;
            margin: 0.15rem 0 0.5rem 0;
        }}
        .wbc-vs {{
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 1.1rem;
            font-weight: 700;
            color: {t["muted"]};
            padding-top: 2.5rem;
        }}
        .wbc-bar-block {{
            margin: 0.35rem 0 0.85rem 0;
        }}
        .wbc-bar-label {{
            display: flex;
            justify-content: space-between;
            font-size: 0.9rem;
            color: {t["text"]};
            margin-bottom: 0.25rem;
        }}
        .wbc-bar-track {{
            width: 100%;
            height: 0.65rem;
            border-radius: 999px;
            background: {t["grid_css"]};
            overflow: hidden;
        }}
        .wbc-bar-fill {{
            height: 100%;
            border-radius: 999px;
            transition: width 0.15s ease;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


@st.cache_resource
def get_model():
    return build_model()


def _apply_mpl_theme(fig, ax) -> None:
    t = _theme()
    fig.patch.set_alpha(0)
    ax.set_facecolor("none")
    ax.tick_params(colors=t["text"], labelsize=10)
    ax.xaxis.label.set_color(t["text"])
    ax.yaxis.label.set_color(t["text"])
    ax.title.set_color(t["text"])
    for spine in ax.spines.values():
        spine.set_color(t["grid_mpl"])
    ax.grid(axis="both", color=t["grid_mpl"], linestyle="-", linewidth=0.6, alpha=0.9)
    legend = ax.get_legend()
    if legend:
        for text in legend.get_texts():
            text.set_color(t["text"])


def _show_fig(fig) -> None:
    fig.tight_layout()
    try:
        st.pyplot(fig, use_container_width=True, transparent=True)
    except TypeError:
        st.pyplot(fig, use_container_width=True)
    plt.close(fig)


def render_win_prob_bars(name_a: str, name_b: str, win_a: float, win_b: float, color_a: str, color_b: str):
    pa, pb = win_a * 100, win_b * 100
    st.markdown(
        f"""
        <div class="wbc-bar-block">
            <div class="wbc-bar-label"><span>{name_a}</span><span>{pa:.1f}%</span></div>
            <div class="wbc-bar-track">
                <div class="wbc-bar-fill" style="width:{pa:.1f}%; background:{color_a};"></div>
            </div>
        </div>
        <div class="wbc-bar-block">
            <div class="wbc-bar-label"><span>{name_b}</span><span>{pb:.1f}%</span></div>
            <div class="wbc-bar-track">
                <div class="wbc-bar-fill" style="width:{pb:.1f}%; background:{color_b};"></div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_matchup_hero(ta, tb, pred, labels, id_a, id_b):
    """승률을 한눈에 파악할 수 있는 메인 카드."""
    win_a = pred.win_prob_a
    win_b = pred.win_prob_b
    winner_id = id_a if win_a >= win_b else id_b
    color_a = TEAM_COLORS.get(id_a, "#1f77b4")
    color_b = TEAM_COLORS.get(id_b, "#ff4b4b")

    st.markdown(
        f"""
        <div class="wbc-card">
            <div class="wbc-winner">🏆 예상 승자 · {labels[winner_id]}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col_a, col_vs, col_b = st.columns([5, 1, 5])
    with col_a:
        st.markdown(
            f'<p class="wbc-team" style="color:{color_a}">{ta.name_ko}</p>'
            f'<p class="wbc-pct" style="color:{color_a}">{win_a * 100:.1f}%</p>',
            unsafe_allow_html=True,
        )
        st.progress(win_a, text=f"승률 {win_a * 100:.1f}%")
    with col_vs:
        st.markdown('<div class="wbc-vs">VS</div>', unsafe_allow_html=True)
    with col_b:
        st.markdown(
            f'<p class="wbc-team" style="color:{color_b}">{tb.name_ko}</p>'
            f'<p class="wbc-pct" style="color:{color_b}">{win_b * 100:.1f}%</p>',
            unsafe_allow_html=True,
        )
        st.progress(win_b, text=f"승률 {win_b * 100:.1f}%")

    st.caption("최종 승률 비교")
    render_win_prob_bars(labels[id_a], labels[id_b], win_a, win_b, color_a, color_b)


def plot_lineup_strength(power, team_name: str, team_id: str):
    names = [f"{s.order}번 {s.name}" for s in power.lineup]
    vals = [s.score for s in power.lineup]
    color = TEAM_COLORS.get(team_id, "#58a6ff")

    fig, ax = plt.subplots(figsize=(8, 3.5))
    y_pos = np.arange(len(names))
    ax.barh(y_pos, vals, color=color, alpha=0.88, edgecolor="none")
    ax.set_yticks(y_pos)
    ax.set_yticklabels(names)
    ax.invert_yaxis()
    ax.set_xlabel("전력 점수")
    ax.set_title(f"{team_name} 추천 라인업 전력")
    _apply_mpl_theme(fig, ax)
    _show_fig(fig)


def plot_win_factors(pred, name_a: str, name_b: str):
    labels = ["타선", "투수", "종합", "예상 득점(λ)"]
    a_vals = [
        pred.power_a.lineup_strength,
        pred.power_a.pitching_strength,
        pred.power_a.overall,
        pred.lambda_a,
    ]
    b_vals = [
        pred.power_b.lineup_strength,
        pred.power_b.pitching_strength,
        pred.power_b.overall,
        pred.lambda_b,
    ]
    x = np.arange(len(labels))
    w = 0.35

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(x - w / 2, a_vals, w, label=name_a, color="#58a6ff", edgecolor="none")
    ax.bar(x + w / 2, b_vals, w, label=name_b, color="#f85149", edgecolor="none")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend(framealpha=0)
    ax.set_title("승률 요인 비교 (포아송 λ 포함)")
    _apply_mpl_theme(fig, ax)
    _show_fig(fig)


def plot_poisson_heatmap(pred):
    from poisson_match import poisson_pmf

    lam_a, lam_b = pred.lambda_a, pred.lambda_b
    grid = np.zeros((8, 8))
    for i in range(8):
        for j in range(8):
            grid[i, j] = poisson_pmf(i, lam_a) * poisson_pmf(j, lam_b)

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(grid, cmap="viridis", origin="lower")
    ax.set_xlabel("B팀 득점")
    ax.set_ylabel("A팀 득점")
    ax.set_title("득점 조합 확률 (포아송)")
    cbar = plt.colorbar(im, ax=ax, fraction=0.046)
    cbar.ax.yaxis.set_tick_params(color=_theme()["text"])
    plt.setp(plt.getp(cbar.ax.axes, "yticklabels"), color=_theme()["text"])
    _apply_mpl_theme(fig, ax)
    _show_fig(fig)


def main():
    st.set_page_config(page_title="2026 WBC 8강 예측", page_icon="⚾", layout="wide")
    _inject_global_css()
    st.title("⚾ 2026 WBC 8강 승부예측")
    st.caption("사이드바 설정 변경 시 승률이 즉시 반영됩니다 · 포아송 기반 단판 예측")

    bundle = get_model()
    ctx = bundle.scoring_ctx
    teams = bundle.teams

    with st.sidebar:
        st.header("⚙️ 경기 설정")
        ids = list(teams.keys())
        labels = {t.id: f"{t.name_ko}" for t in TEAMS_QF8}

        id_a = st.selectbox("A팀", ids, format_func=lambda x: labels[x], index=0)
        id_b = st.selectbox("B팀", ids, format_func=lambda x: labels[x], index=1)
        home = st.selectbox(
            "홈팀",
            [id_a, id_b, "neutral"],
            format_func=lambda x: labels.get(x, "중립 구장"),
        )

        st.divider()
        st.subheader("🌤 날씨")
        use_api = st.checkbox("Open-Meteo 실시간 API", value=False)
        if use_api:
            ht = teams[id_a if home != "neutral" else id_a]
            wmult, wnote = fetch_temperature_run_multiplier(ht.venue_lat, ht.venue_lon)
        else:
            temp = st.slider("기온 (°C)", 5, 38, 22)
            wind = st.slider("풍속 (km/h)", 0, 40, 8)
            wmult, wnote = manual_weather_multiplier(temp, wind)
        st.caption(wnote)

        st.divider()
        st.subheader("🏥 부상")
        injured_b = st.multiselect(
            "부상 타자",
            [b.name for t in TEAMS_QF8 for b in t.batters],
        )
        injured_p = st.multiselect(
            "부상 투수",
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

    # ── TAB 1: 경기 예측 ──
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

            render_matchup_hero(ta, tb, pred, labels, id_a, id_b)

            c1, c2, c3 = st.columns(3)
            c1.metric("예상 득점 λ", f"{pred.lambda_a:.2f} : {pred.lambda_b:.2f}")
            c2.metric("A팀 기대 득점", f"{pred.expected_runs_a:.2f}")
            c3.metric("B팀 기대 득점", f"{pred.expected_runs_b:.2f}")

            st.info(f"🌤 {pred.weather_note}  ·  🏟 {pred.home_away_note}  ·  🧤 {pred.defense_note}")

            with st.expander("리그 계수 보기"):
                st.dataframe(
                    pd.DataFrame(
                        [
                            {
                                "리그": k,
                                "OPS 계수": v.ops,
                                "ERA 계수": v.era,
                                "연봉 비율": v.salary_ratio,
                            }
                            for k, v in bundle.coeffs.items()
                        ]
                    ),
                    hide_index=True,
                    use_container_width=True,
                )

            with st.expander("추천 라인업 상세"):
                col_l, col_r = st.columns(2)
                with col_l:
                    st.subheader(ta.name_ko)
                    st.dataframe(
                        pd.DataFrame(
                            [
                                {"타순": s.order, "선수": s.name, "전력": round(s.score, 1)}
                                for s in pred.power_a.lineup
                            ]
                        ),
                        hide_index=True,
                        use_container_width=True,
                    )
                with col_r:
                    st.subheader(tb.name_ko)
                    st.dataframe(
                        pd.DataFrame(
                            [
                                {"타순": s.order, "선수": s.name, "전력": round(s.score, 1)}
                                for s in pred.power_b.lineup
                            ]
                        ),
                        hide_index=True,
                        use_container_width=True,
                    )

            st.subheader("📊 상세 시각화")
            viz_tabs = st.tabs(["라인업 전력", "승률 요인", "득점 확률 분포"])
            with viz_tabs[0]:
                c1, c2 = st.columns(2)
                with c1:
                    plot_lineup_strength(pred.power_a, ta.name_ko, id_a)
                with c2:
                    plot_lineup_strength(pred.power_b, tb.name_ko, id_b)
            with viz_tabs[1]:
                plot_win_factors(pred, ta.name_ko, tb.name_ko)
            with viz_tabs[2]:
                plot_poisson_heatmap(pred)

            st.subheader("🎯 유력 스코어 시나리오")
            score_df = pd.DataFrame(
                [
                    {"A팀": a, "B팀": b, "확률 (%)": round(p * 100, 2)}
                    for a, b, p in pred.score_matrix_top
                ]
            )
            st.dataframe(score_df, hide_index=True, use_container_width=True)
            if not score_df.empty:
                top = score_df.iloc[0]
                st.success(
                    f"가장 유력한 스코어: **{int(top['A팀'])} : {int(top['B팀'])}** "
                    f"(확률 {top['확률 (%)']:.1f}%)"
                )

    # ── TAB 2: 8강 시뮬 ──
    with tabs[1]:
        st.subheader("2026 WBC 8강 몬테카를로 시뮬레이션")
        n_sim = st.slider("시뮬레이션 횟수", 1000, 20000, 10000, 1000)
        home_qf = st.selectbox("8강 홈 구장 기준 팀", ids, format_func=lambda x: labels[x])
        if st.button("시뮬레이션 실행", type="primary"):
            with st.spinner("8강 토너먼트 시뮬레이션 중..."):
                sim = run_monte_carlo(teams, ctx, n_sim, wmult, home_qf)
            st.success(f"{sim.n_sims:,}회 시뮬레이션 완료")

            champ_df = pd.DataFrame(
                [
                    {"팀": labels[k], "우승 확률 (%)": round(v * 100, 2)}
                    for k, v in sorted(sim.champion_probs.items(), key=lambda x: -x[1])
                ]
            )

            top_team = champ_df.iloc[0]
            c1, c2 = st.columns(2)
            c1.metric("우승 최유력", top_team["팀"])
            c2.metric("우승 확률", f"{top_team['우승 확률 (%)']:.1f}%")

            st.bar_chart(
                champ_df.set_index("팀")["우승 확률 (%)"],
                horizontal=True,
            )
            st.dataframe(champ_df, hide_index=True, use_container_width=True)

            st.subheader("8강 대진")
            for a, b in BRACKET_QF:
                st.write(f"**{labels[a]}** vs **{labels[b]}**")

    # ── TAB 3: 백테스트/검증 ──
    with tabs[2]:
        st.subheader(f"백테스트 ({len(BACKTEST_GAMES)}경기)")
        if st.button("백테스트 실행"):
            bt = run_backtest(BACKTEST_GAMES, teams, ctx)
            c1, c2 = st.columns(2)
            c1.metric("승자 적중률", f"{bt.winner_accuracy * 100:.1f}%")
            c2.metric("Brier Score", f"{bt.brier_score:.4f}")
            st.dataframe(pd.DataFrame(bt.details), hide_index=True, use_container_width=True)

        st.subheader("2026 WBC 실제 결과 검증")
        st.json(ACTUAL_2026_WBC)
        if st.button("검증 리포트 실행"):
            sim = run_monte_carlo(teams, ctx, 5000, wmult, id_a)
            val = validate_2026(sim.champion_probs, ACTUAL_2026_WBC, [])
            st.write(f"예측 우승팀: **{labels.get(val.predicted_champion, '?')}**")
            st.write(f"실제 우승팀: **{val.actual_champion or '미입력'}**")
            st.write(f"우승 적중: **{val.champion_match}**")
            st.caption(val.notes)

    # ── TAB 4: AI·논문 ──
    with tabs[3]:
        st.markdown(PAPER_VS_PROJECT)
        st.markdown(AI_USAGE)
        st.markdown(DIFFERENTIATION)
        st.markdown(ROOKIE_INJURY)
        st.markdown(FUTURE_PARAMS)
        st.markdown(DATA_SOURCES)

    # ── TAB 5: 월드컵 확장 ──
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
        st.dataframe(football, hide_index=True, use_container_width=True)
        st.bar_chart(football.set_index("국가")["λ(골)"])
        st.caption("동일한 포아송+MC 프레임워크에 xG 기반 접근을 적용하면 월드컵 모듈이 완성됩니다.")


if __name__ == "__main__":
    main()
