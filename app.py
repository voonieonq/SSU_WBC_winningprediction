"""
2026 WBC 8강 승부예측 시스템
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
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


@st.cache_resource
def get_model():
    return build_model()


def inject_custom_css():
    st.markdown(
        """
        <style>
        .block-container {
            padding-top: 2rem;
            padding-bottom: 2rem;
        }

        div[data-testid="stMetric"] {
            background-color: #ffffff;
            border: 1px solid #e5e7eb;
            padding: 16px;
            border-radius: 14px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.06);
        }

        div[data-testid="stMetricLabel"] {
            font-size: 0.95rem;
        }

        div[data-testid="stMetricValue"] {
            font-size: 1.6rem;
            font-weight: 700;
        }

        section[data-testid="stSidebar"] {
            border-right: 1px solid #e5e7eb;
        }

        h1, h2, h3 {
            letter-spacing: -0.03em;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def plot_lineup_stack(power, team_name: str, color: str):
    df = pd.DataFrame(
        {
            "Player": [f"{s.order}. {s.name}" for s in power.lineup],
            "Score": [s.score for s in power.lineup],
        }
    )

    palette = "Blues" if "blue" in color.lower() else "Reds"

    fig = px.bar(
        df,
        x="Score",
        y="Player",
        orientation="h",
        color="Score",
        color_continuous_scale=palette,
        text="Score",
        title=f"{team_name} Recommended Lineup",
    )

    fig.update_traces(
        texttemplate="%{text:.1f}",
        textposition="outside",
        hovertemplate="<b>%{y}</b><br>Ability Score: %{x:.2f}<extra></extra>",
    )

    fig.update_yaxes(autorange="reversed")

    fig.update_layout(
        height=430,
        template="plotly_white",
        title_font_size=20,
        xaxis_title="Ability Score",
        yaxis_title="Batting Order",
        coloraxis_showscale=False,
        margin=dict(l=20, r=20, t=60, b=20),
    )

    return fig


def plot_win_factors(pred):
    labels = ["Lineup", "Pitching", "Overall", "Expected Goals"]

    df = pd.DataFrame(
        {
            "Factor": labels * 2,
            "Team": ["Team A"] * 4 + ["Team B"] * 4,
            "Value": [
                pred.power_a.lineup_strength,
                pred.power_a.pitching_strength,
                pred.power_a.overall,
                pred.lambda_a,
                pred.power_b.lineup_strength,
                pred.power_b.pitching_strength,
                pred.power_b.overall,
                pred.lambda_b,
            ],
        }
    )

    fig = px.bar(
        df,
        x="Factor",
        y="Value",
        color="Team",
        barmode="group",
        text="Value",
        title="Win Probability Factors",
        color_discrete_map={
            "Team A": "#3b82f6",
            "Team B": "#ef4444",
        },
    )

    fig.update_traces(
        texttemplate="%{text:.2f}",
        textposition="outside",
        hovertemplate="<b>%{x}</b><br>%{fullData.name}: %{y:.2f}<extra></extra>",
    )

    fig.update_layout(
        height=430,
        template="plotly_white",
        title_font_size=20,
        xaxis_title=None,
        yaxis_title="Score / Expected Goals",
        legend_title=None,
        margin=dict(l=20, r=20, t=60, b=20),
    )

    return fig


def plot_poisson_heatmap(pred):
    lam_a, lam_b = pred.lambda_a, pred.lambda_b
    grid = np.zeros((8, 8))

    from poisson_match import poisson_pmf

    for i in range(8):
        for j in range(8):
            grid[i, j] = poisson_pmf(i, lam_a) * poisson_pmf(j, lam_b)

    fig = go.Figure(
        data=go.Heatmap(
            z=grid,
            x=list(range(8)),
            y=list(range(8)),
            colorscale="Viridis",
            hovertemplate=(
                "Team A Goals: %{y}<br>"
                "Team B Goals: %{x}<br>"
                "Probability: %{z:.2%}<extra></extra>"
            ),
            colorbar=dict(title="Probability"),
        )
    )

    fig.update_layout(
        title="Goal Combination Probabilities",
        height=520,
        template="plotly_white",
        xaxis_title="Team B Goals",
        yaxis_title="Team A Goals",
        margin=dict(l=20, r=20, t=60, b=20),
    )

    return fig


def plot_championship_probabilities(champ_df: pd.DataFrame):
    fig = px.bar(
        champ_df,
        x="TEAM",
        y="Championship Probability %",
        text="Championship Probability %",
        title="Championship Probability",
    )

    fig.update_traces(
        texttemplate="%{text:.2f}%",
        textposition="outside",
        hovertemplate="<b>%{x}</b><br>Championship Probability: %{y:.2f}%<extra></extra>",
    )

    fig.update_layout(
        height=480,
        template="plotly_white",
        xaxis_title=None,
        yaxis_title="Probability (%)",
        margin=dict(l=20, r=20, t=60, b=20),
    )

    return fig


def plot_world_cup_lambda(football: pd.DataFrame):
    fig = px.bar(
        football,
        x="Country",
        y="λ(Goals)",
        text="λ(Goals)",
        title="Demo Goal λ by Country",
    )

    fig.update_traces(
        texttemplate="%{text:.2f}",
        textposition="outside",
        hovertemplate="<b>%{x}</b><br>Expected Goals λ: %{y:.2f}<extra></extra>",
    )

    fig.update_layout(
        height=420,
        template="plotly_white",
        xaxis_title=None,
        yaxis_title="Expected Goals λ",
        margin=dict(l=20, r=20, t=60, b=20),
    )

    return fig


def main():
    st.set_page_config(page_title="2026 WBC 8강 예측", layout="wide")
    inject_custom_css()

    st.title("2026 WBC 8강 승부예측 시스템")
    st.caption(
        "홈어웨이, 날씨, 부상 등 다양한 요소를 반영한 승부예측과 시뮬레이션 대시보드입니다!"
    )

    bundle = get_model()
    ctx = bundle.scoring_ctx
    teams = bundle.teams

    with st.sidebar:
        st.header("Match Settings")

        ids = list(teams.keys())
        labels = {t.id: f"{t.name_ko}" for t in TEAMS_QF8}

        id_a = st.selectbox(
            "Team A",
            ids,
            format_func=lambda x: labels[x],
            index=0,
        )

        id_b = st.selectbox(
            "Team B",
            ids,
            format_func=lambda x: labels[x],
            index=1,
        )

        home = st.selectbox(
            "Home Team",
            [id_a, id_b, "neutral"],
            format_func=lambda x: labels.get(x, "Neutral"),
        )

        st.subheader("Weather → Goals → Outcome")

        use_api = st.checkbox("Open-Meteo API", value=False)

        if use_api:
            ht = teams[id_a if home != "neutral" else id_a]
            wmult, wnote = fetch_temperature_run_multiplier(
                ht.venue_lat,
                ht.venue_lon,
            )
        else:
            temp = st.slider("Temperature (°C)", 5, 38, 22)
            wind = st.slider("Wind Speed (km/h)", 0, 40, 8)
            wmult, wnote = manual_weather_multiplier(temp, wind)

        st.caption(wnote)

        injured_b = st.multiselect(
            "Injured Batters",
            [b.name for t in TEAMS_QF8 for b in t.batters],
        )

        injured_p = st.multiselect(
            "Injured Pitchers",
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

            pred = predict_match(
                ta,
                tb,
                ctx,
                mc,
                weather_mult=wmult,
                weather_note=wnote,
            )

            with st.container(border=True):
                c1, c2, c3, c4 = st.columns(4)

                win = id_a if pred.win_prob_a >= pred.win_prob_b else id_b

                c1.metric("Expected Winner", labels[win])
                c2.metric(
                    f"{labels[id_a]} Win Probability",
                    f"{pred.win_prob_a * 100:.1f}%",
                )
                c3.metric(
                    f"{labels[id_b]} Win Probability",
                    f"{pred.win_prob_b * 100:.1f}%",
                )
                c4.metric(
                    "Expected λ",
                    f"{pred.lambda_a:.2f} : {pred.lambda_b:.2f}",
                )

            st.info(
                f"🌤️ {pred.weather_note} | 🏟️ {pred.home_away_note} | 🧤 {pred.defense_note}"
            )

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
                    use_container_width=True,
                )

            col_l, col_r = st.columns(2)

            with col_l:
                with st.container(border=True):
                    st.subheader(f"{ta.name_ko} lineup")
                    st.dataframe(
                        pd.DataFrame(
                            [
                                {
                                    "batting order": s.order,
                                    "Player": s.name,
                                    "Rate": round(s.score, 1),
                                }
                                for s in pred.power_a.lineup
                            ]
                        ),
                        hide_index=True,
                        use_container_width=True,
                    )

            with col_r:
                with st.container(border=True):
                    st.subheader(f"{tb.name_ko} lineup")
                    st.dataframe(
                        pd.DataFrame(
                            [
                                {
                                    "batting order": s.order,
                                    "Player": s.name,
                                    "Rate": round(s.score, 1),
                                }
                                for s in pred.power_b.lineup
                            ]
                        ),
                        hide_index=True,
                        use_container_width=True,
                    )

            st.subheader("Visualization Dashboard")

            viz_tab1, viz_tab2, viz_tab3 = st.tabs(
                ["Lineup Strength", "Win Factors", "Score Probability"]
            )

            with viz_tab1:
                col_a, col_b = st.columns(2)

                with col_a:
                    with st.container(border=True):
                        st.markdown(f"#### {ta.name_ko} Lineup Strength")
                        fig_a = plot_lineup_stack(pred.power_a, ta.name_ko, "blue")
                        st.plotly_chart(fig_a, use_container_width=True)

                with col_b:
                    with st.container(border=True):
                        st.markdown(f"#### {tb.name_ko} Lineup Strength")
                        fig_b = plot_lineup_stack(pred.power_b, tb.name_ko, "red")
                        st.plotly_chart(fig_b, use_container_width=True)

            with viz_tab2:
                with st.container(border=True):
                    st.markdown("#### Match Factor Comparison")
                    fig_factors = plot_win_factors(pred)
                    st.plotly_chart(fig_factors, use_container_width=True)

            with viz_tab3:
                with st.container(border=True):
                    st.markdown("#### Poisson Score Matrix")
                    fig_heatmap = plot_poisson_heatmap(pred)
                    st.plotly_chart(fig_heatmap, use_container_width=True)

            st.subheader("Top Scoring Scenarios")

            scenario_df = pd.DataFrame(
                [
                    {
                        "A": a,
                        "B": b,
                        "Probability %": round(p * 100, 2),
                    }
                    for a, b, p in pred.score_matrix_top
                ]
            )

            st.dataframe(
                scenario_df,
                hide_index=True,
                use_container_width=True,
            )

    # ── TAB 2 ──
    with tabs[1]:
        st.subheader("8강 Monte Carlo Simulation")

        n_sim = st.slider(
            "Number of Simulations",
            1000,
            20000,
            10000,
            1000,
        )

        home_qf = st.selectbox(
            "8th Round Home Stadium Criteria Team",
            ids,
            format_func=lambda x: labels[x],
        )

        if st.button("Run Monte Carlo"):
            with st.spinner("Simulating..."):
                sim = run_monte_carlo(
                    teams,
                    ctx,
                    n_sim,
                    wmult,
                    home_qf,
                )

            st.success(f"{sim.n_sims} simulations completed")

            champ_df = pd.DataFrame(
                [
                    {
                        "TEAM": labels[k],
                        "Championship Probability %": round(v * 100, 2),
                    }
                    for k, v in sorted(
                        sim.champion_probs.items(),
                        key=lambda x: -x[1],
                    )
                ]
            )

            with st.container(border=True):
                fig_champ = plot_championship_probabilities(champ_df)
                st.plotly_chart(fig_champ, use_container_width=True)

            st.dataframe(
                champ_df,
                hide_index=True,
                use_container_width=True,
            )

            st.subheader("Round of 16 Match ups")

            for a, b in BRACKET_QF:
                st.write(f"**{labels[a]}** vs **{labels[b]}**")

    # ── TAB 3 ──
    with tabs[2]:
        st.subheader(f"Backtest ({len(BACKTEST_GAMES)} Games)")

        if st.button("Run Backtest"):
            bt = run_backtest(BACKTEST_GAMES, teams, ctx)

            c1, c2 = st.columns(2)

            with c1:
                st.metric("Winner Accuracy", f"{bt.winner_accuracy * 100:.1f}%")

            with c2:
                st.metric("Brier Score", f"{bt.brier_score:.4f}")

            st.dataframe(
                pd.DataFrame(bt.details),
                hide_index=True,
                use_container_width=True,
            )

        st.subheader("2026 WBC Verification")
        st.json(ACTUAL_2026_WBC)

        if st.button("Run Validation Report"):
            sim = run_monte_carlo(
                teams,
                ctx,
                5000,
                wmult,
                id_a,
            )

            val = validate_2026(
                sim.champion_probs,
                ACTUAL_2026_WBC,
                [],
            )

            st.write(
                f"Predicted Champion: **{labels.get(val.predicted_champion, '?')}**"
            )
            st.write(
                f"Actual Champion: **{val.actual_champion or 'Not Entered'}**"
            )
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

        st.subheader("Demo: 4 countries goal λ")

        football = pd.DataFrame(
            {
                "Country": ["Brazil", "France", "Korea", "Japan"],
                "Attack": [88, 85, 72, 74],
                "Defense": [82, 84, 78, 76],
                "λ(Goals)": [1.65, 1.58, 1.12, 1.18],
            }
        )

        st.dataframe(
            football,
            hide_index=True,
            use_container_width=True,
        )

        with st.container(border=True):
            fig_wc = plot_world_cup_lambda(football)
            st.plotly_chart(fig_wc, use_container_width=True)

        st.caption(
            "Replacing the same Poisson+MC framework with an xG-based approach will complete the World Cup module."
        )


if __name__ == "__main__":
    main()