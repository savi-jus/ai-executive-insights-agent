import streamlit as st
import pandas as pd

from tools.data_loader import profile_data
from agents.insight_agent import generate_insights
from tools.charts_renderer import render_chart
from tools.chart_recommender import recommend_charts
from tools.chart_validator import normalize_chart_spec, validate_chart

st.set_page_config(
    page_title="Executive Insights",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

DASHBOARD_CSS = """
<style>
    .block-container { padding-top: 1.25rem; padding-bottom: 2rem; }
    div[data-testid="stMetric"] {
        background-color: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 8px;
        padding: 0.65rem 0.85rem;
    }
    .dashboard-panel {
        background-color: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 8px;
        padding: 1rem 1.25rem;
        margin-bottom: 1rem;
    }
</style>
"""


def init_session_state() -> None:
    defaults = {
        "analysis_ready": False,
        "file_name": None,
        "df": None,
        "profile": None,
        "insights": None,
        "chart_recommendations": None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def reset_analysis() -> None:
    st.session_state.analysis_ready = False
    st.session_state.file_name = None
    st.session_state.df = None
    st.session_state.profile = None
    st.session_state.insights = None
    st.session_state.chart_recommendations = None


def load_dataframe(uploaded_file) -> pd.DataFrame:
    if uploaded_file.name.endswith(".csv"):
        return pd.read_csv(uploaded_file)
    return pd.read_excel(uploaded_file)


def run_analysis(df: pd.DataFrame) -> None:
    profile = profile_data(df)

    with st.status("Analyzing dataset…", expanded=True) as status:
        st.write("Profiling data")
        st.write("Generating executive insights")
        insights = generate_insights(profile, df.head(20).to_string())

        st.write("Recommending charts")
        chart_recommendations = recommend_charts(df)
        status.update(label="Analysis complete", state="complete")

    st.session_state.profile = profile
    st.session_state.insights = insights
    st.session_state.chart_recommendations = chart_recommendations
    st.session_state.analysis_ready = True


def total_missing_cells(profile: dict) -> int:
    return int(sum(profile["missing_values"].values()))


def render_header() -> None:
    col_title, _ = st.columns([3, 1])
    with col_title:
        st.title("Executive Insights")
        st.caption("AI-powered business intelligence from your spreadsheets")


def render_kpi_row(profile: dict) -> None:
    missing = total_missing_cells(profile)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Rows", f"{profile['rows']:,}")
    c2.metric("Columns", profile["columns"])
    c3.metric("Missing cells", f"{missing:,}")
    c4.metric(
        "Numeric columns",
        sum(1 for t in profile["data_types"].values() if "int" in t or "float" in t),
    )


def render_chart_card(df: pd.DataFrame, chart, *, chart_key: str) -> None:
    chart = normalize_chart_spec(df, chart)
    is_valid, error = validate_chart(df, chart)

    with st.container(border=True):
        st.subheader(chart.title, divider=False)
        st.caption(chart.reason)

        if not is_valid:
            st.warning(error or "Chart could not be displayed.")
            return

        fig = render_chart(df, chart)
        if fig:
            st.plotly_chart(fig, use_container_width=True, key=chart_key)
        else:
            st.warning("Chart could not be rendered from this specification.")


def tab_overview(df: pd.DataFrame, profile: dict, insights: str) -> None:
    render_kpi_row(profile)

    st.markdown("##### Executive snapshot")
    preview = insights.strip()
    if len(preview) > 600:
        preview = preview[:600].rsplit(" ", 1)[0] + "…"
    st.markdown(preview)
    st.caption("Open the **Insights** tab for the full analysis.")

    recommendations = st.session_state.chart_recommendations
    if recommendations and recommendations.charts:
        st.markdown("##### Featured chart")
        render_chart_card(df, recommendations.charts[0], chart_key="overview-chart-0")


def tab_insights(insights: str) -> None:
    st.markdown('<div class="dashboard-panel">', unsafe_allow_html=True)
    st.markdown(insights)
    st.markdown("</div>", unsafe_allow_html=True)


def tab_charts(df: pd.DataFrame) -> None:
    recommendations = st.session_state.chart_recommendations
    if not recommendations or not recommendations.charts:
        st.info("No chart recommendations were returned for this dataset.")
        return

    charts = recommendations.charts
    if len(charts) >= 2:
        left, right = st.columns(2)
        with left:
            render_chart_card(df, charts[0], chart_key="charts-tab-0")
        with right:
            render_chart_card(df, charts[1], chart_key="charts-tab-1")

    if len(charts) > 2:
        render_chart_card(df, charts[2], chart_key="charts-tab-2")
    elif len(charts) == 1:
        render_chart_card(df, charts[0], chart_key="charts-tab-0")


def tab_data(df: pd.DataFrame, profile: dict) -> None:
    st.markdown("##### Data preview")
    st.dataframe(df.head(100), use_container_width=True, hide_index=True)

    with st.expander("Column profile", expanded=False):
        st.json(profile)


def render_dashboard() -> None:
    df = st.session_state.df
    profile = st.session_state.profile
    insights = st.session_state.insights

    tab_overview_view, tab_insights_view, tab_charts_view, tab_data_view = st.tabs(
        ["Overview", "Insights", "Charts", "Data"]
    )

    with tab_overview_view:
        tab_overview(df, profile, insights)

    with tab_insights_view:
        tab_insights(insights)

    with tab_charts_view:
        tab_charts(df)

    with tab_data_view:
        tab_data(df, profile)


def main() -> None:
    st.markdown(DASHBOARD_CSS, unsafe_allow_html=True)
    init_session_state()
    render_header()

    with st.sidebar:
        st.markdown("### Data source")
        uploaded_file = st.file_uploader(
            "Upload CSV or Excel",
            type=["csv", "xlsx"],
            label_visibility="collapsed",
        )

        if uploaded_file is None:
            if st.session_state.analysis_ready:
                reset_analysis()
            st.info("Upload a file to start the dashboard.")
        else:
            st.caption(f"**File:** {uploaded_file.name}")

            if uploaded_file.name != st.session_state.file_name:
                st.session_state.file_name = uploaded_file.name
                st.session_state.df = load_dataframe(uploaded_file)
                run_analysis(st.session_state.df)
                st.rerun()

            if st.session_state.analysis_ready and st.button(
                "Re-run analysis", use_container_width=True
            ):
                run_analysis(st.session_state.df)
                st.rerun()

        st.divider()
        st.caption("Executive Insights AI Agent")

    if st.session_state.analysis_ready:
        render_dashboard()
    else:
        st.markdown(
            """
            <div class="dashboard-panel">
            <h4>Welcome</h4>
            <p>Upload a CSV or Excel file in the sidebar to generate KPIs,
            executive insights, and recommended charts.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )


if __name__ == "__main__":
    main()
