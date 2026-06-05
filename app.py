"""Streamlit dashboard for Executive Insights.

Upload a CSV or Excel file to profile the data, generate AI-written executive
insights, and display recommended charts in a tabbed layout.
"""

import streamlit as st
import pandas as pd

from tools.data_loader import profile_data
from agents.insight_agent import generate_insights
from tools.charts_renderer import render_chart
from tools.chart_recommender import recommend_charts
from tools.chart_schema import (
    TREND_CHART_TYPES,
    chart_display_reason,
    chart_display_title,
    supports_legend_filter,
    supports_time_slider,
    supports_trend_chart_controls,
    trend_group_options,
)
from tools.chart_validator import normalize_chart_spec, validate_chart


def _apply_trend_chart_controls(df: pd.DataFrame, chart, *, chart_key: str):
    """Render dropdowns/radio for time-series trend charts; return an updated ChartSpec."""
    control_key = f"{chart_key}-{st.session_state.file_name}"
    updates: dict = {}

    col_type, col_group, col_agg = st.columns(3)

    with col_type:
        type_index = (
            TREND_CHART_TYPES.index(chart.chart_type)
            if chart.chart_type in TREND_CHART_TYPES
            else 0
        )
        chart_type = st.selectbox(
            "Chart type",
            TREND_CHART_TYPES,
            index=type_index,
            key=f"trend-type-{control_key}",
        )
        if chart_type != chart.chart_type:
            updates["chart_type"] = chart_type

    group_options = trend_group_options(df, chart, y_column=chart.y)
    with col_group:
        default_group = (
            chart.group_by if chart.group_by in group_options else "All groups"
        )
        group_choice = st.selectbox(
            #Compare groups",
            "Compare by",
            group_options,
            index=group_options.index(default_group),
            key=f"trend-group-{control_key}",
        )
        if group_choice == "All groups":
            updates["group_by"] = None
            updates["bar_mode"] = None
        else:
            updates["group_by"] = group_choice
            effective_type = updates.get("chart_type", chart.chart_type)
            if effective_type == "bar":
                updates["bar_mode"] = chart.bar_mode or "group"

    with col_agg:
        if chart.aggregation == "count" and not chart.y:
            st.caption("Aggregation: **Total (row count)**")
        elif chart.y:
            current_agg = updates.get("aggregation", chart.aggregation)
            if current_agg not in ("sum", "mean"):
                current_agg = "sum"
            agg_choice = st.radio(
                "Aggregation",
                options=["sum", "mean"],
                format_func=lambda value: "Total" if value == "sum" else "Average",
                horizontal=True,
                index=0 if current_agg == "sum" else 1,
                key=f"trend-agg-{control_key}",
            )
            updates["aggregation"] = agg_choice
        else:
            st.caption("—")

    display_chart = chart.model_copy(update=updates) if updates else chart
    return normalize_chart_spec(df, display_chart)

# Page layout and sidebar defaults
st.set_page_config(
    page_title="Executive Insights",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for metric cards and dashboard panels
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
    """Ensure all session keys exist before the app reads them."""
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
    """Clear cached analysis when the user removes their uploaded file."""
    st.session_state.analysis_ready = False
    st.session_state.file_name = None
    st.session_state.df = None
    st.session_state.profile = None
    st.session_state.insights = None
    st.session_state.chart_recommendations = None


def load_dataframe(uploaded_file) -> pd.DataFrame:
    """Parse an uploaded file into a pandas DataFrame (CSV or Excel)."""
    if uploaded_file.name.endswith(".csv"):
        return pd.read_csv(uploaded_file)
    return pd.read_excel(uploaded_file)


def run_analysis(df: pd.DataFrame) -> None:
    """Profile the dataset, call the AI agent, and recommend charts."""
    profile = profile_data(df)

    with st.status("Analyzing dataset…", expanded=True) as status:
        st.write("Profiling data")
        st.write("Generating executive insights")
        # Pass a text preview of the first 20 rows so the LLM sees sample values
        insights = generate_insights(profile, df.head(20).to_string())

        st.write("Recommending charts")
        chart_recommendations = recommend_charts(df)
        status.update(label="Analysis complete", state="complete")

    # Persist results in session state for the dashboard tabs
    st.session_state.profile = profile
    st.session_state.insights = insights
    st.session_state.chart_recommendations = chart_recommendations
    st.session_state.analysis_ready = True


def total_missing_cells(profile: dict) -> int:
    """Sum missing-value counts across all columns from the data profile."""
    return int(sum(profile["missing_values"].values()))


def render_header() -> None:
    """Render the page title and subtitle."""
    col_title, _ = st.columns([3, 1])
    with col_title:
        st.title("Executive Insights")
        st.caption("AI-powered business intelligence from your spreadsheets")


def render_kpi_row(profile: dict) -> None:
    """Show top-level dataset metrics (rows, columns, missing data, numeric cols)."""
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
    """Validate, render, and display a single recommended chart."""
    chart = normalize_chart_spec(df, chart)

    with st.container(border=True):
        if supports_trend_chart_controls(df, chart):
            display_chart = _apply_trend_chart_controls(df, chart, chart_key=chart_key)
        else:
            display_chart = chart

        st.subheader(chart_display_title(display_chart), divider=False)
        st.caption(chart_display_reason(display_chart))

        is_valid, error = validate_chart(df, display_chart)
        if not is_valid:
            st.warning(error or "Chart could not be displayed.")
            return

        filter_period = None
        filter_granularity = None
        time_slider = supports_time_slider(df, display_chart)
        if time_slider:
            selected_label = st.select_slider(
                time_slider.slider_label,
                options=time_slider.labels,
                value=time_slider.labels[-1],
                key=f"time-slider-{chart_key}-{st.session_state.file_name}",
            )
            selected_index = time_slider.labels.index(selected_label)
            filter_period = time_slider.periods[selected_index]
            filter_granularity = time_slider.granularity

        if supports_legend_filter(df, display_chart):
            st.caption(
                "Tip: click a legend item to show or hide a series."
            )

        fig = render_chart(
            df,
            display_chart,
            filter_period=filter_period,
            filter_granularity=filter_granularity,
        )
        if fig:
            # chart_key must be unique when multiple charts appear on one page
            st.plotly_chart(fig, use_container_width=True, key=chart_key)
        else:
            st.warning("Chart could not be rendered from this specification.")


def tab_overview(df: pd.DataFrame, profile: dict, insights: str) -> None:
    """Overview tab: KPIs, a short insight preview, and one featured chart."""
    render_kpi_row(profile)

    st.markdown("##### Executive snapshot")
    preview = insights.strip()
    # Truncate long insight text at a word boundary for the overview
    if len(preview) > 600:
        preview = preview[:600].rsplit(" ", 1)[0] + "…"
    st.markdown(preview)
    st.caption("Open the **Insights** tab for the full analysis.")

    recommendations = st.session_state.chart_recommendations
    if recommendations and recommendations.charts:
        st.markdown("##### Featured chart")
        render_chart_card(df, recommendations.charts[0], chart_key="overview-chart-0")


def tab_insights(insights: str) -> None:
    """Insights tab: full AI-generated executive analysis."""
    st.markdown('<div class="dashboard-panel">', unsafe_allow_html=True)
    st.markdown(insights)
    st.markdown("</div>", unsafe_allow_html=True)


def tab_charts(df: pd.DataFrame) -> None:
    """Charts tab: recommended visualizations excluding the Overview featured chart."""
    recommendations = st.session_state.chart_recommendations
    if not recommendations or not recommendations.charts:
        st.info("No chart recommendations were returned for this dataset.")
        return

    # First chart is featured on the Overview tab — show the rest here
    charts = recommendations.charts[1:]
    if not charts:
        st.info("Only one chart was recommended; see the **Overview** tab for the featured chart.")
        return

    if len(charts) >= 2:
        left, right = st.columns(2)
        with left:
            render_chart_card(df, charts[0], chart_key="charts-tab-1")
        with right:
            render_chart_card(df, charts[1], chart_key="charts-tab-2")

    if len(charts) > 2:
        render_chart_card(df, charts[2], chart_key="charts-tab-3")
    elif len(charts) == 1:
        render_chart_card(df, charts[0], chart_key="charts-tab-1")


def tab_data(df: pd.DataFrame, profile: dict) -> None:
    """Data tab: raw preview and expandable column profile JSON."""
    st.markdown("##### Data preview")
    st.dataframe(df.head(100), use_container_width=True, hide_index=True)

    with st.expander("Column profile", expanded=False):
        st.json(profile)


def render_dashboard() -> None:
    """Build the four-tab dashboard once analysis results are in session state."""
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
    """App entry point: sidebar upload flow and main content area."""
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
            # User cleared the uploader — drop stale results
            if st.session_state.analysis_ready:
                reset_analysis()
            st.info("Upload a file to start the dashboard.")
        else:
            st.caption(f"**File:** {uploaded_file.name}")

            # New file detected: load, analyze, and rerun to show the dashboard
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
