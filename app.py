import streamlit as st
import pandas as pd

from tools.data_loader import profile_data
from agents.insight_agent import generate_insights
from tools.charts_renderer import render_chart
from tools.chart_recommender import recommend_charts
from tools.chart_validator import validate_chart

st.title("Executive Insights AI Agent")

uploaded_file = st.file_uploader("Upload CSV or Excel", type=["csv", "xlsx"])

def main(df):
    # Show dataframe
    st.write(df.head())

    # Profile data
    profile = profile_data(df)

    # Generate insights
    insights = generate_insights(profile, df.head(20).to_string())

    # Display insights
    st.write(insights)

    # Recommend charts
    chart_recommendations = recommend_charts(df)
    
    for chart in chart_recommendations.charts:
        st.subheader(chart.title)
        st.caption(chart.reason)

        if validate_chart(df, chart):

            fig = render_chart(
                df,
                chart
            )

            if fig:
                st.plotly_chart(
                    fig,
                    use_container_width=True
                )


if uploaded_file:
    # Load dataframe
    if uploaded_file.name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)

    main(df)   

else:
    st.info("Please upload a file to begin analysis.")
