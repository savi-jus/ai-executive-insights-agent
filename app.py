import streamlit as st
import pandas as pd

from tools.chart_tool import sales_chart
from tools.data_loader import profile_data
from agents.insight_agent import generate_insights

st.title("Executive Insights AI Agent")

uploaded_file = st.file_uploader("Upload CSV or Excel", type=["csv", "xlsx"])

if uploaded_file:
    # Load dataframe
    if uploaded_file.name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)

    # Show dataframe
    st.write(df.head())

    # Profile data
    profile = profile_data(df)

    # Generate insights
    insights = generate_insights(profile, df.head(20).to_string())

    # Display insights
    st.write(insights)

    # Generate chart
    fig = sales_chart(df)

    # Display chart
    st.plotly_chart(fig)

else:
    st.info("Please upload a file to begin analysis.")
