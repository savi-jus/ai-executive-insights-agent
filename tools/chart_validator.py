from tools.charts_model import ChartSpec

# Validate if the columns returned by the LLM match the given dataframe
def validate_chart(df, chart: ChartSpec):

    required_columns = [
        chart.x,
        chart.y
    ]

    return all(
        col in df.columns
        for col in required_columns
    )