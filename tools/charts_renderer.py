from tools.charts_model import ChartSpec

import plotly.express as px


def render_chart(df, chart: ChartSpec):

    if chart.chart_type == "line":

        fig = px.line(
            df,
            x=chart.x,
            y=chart.y,
            title=chart.title
        )

    elif chart.chart_type == "bar":

        fig = px.bar(
            df,
            x=chart.x,
            y=chart.y,
            title=chart.title
        )

    elif chart.chart_type == "scatter":

        fig = px.scatter(
            df,
            x=chart.x,
            y=chart.y,
            title=chart.title
        )

    elif chart.chart_type == "pie":

        fig = px.pie(
            df,
            names=chart.x,
            values=chart.y,
            title=chart.title
        )

    else:
        return None

    return fig