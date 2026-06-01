import plotly.express as px


def sales_chart(df):

    fig = px.line(df, x="Date", y="Amount", title="Sales Over Time")

    return fig
