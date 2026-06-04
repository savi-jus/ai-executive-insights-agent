import json
import os

from openai import OpenAI
from dotenv import load_dotenv

from tools.chart_schema import build_chart_schema
from tools.charts_model import ChartRecommendations

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

CHART_PROMPT = """You are a senior business intelligence analyst.

Review the dataset schema below and recommend the 3 most useful executive-level charts.

Rules:
- Return JSON matching the schema exactly (structured output).
- Use only column names that exist in the schema.
- Supported chart_type values: line, bar, pie, scatter.
- NEVER use identifier columns (looks_like_id: true) as y for line, bar, or scatter.
- For trends over time (headcount, hiring, growth, volume by date):
  - Set chart_type to "line"
  - Set x to the datetime column
  - Set aggregation to "count"
  - Set time_freq to "ME" (monthly) or "QE" (quarterly)
  - Omit y or set y to null
- For pie charts: x = low-cardinality category (nunique <= 12); use aggregation "count" or a numeric y.
- For bar charts: prefer aggregated data; avoid plotting raw row-level IDs.
- aggregation must be one of: none, count, sum, mean
- time_freq (optional): D, W, ME, QE, YE — only when aggregation is count/sum/mean and x is datetime
- sum/mean require a numeric column that is NOT an identifier

Dataset schema:

{schema}

Each chart object must include:
- title (executive-friendly)
- chart_type
- x
- y (optional when aggregation is count)
- reason (one sentence for executives)
- aggregation
- time_freq (optional)
"""


def recommend_charts(df):
    schema = build_chart_schema(df)

    response = client.responses.parse(
        model=os.getenv("CHARTING_MODEL"),
        input=CHART_PROMPT.format(schema=json.dumps(schema, indent=2)),
        text_format=ChartRecommendations,
    )
    return response.output_parsed
