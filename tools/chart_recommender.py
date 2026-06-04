import json
import os

from openai import OpenAI
from dotenv import load_dotenv

from tools.chart_schema import build_chart_schema, to_json_safe
from tools.charts_model import ChartRecommendations

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

CHART_PROMPT = """You are a senior business intelligence analyst.

Review the dataset schema below and recommend the 3 most useful executive-level charts.

Rules:
- Return JSON matching the schema exactly (structured output).
- Use only column names that exist in the schema.
- Read the "hints" object in the schema (year/month pairs, metric columns).
- Supported chart_type values: line, bar, pie, scatter.
- NEVER use identifier columns (looks_like_id: true) as y for line, bar, or scatter.
- NEVER use aggregation "count" on a Year column for metric trends (API Hits, Users, Revenue, etc.).
- For metric trends over time (API Hits, Users, Revenue, usage, engagement):
  - chart_type: line
  - x: Year (when hints include year_month_pair) or a datetime column
  - y: the metric column (e.g. API Hits)
  - aggregation: sum
  - time_freq: ME when data is monthly
- For headcount / hiring / employee trends (row counts):
  - aggregation: count, omit y, use datetime x or Year+Month via hints
- For pie charts: x = low-cardinality category (nunique <= 12); use aggregation "count" or numeric y.
- For bar charts: compare categories (e.g. FSDF Theme) with sum of a metric on y.
- aggregation must be one of: none, count, sum, mean
- time_freq (optional): D, W, ME, QE, YE — for datetime x or Year+Month monthly series
- sum/mean require a numeric metric column (from hints.metric_columns), NOT Year or Month

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
        input=CHART_PROMPT.format(schema=json.dumps(to_json_safe(schema), indent=2)),
        text_format=ChartRecommendations,
    )
    return response.output_parsed
