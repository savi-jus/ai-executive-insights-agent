"""Ask the LLM to recommend executive charts from a dataset schema."""

import json
import os

from openai import OpenAI
from dotenv import load_dotenv

from tools.chart_schema import build_chart_schema, to_json_safe
from tools.charts_model import ChartRecommendations

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# System prompt with BI rules so the model picks sensible axes and aggregations
CHART_PROMPT = """You are a senior business intelligence analyst.

Review the dataset schema below and recommend the 4 most useful executive-level charts.

Rules:
- Return JSON matching the schema exactly (structured output).
- Use only column names that exist in the schema.
- Read the "hints" object in the schema (year/month pairs, metric columns, group_by_candidates).
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
- For pie charts: ONLY when x has at most 5-6 distinct groups (nunique <= 6 in schema).
  - If a category column has more than 6 groups, do NOT use pie — recommend a bar chart instead.
  - When pie is appropriate, use aggregation "count" or a numeric y.
- For simple bar charts: compare categories (e.g. FSDF Theme) with sum of a metric on y.
- For group comparison charts (clustered or stacked bar/column, or multi-series line):
  - chart_type: bar or line
  - x: primary category axis (e.g. Department, Theme, Quarter)
  - group_by: series/group column (pick from hints.group_by_candidates when possible)
  - y: numeric metric with aggregation sum or mean (or count with y omitted)
  - bar_mode: "group" for clustered/side-by-side bars, "stack" for stacked bars/columns
  - Example clustered: x=Theme, group_by=Region, y=Revenue, aggregation=sum, bar_mode=group
  - Example stacked: x=Department, group_by=Segment, y=API Hits, aggregation=sum, bar_mode=stack
  - Example multi-line: x=Month, group_by=Product Line, y=Sales, aggregation=sum, chart_type=line
- When hints.available_years spans multiple years and the chart compares categories or groups
  (NOT a single-metric trend with x=Year), set time_column to hints.year_column when available.
  Datetime columns (e.g. hire dates) are also accepted; the UI picks slider steps dynamically
  (day, week, month, etc.) based on the timeline span.
  Do NOT set time_column to Month or other non-year/non-date fields.
- For line, bar, and scatter charts with wide value ranges (min-to-max ratio ~100x or more,
  or suggests_log_scale: true on the y metric in schema): set log_y: true for readability.
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
- log_y (optional boolean; true when y spans a large range on line/bar/scatter)
- group_by (optional; series column for clustered/stacked bar or multi-line comparisons)
- bar_mode (optional; "group" or "stack" when chart_type is bar and group_by is set)
- time_column (optional; Year or datetime column; slider granularity is chosen automatically)
"""


def recommend_charts(df):
    """Build a schema from df and return parsed ChartRecommendations from the LLM."""
    schema = build_chart_schema(df)

    # Structured output parsing ensures the response matches ChartRecommendations
    response = client.responses.parse(
        model=os.getenv("CHARTING_MODEL"),
        input=CHART_PROMPT.format(schema=json.dumps(to_json_safe(schema), indent=2)),
        text_format=ChartRecommendations,
    )
    return response.output_parsed
