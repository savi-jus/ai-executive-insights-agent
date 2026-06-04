import json
import os

from openai import OpenAI
from dotenv import load_dotenv
from tools.charts_model import (ChartSpec, ChartRecommendations)

load_dotenv()

client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY")
)


def recommend_charts(df):

    schema = {
        "columns": list(df.columns),
        "dtypes": {
            col: str(dtype)
            for col, dtype in df.dtypes.items()
        },
        "sample_rows": json.loads(
            df.head(5).to_json(orient="records", date_format="iso")
        ),
    }

    prompt = f"""
You are a senior business intelligence analyst.

Review the dataset schema below and recommend the 3 most useful
executive-level charts.

Rules:
- Return valid JSON only.
- Do not explain anything.
- Use only columns that exist.
- Supported chart types:
  - line
  - bar
  - pie
  - scatter

Schema:

{json.dumps(schema, indent=2)}

Return:

{{
  "charts": [
    {{
      "title": "...",
      "chart_type": "...",
      "x": "...",
      "y": "...",
      "reason": "..."
    }}
  ]
}}
"""

    response = client.responses.parse(
        model=os.getenv("CHARTING_MODEL"),
        input=prompt,
        text_format=ChartRecommendations,
    )
    return response.output_parsed

    #return json.loads(response.output_text)