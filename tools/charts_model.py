from pydantic import BaseModel
from typing import List


class ChartSpec(BaseModel):
    title: str
    chart_type: str
    x: str
    y: str
    reason: str


class ChartRecommendations(BaseModel):
    charts: List[ChartSpec]