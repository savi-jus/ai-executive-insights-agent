"""OpenAI-backed agent that turns dataset profiles into executive insights."""

from openai import OpenAI
import os
from dotenv import load_dotenv

load_dotenv()

# Shared client; model name comes from INSIGHTS_MODEL in .env
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def generate_insights(profile, sample_data):
    """Call the LLM with a profile dict and a text preview of sample rows."""
    prompt = f"""
    You are an executive business analyst.

    Analyze this dataset profile:

    {profile}

    Sample data:
    {sample_data}

    Generate:
    1. Key business insights
    2. Risks
    3. Trends
    4. Executive summary
    """

    response = client.chat.completions.create(
        model=os.getenv("INSIGHTS_MODEL"), messages=[{"role": "user", "content": prompt}]
    )

    return response.choices[0].message.content
