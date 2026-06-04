from openai import OpenAI
import os
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def generate_insights(profile, sample_data):

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
