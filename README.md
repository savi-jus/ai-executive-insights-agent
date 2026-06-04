# Executive Insights AI Agent

An AI-powered business intelligence agent that analyzes datasets, identifies trends, detects risks, and generates executive-level insights from CSV and Excel files.

The goal of this project is to help decision-makers quickly understand business performance without manually exploring raw data.

## Overview

Executive Insights AI Agent combines data analytics with Large Language Models (LLMs) to transform business data into actionable insights.

After you upload a dataset, the app:

- Profiles the data (shape, types, missing values)
- Generates an executive analysis (insights, risks, trends, summary)
- Recommends up to three charts with explanations
- Validates chart specs against your columns and renders interactive Plotly charts

## Features

### Data upload

- CSV (`.csv`)
- Excel (`.xlsx`)

### Automated data profiling

- Row and column counts
- Data type detection
- Missing value analysis

### AI-powered insights

Powered by the OpenAI Chat Completions API (`agents/insight_agent.py`):

- Key business insights
- Risk detection
- Trend analysis
- Executive summary

### AI-recommended visualisations

Powered by the OpenAI Responses API with structured output (`tools/chart_recommender.py`):

- LLM suggests three executive-level charts from the dataset schema
- Each recommendation includes a title, chart type, axes, and rationale
- Column names are validated before rendering (`tools/chart_validator.py`)
- Supported chart types: line, bar, pie, scatter (rendered with Plotly in `tools/charts_renderer.py`)

## Architecture

<img width="217" height="574" alt="Architecture diagram" src="https://github.com/user-attachments/assets/3dfc39b4-a26a-475f-810d-88693e3401d5" />

## Project structure

```
ai-executive-insights-agent/
├── app.py                      # Streamlit entry point
├── agents/
│   └── insight_agent.py        # Executive insights via Chat Completions
├── tools/
│   ├── data_loader.py          # Dataset profiling
│   ├── charts_model.py         # Pydantic models for chart specs
│   ├── chart_recommender.py    # LLM chart recommendations
│   ├── chart_validator.py      # Column validation before render
│   └── charts_renderer.py      # Plotly chart rendering
├── .env.example                # Environment variable template
├── pyproject.toml              # Project metadata and dependencies (uv)
├── requirements.txt            # Locked dependencies for pip / deployment
├── uv.lock
└── README.md
```

<img width="267" height="534" alt="Project structure diagram" src="https://github.com/user-attachments/assets/3a65c068-8a84-4452-b8d4-6f19d174df95" />

## Technology stack

| Area | Tools |
|------|--------|
| Language | Python 3.14+ |
| Data | Pandas, NumPy |
| AI | OpenAI API (Chat Completions + Responses with structured parsing) |
| Models | Pydantic (`ChartSpec`, `ChartRecommendations`) |
| UI | Streamlit |
| Charts | Plotly |
| Config | python-dotenv |
| Packaging | [uv](https://docs.astral.sh/uv/) (`pyproject.toml`, `uv.lock`) |

## Installation

### Clone the repository

```bash
git clone https://github.com/savi-jus/ai-executive-insights-agent.git
cd ai-executive-insights-agent
```

### Option A: uv (recommended)

Requires [uv](https://docs.astral.sh/uv/) installed.

```bash
uv sync
```

### Option B: pip + virtual environment

**Windows**

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

**macOS / Linux**

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

To regenerate `requirements.txt` after dependency changes:

```bash
uv export --no-dev --no-hashes -o requirements.txt
```

## Environment variables

Copy the example file and add your keys:

```bash
cp .env.example .env
```

| Variable | Required | Description |
|----------|----------|-------------|
| `OPENAI_API_KEY` | Yes | Your OpenAI API key |
| `INSIGHTS_MODEL` | Yes | Model for executive insights (e.g. `gpt-4o-mini`) |
| `CHARTING_MODEL` | Yes | Model for chart recommendations (must support structured Responses parsing) |
| `DATABASE_URL` | No | Reserved for future use; not used by the app today |
| `DEBUG` | No | Reserved for future use |

Example `.env`:

```env
OPENAI_API_KEY=sk-your-key-here
INSIGHTS_MODEL=gpt-4o-mini
CHARTING_MODEL=gpt-4o-mini
```

Never commit `.env` to version control (it is listed in `.gitignore`).

## Running the application

With your virtual environment active:

```bash
streamlit run app.py
```

Open [http://localhost:8501](http://localhost:8501) in your browser.

## Example workflow

1. **Upload** a CSV or Excel file via the Streamlit file uploader.
2. **Preview** the first rows of the dataset.
3. **Profile** — the app computes row/column counts, dtypes, and missing values.
4. **Insights** — the insight agent returns key findings, risks, trends, and an executive summary.
5. **Charts** — the chart recommender proposes three charts; valid specs are rendered as interactive Plotly charts with titles and rationale.

## Deployment (Streamlit Community Cloud)

1. Push the project to GitHub (include `app.py`, `requirements.txt`, and all source under `agents/` and `tools/`). Do not push `.env`.
2. Sign in at [share.streamlit.io](https://share.streamlit.io) and create a **New app**.
3. Select your repository, branch, and set **Main file path** to `app.py`.
4. Under **Secrets**, add:

```toml
OPENAI_API_KEY = "sk-your-key-here"
INSIGHTS_MODEL = "gpt-4o-mini"
CHARTING_MODEL = "gpt-4o-mini"
```

5. Deploy and verify the build completes. Upload a sample file to test insights and charts.

**Note:** The project targets Python 3.14. If the host does not support it yet, lower `requires-python` in `pyproject.toml` (e.g. `>=3.12`), refresh the lockfile with `uv lock`, re-export `requirements.txt`, and redeploy.

The app has no built-in authentication. Restrict access to your deployment URL if the data is sensitive.

## Sample executive summary

> Revenue increased 12% quarter-over-quarter, driven primarily by enterprise customers in New South Wales.
>
> Customer churn increased in March and April, particularly among small business customers.
>
> **Recommendation:** Investigate customer retention initiatives for the SMB segment and expand successful enterprise acquisition strategies.

## Potential use cases

- Executive reporting
- Business intelligence
- Sales performance analysis
- Customer analytics
- Financial reporting
- Operational insights
- Survey analysis
- Strategic planning
