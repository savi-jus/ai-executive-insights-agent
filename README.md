# Executive Insights AI Agent

An AI-powered business intelligence agent that analyzes datasets, identifies trends, detects risks, and generates executive-level insights from CSV and Excel files.

The goal of this project is to help decision-makers quickly understand business performance without manually exploring raw data.
_________________________________________________________________________________________________________________________________
## Overview

Executive Insights AI Agent combines data analytics with Large Language Models (LLMs) to transform business data into actionable insights.

Users can upload a dataset and receive:

- Automated data profiling
- KPI identification
- Trend analysis
- Executive summaries
- Business recommendations
- Natural language question answering
_________________________________________________________________________________________________________________________________
## Features
### Data Upload
- CSV support
- Excel support
### Automated Data Profiling
- Row and column counts
- Data type detection
- Missing value analysis
- Dataset summary
### AI-Powered Insights
- Business trend identification
- Risk detection
- Performance analysis
- Executive summaries
### Interactive Analysis
Ask questions such as:
- Which region performed best?
- What trends are emerging?
- What are the key risks?
- Which products are underperforming?
### Visualisations
- Line charts
- Trend charts
- KPI dashboards
- Business performance summaries
_________________________________________________________________________________________________________________________________
## Architecture
<img width="217" height="574" alt="image" src="https://github.com/user-attachments/assets/3dfc39b4-a26a-475f-810d-88693e3401d5" />
_________________________________________________________________________________________________________________________________
## Project Structure
<img width="267" height="534" alt="image" src="https://github.com/user-attachments/assets/3a65c068-8a84-4452-b8d4-6f19d174df95" />
_________________________________________________________________________________________________________________________________
## Technology Stack
### Programming Language
- Python
### Data Analytics
- Pandas
- NumPy
### AI
- OpenAI API
- GPT Models
### Frontend
- Streamlit
### Visualisation
- Plotly
- Matplotlib
### Environment Management
- Python Virtual Environment
- python-dotenv
_________________________________________________________________________________________________________________________________
## Installation

### Clone Repository
git clone https://github.com/yourusername/executive-insights-agent.git
cd executive-insights-agent

### Create Virtual Environment

#### Windows: 
</>Bash 
python -m venv .venv

Activate: </>Bash
</>Bash
.venv\Scripts\activate

#### Mac/Linux:
</>Bash
python3 -m venv .venv
source .venv/bin/activate

### Install Dependencies
</>Bash
pip install -r requirements.txt
_________________________________________________________________________________________________________________________________
## Environment Variables
Create a .env file in the root directory: 
OPENAI_API_KEY=your_api_key_here
_________________________________________________________________________________________________________________________________
## Running the Application

Start the Streamlit application:
</>Bash
streamlit run app.py

Open: http://localhost:8501
_________________________________________________________________________________________________________________________________
## Example Workflow
### Step 1
Upload a CSV or Excel dataset.

### Step 2
The application profiles the data and identifies:

- Column types
- Missing values
- Dataset dimensions

### Step 3
The AI agent analyzes the data and generates:

- Key insights
- Risks
- Opportunities
- Executive summary

### Step 4
Ask follow-up questions using natural language.
_________________________________________________________________________________________________________________________________
## Sample Executive Summary
Revenue increased 12% quarter-over-quarter, driven primarily by
enterprise customers in New South Wales.

Customer churn increased in March and April, particularly among
small business customers.

Recommendation:
Investigate customer retention initiatives for the SMB segment
and expand successful enterprise acquisition strategies.
_________________________________________________________________________________________________________________________________
## Potential Use Cases
- Executive reporting
- Business intelligence
- Sales performance analysis
- Customer analytics
- Financial reporting
- Operational insights
- Survey analysis
- Strategic planning
_________________________________________________________________________________________________________________________________
