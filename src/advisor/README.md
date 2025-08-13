# AI Advisor & Recommendations

This directory contains systems for providing intelligent advice and recommendations based on randomness assessment results.

## Components

### `rules.py` - Rule-Based Advisor
- Purpose: Provides deterministic advice based on test results
- Method: Hard-coded rules and heuristics
- Output: Structured advice with explanations

### `llm.py` - AI-Powered Advisor
- Purpose: Provides AI-generated insights using language models
- Providers: OpenAI, Google Gemini, Anthropic Claude
- Input: Summarized test results (no raw data)
- Output: Natural language explanations and recommendations 