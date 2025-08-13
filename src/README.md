# Source Code

This directory contains the main source code for the randqa tool.

## Core Files

- `main.py` - Core runner shared by API/GUI/CLI
- `web.py` - Flask API server
- `gui.py` - Streamlit web interface

## Subdirectories

- `sources/` - Random number generators (LCG, XorShift, OS Random)
- `metrics/` - SP 800-22 Statistical tests and entropy metrics
- `health/` - SP 800-90B health tests (RCT, APT)
- `ml/` - Machine learning predictability analysis
- `util/` - Utility functions and helpers
- `advisor/` - AI-powered recommendations and rule-based advice
- `tests/` - Test suite for all components 