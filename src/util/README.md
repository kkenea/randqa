# Utility Functions

This directory contains helper functions and utilities used throughout the randqa tool.

## Components

### `fdr.py` - False Discovery Rate Correction
- Purpose: Implements Benjamini-Hochberg FDR correction
- Use case: Multiple testing correction for statistical tests
- Input: Raw p-values array
- Output: FDR-corrected q-values

### `report.py` - Report Generation
- Purpose: Generates human-readable reports and interpretations
- Functions:
  - interpret(): Provides test result explanations
  - glossary_md(): Returns test glossary in markdown
- Use case: GUI display and report generation

### `config_help.py` - Configuration Help
- Purpose: Provides help text and warnings for configuration
- Functions:
  - CONFIG_HELP: Help text for parameters
  - config_warnings(): Generates configuration warnings
- Use case: GUI help system and validation 