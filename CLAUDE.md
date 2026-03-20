# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is an Invoice Parser CLI that uses **Instructor** (structured output library) with **DeepSeek** as the LLM backend. It demonstrates two parsing approaches: structured output via Instructor SDK vs. raw JSON extraction with manual parsing.

## Commands

```bash
# Install dependencies
pip install -e .

# Parse a single invoice (quoted text)
python main.py parse "INVOICE\nInvoice Number: INV-001..."

# Batch parse all invoices from invoices.py → outputs to invoices.json
python main.py batch

# Compare both parsing methods on all invoices
python main.py compare
```

**API Key**: Set `DEEPSEEK_API_KEY` in `config.env` (gitignored, contains actual key).

## Architecture

```
client.py       - OpenAI client configured for DeepSeek, patched with instructor.from_openai()
models.py       - Pydantic models: Invoice (invoice_number, total_amount, items, due_date) and Item
parser.py       - Two parsing functions:
                   • parse_with_instructor() - returns Pydantic object directly via response_model
                   • parse_with_normal_prompt() - extracts JSON from raw response, manually constructs objects
invoices.py     - Sample invoices in various formats (English, Chinese, table, mixed Chinese-English)
main.py         - CLI entry point with subcommands: parse, batch, compare
config.env      - Environment variables (API key); gitignored
pyproject.toml  - Package metadata; exposes client, invoices, main, models, parser as top-level modules
```

The key architectural difference: Instructor handles JSON schema enforcement and validation automatically, while the normal prompt approach requires manual regex extraction and Pydantic construction.
