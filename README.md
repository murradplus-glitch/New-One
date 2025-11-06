# Triage Agent (Citizen-Facing)

This project implements a LangChain-based triage agent that accepts Urdu or
English symptom descriptions, reasons about severity, and recommends one of
three urgency levels:

- **Self-care**
- **BHU Visit** (Basic Health Unit / primary care)
- **Emergency**

The agent uses Google's Gemini models through the `langchain-google-genai`
integration and returns structured XML output containing reasoning, urgency, and
next-step guidance.

## Prerequisites

1. Python 3.9 or newer.
2. A Google Gemini API key stored in the `GOOGLE_API_KEY` environment variable.
3. Install dependencies in a virtual environment:

   ```bash
   python -m venv .venv
   source .venv/bin/activate
   pip install -e .
   ```

## Usage

You can interact with the agent from the command line once the dependencies are
installed:

```bash
export GOOGLE_API_KEY="your-api-key"
python -m triage_agent.cli "Chest pain for 10 minutes with nausea"
```

Alternatively, omit the symptom text to enter multi-line input via standard
input:

```bash
export GOOGLE_API_KEY="your-api-key"
python -m triage_agent.cli
Enter symptoms (Ctrl-D to finish):
میرے والد کو دو دن سے بخار ہے اور کھانسی ہے، سانس لینے میں مشکل نہیں۔
```

The agent returns structured XML:

```xml
<analysis>...</analysis>
<urgency>...</urgency>
<plan>...</plan>
```

## Project Structure

- `src/triage_agent/agent.py` – Core agent implementation and structured output.
- `src/triage_agent/prompt.py` – Few-shot prompt and instructions for Gemini.
- `src/triage_agent/cli.py` – Command-line interface for quick testing.

## Notes

- The `TriageAgent` expects the Gemini API to be available; network access is
  required at runtime.
- Customize `default_examples` in `prompt.py` to adjust few-shot guidance for
  local clinical protocols.
