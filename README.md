# LangGraph-playground

for learning LangGraph agent

## Claude Agent SDK demo (no LangGraph)

- Install: `python -m pip install claude-agent-sdk python-dotenv`
- Configure: set `ANTHROPIC_API_KEY` in `.env` (or environment)
- Run (default auto-runs golden dataset first): `python waiter_claude_agent_sdk_demo.py`
- Skip golden dataset: `python waiter_claude_agent_sdk_demo.py --free`
- Logs (stable): `waiter_claude_agent_sdk.log`
- Logs (archive): `logs/waiter_claude_agent_sdk_YYYYMMDD_HHMMSS.log`
- Golden dataset: uses `waiter_agent_accessment_golden_dataset_v2.json` (override with `--dataset <path>`)
