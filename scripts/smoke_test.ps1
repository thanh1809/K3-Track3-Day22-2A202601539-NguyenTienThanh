$ErrorActionPreference = "Stop"

pref-lab validate data/sample_preferences.jsonl
pref-lab evaluate --config configs/local.yaml
Get-Content outputs\metrics.json
