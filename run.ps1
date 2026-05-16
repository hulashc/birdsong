# run.ps1 — start the dev server on Windows
# Usage: .\run.ps1
# Or with auto-reload on file changes: .\run.ps1 --reload

$env:PYTHONPATH = $PSScriptRoot
.venv\Scripts\python.exe -m uvicorn api.main:app --reload @args
