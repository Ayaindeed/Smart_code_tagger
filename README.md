# Smart Code Tagger

AI-powered codebase analyzer and tag suggester. The tool scans source code, README files, comments and identifiers, then uses a language-aware TF-IDF pipeline to propose relevant tags (language, technology, domain, and quality indicators) with confidence scores. It includes a Streamlit front-end for exploration and exporting results.

## Highlights

- Multi-source analysis: docstrings, comments, identifiers, imports and README content
- Language-aware TF-IDF with domain boosting to surface meaningful terms
- Tag suggestion engine with confidence scoring and Stack Overflow validation
- Streamlit UI for interactive inspection and exporting (CSV / JSON / Markdown)
- Sample project generator for quick demos

## Repository layout

- `app.py` — Streamlit application and UI
- `src/code_tagger/` — core modules:
  - `analyzer.py` — code parsing and extraction (AST + regex)
  - `tfidf_processor.py` — TF-IDF feature extraction and domain/quality scoring
  - `tag_engine.py` — tag suggestion logic and SO validation
- `sample_repos/` — created sample projects used by the demo
- `tests/` — basic unit test(s) for the analysis pipeline

## Quick start (Windows PowerShell)

1. Create and activate a virtual environment:

```powershell
python -m venv .venv; .\.venv\Scripts\Activate.ps1
```

2. Install dependencies:

```powershell
pip install -r requirements.txt
```

3. Run the Streamlit app:

```powershell
streamlit run app.py
```

Open the URL shown by Streamlit in your browser. Use the UI to upload a ZIP, paste a GitHub repo URL, or try one of the sample projects.

## How to use

- Upload a ZIP containing source files or paste a public GitHub repository URL.
- The analyzer will (1) parse files, (2) extract docstrings/comments/identifiers/imports, (3) build TF-IDF features, and (4) generate tag suggestions with confidence and reasoning.
- Use sidebar options to tune min-confidence, max tags, and which tag categories to include.
- Export suggestions as CSV, JSON or Markdown.

## Development

Run tests (recommended inside the virtual environment):

```powershell
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
pytest -q
```

Code modules of interest:
- `src/code_tagger/analyzer.py` — Walks the repository, detects languages, extracts docstrings/comments/imports/identifiers.
- `src/code_tagger/tfidf_processor.py` — Builds TF-IDF vectors, applies domain/quality boosts, returns top terms and quality signals.
- `src/code_tagger/tag_engine.py` — Heuristics to map TF-IDF signals to human-readable tags and boost/validate against common Stack Overflow tags.

## Sample repos (from UI)

The app can create quick sample projects (ML, Web API, React) placed under `sample_repos/`. These are helpful to try the full pipeline without uploading your own repo.

## Notes & design

- Primary language detection is based on file extensions with a Pygments fallback.
- The TF-IDF processor weights imports and documentation more heavily to favor technology and intent signals.
- Tag suggestions include reasoning and confidence level (`Very High` → `Very Low`) to help triage.

## Architecture diagram

![Architecture diagram](assets/diag.png)

## Contributing

1. Fork the repo and create a branch for your feature.
2. Run and add tests for new behavior (`pytest`).
3. Submit a PR with a clear description and any design notes.
