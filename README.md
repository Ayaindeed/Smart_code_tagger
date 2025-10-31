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

- **Upload a ZIP** containing source files, **paste a GitHub repository URL**, or **try a sample project**.
- The analyzer **automatically starts** when you upload/download and will: (1) parse files, (2) extract docstrings/comments/identifiers/imports, (3) build TF-IDF features, and (4) generate tag suggestions with confidence and reasoning.
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

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   User Input    │    │  GitHub Repo    │    │   ZIP Upload    │
│                 │    │   Download      │    │                 │
└─────────┬───────┘    └─────────┬───────┘    └─────────┬───────┘
          │                      │                      │
          └──────────────────────┼──────────────────────┘
                                 │
                    ┌────────────▼────────────┐
                    │     Code Analyzer       │
                    │  ┌─────────────────┐    │
                    │  │ Language Detect │    │
                    │  │ AST Parsing     │    │
                    │  │ Regex Patterns  │    │
                    │  │ README Extract  │    │
                    │  └─────────────────┘    │
                    └────────────┬────────────┘
                                 │
                    ┌────────────▼────────────┐
                    │   TF-IDF Processor      │
                    │  ┌─────────────────┐    │
                    │  │ Text Weighting  │    │
                    │  │ Domain Boosting │    │
                    │  │ Term Extraction │    │
                    │  │ Quality Metrics │    │
                    │  └─────────────────┘    │
                    └────────────┬────────────┘
                                 │
                    ┌────────────▼────────────┐
                    │    Tag Engine           │
                    │  ┌─────────────────┐    │
                    │  │ Pattern Match   │    │
                    │  │ SO Validation   │    │
                    │  │ Confidence Score│    │
                    │  │ Categorization  │    │
                    │  └─────────────────┘    │
                    └────────────┬────────────┘
                                 │
                    ┌────────────▼────────────┐
                    │   Streamlit UI          │
                    │  ┌─────────────────┐    │
                    │  │ Results Display │    │
                    │  │ Export Options  │    │
                    │  │ Interactive UI  │    │
                    │  │ Filtering       │    │
                    │  └─────────────────┘    │
                    └─────────────────────────┘
```

**Data Flow:**
1. **Input** → User uploads ZIP, enters GitHub URL, or creates sample project
2. **Analysis** → Code Analyzer extracts text from source files, comments, and documentation  
3. **Processing** → TF-IDF Processor applies language-aware term weighting and domain boosting
4. **Tagging** → Tag Engine maps features to human-readable tags with confidence scoring
5. **Output** → Streamlit UI displays results with export options (CSV/JSON/Markdown)

## Feature Analysis Diagram

### Code Analyzer Detection Process

```
┌─────────────────────────────────────────────────────────────────────┐
│                        FEATURE ANALYSIS                            │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│     File Discovery                      Language Detection          │
│  ┌─────────────────┐                 ┌─────────────────┐            │
│  │ • README.*      │                 │ • File Extension│            │
│  │ • *.py, *.js    │────────────────▶│ • Pygments      │            │
│  │ • *.java, *.cpp │                 │ • Content Hints │            │
│  │ • Skip .git/    │                 └─────────────────┘            │
│  └─────────────────┘                           │                    │
│           │                                    ▼                    │
│           ▼                          ┌─────────────────┐            │
│     Content Extraction               │ AST Parsing     │            │
│  ┌─────────────────┐                 │ ┌─────────────┐ │            │
│  │ • Docstrings    │◀────────────────┤ │ Python AST  │ │            │
│  │ • Comments      │                 │ │ Functions   │ │            │
│  │ • Identifiers   │                 │ │ Classes     │ │            │
│  │ • Import Lists  │                 │ │ Imports     │ │            │
│  └─────────────────┘                 │ └─────────────┘ │            │
│           │                          │ ┌─────────────┐ │            │
│           │                          │ │ Regex Parse │ │            │
│           │                          │ │ JS/Java/C++ │ │            │
│           │                          │ │ Comments    │ │            │
│           │                          │ │ Functions   │ │            │
│           │                          │ └─────────────┘ │            │
│           │                          └─────────────────┘            │
│           ▼                                    │                    │
│      Text Aggregation                          │                    │
│  ┌─────────────────────────────────────────────┼────────────────────┤
│  │ Source Priority:                            ▼                    │
│  │ • Imports        → 5x weight    ┌─────────────────┐              │
│  │ • Documentation → 3x weight    │ Quality Metrics │              │
│  │ • Identifiers   → 2x weight    │ • Doc ratio     │              │
│  │ • README        → 1x weight    │ • Test presence │              │
│  └─────────────────────────────────│ • README exists │              │
│                                   │ • Perf keywords │              │
│                                   └─────────────────┘              │
└─────────────────────────────────────────────────────────────────────┘
```

## Tag Suggestion Engine Diagram

### How Tags Are Generated and Scored

```
┌─────────────────────────────────────────────────────────────────────┐
│                       TAG SUGGESTION ENGINE                        │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│    Domain Detection                  TF-IDF Processing              │
│  ┌─────────────────┐              ┌─────────────────┐              │
│  │ Technology:     │              │ • Term Frequency│              │
│  │ • flask→django  │              │ • Inverse Doc   │              │
│  │ • pandas→ML     │──────────────│   Frequency     │              │
│  │ • react→frontend│              │ • N-gram (1,2)  │              │
│  │ Domain:         │              │ • Stop Words    │              │
│  │ • model→ML      │              └─────────────────┘              │
│  │ • api→web       │                        │                      │
│  │ • test→quality  │                        ▼                      │
│  └─────────────────┘              ┌─────────────────┐              │
│           │                       │ Domain Boosting │              │
│           │                       │ ┌─────────────┐ │              │
│           │                       │ │ ML Terms    │ │              │
│           │                       │ │ × 2.0       │ │              │
│           │                       │ └─────────────┘ │              │
│           │                       │ ┌─────────────┐ │              │
│           │                       │ │ Framework   │ │              │
│           │                       │ │ × 2.5       │ │              │
│           │                       │ └─────────────┘ │              │
│           │                       │ ┌─────────────┐ │              │
│           │                       │ │ Quality     │ │              │
│           │                       │ │ × 1.5       │ │              │
│           │                       │ └─────────────┘ │              │
│           │                       └─────────────────┘              │
│           ▼                                 │                      │
│      Tag Mapping                           │                      │
│  ┌─────────────────────────────────────────┼──────────────────────┤
│  │ Category Assignment:                    ▼                      │
│  │                              ┌─────────────────┐              │
│  │    Language Tags             │ Confidence Calc │              │
│  │ • File count ratio           │ ┌─────────────┐ │              │
│  │ • Primary language           │ │ Base Score  │ │              │
│  │                              │ │ + Import    │ │              │
│  │  Technology Tags           │ │ + TF-IDF    │ │              │
│  │ • Import detection           │ │ + Patterns  │ │              │
│  │ • Framework patterns         │ │ + SO Valid  │ │              │
│  │                              │ └─────────────┘ │              │
│  │    Domain Tags               │ ┌─────────────┐ │              │
│  │ • Keyword clustering         │ │ SO Validation│ │              │
│  │ • Context analysis           │ │ • Boost +20%│ │              │
│  │                              │ │ • Similar   │ │              │
│  │    Quality Tags              │ │ • Penalty   │ │              │
│  │ • Documentation ratio        │ └─────────────┘ │              │
│  │ • Test file presence         └─────────────────┘              │
│  │ • README existence                     │                      │
│  └────────────────────────────────────────┼──────────────────────┤
│                                           ▼                      │
│    Final Ranking                ┌─────────────────┐              │
│  • Sort by confidence           │ Confidence Level │              │
│  • Category filtering           │ • Very High 80%+ │              │
│  • Max tag limits               │ • High     60%+ │              │
│  • Reasoning generation         │ • Medium   40%+ │              │
│                                 │ • Low      20%+ │              │
│                                 └─────────────────┘              │
└─────────────────────────────────────────────────────────────────────┘
```

## Codebase Insights Diagram

### Analysis Metrics and Quality Assessment

```
┌─────────────────────────────────────────────────────────────────────┐
│                       CODEBASE INSIGHTS                           │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│    Language Distribution             Quality Assessment             │
│  ┌─────────────────┐              ┌─────────────────┐              │
│  │ File Counting:  │              │ Documentation:  │              │
│  │ • .py files     │              │ • Docstrings    │              │
│  │ • .js files     │──────────────│ • Comments      │              │
│  │ • .java files   │              │ • README size   │              │
│  │ • Extensions    │              │ • Ratio calc    │              │
│  │ • Primary lang  │              └─────────────────┘              │
│  └─────────────────┘                        │                      │
│           │                                 ▼                      │
│           ▼                        ┌─────────────────┐              │
│     Domain Signals                 │ Test Detection  │              │
│  ┌─────────────────┐               │ • test*.py      │              │
│  │ Import Analysis:│               │ • *_test.js     │              │
│  │ • ML: pandas,   │               │ • pytest        │              │
│  │   sklearn, tf   │               │ • jest, mocha   │              │
│  │ • Web: flask,   │               │ • unittest      │              │
│  │   express, react│               └─────────────────┘              │
│  │ • Data: numpy,  │                         │                      │
│  │   matplotlib    │                         ▼                      │
│  │ • Mobile: react-│               ┌─────────────────┐              │
│  │   native, flutter│              │ Performance     │              │
│  └─────────────────┘               │ • async/await   │              │
│           │                        │ • cache         │              │
│           ▼                        │ • optimization  │              │
│     Term Analysis                  │ • parallel      │              │
│  ┌─────────────────────────────────┤ • benchmark     │              │
│  │ TF-IDF Top Terms:               └─────────────────┘              │
│  │                                          │                      │
│  │ 1. Extract 1000 features                ▼                      │
│  │ 2. Calculate frequencies        ┌─────────────────┐              │
│  │ 3. Apply domain boosting        │ Quality Score   │              │
│  │ 4. Sort by relevance            │ ┌─────────────┐ │              │
│  │ 5. Display top 20               │ │ README: 25% │ │              │
│  │                                 │ │ Tests:  25% │ │              │
│  │ Example Output:                 │ │ Docs:   25% │ │              │
│  │ • flask: 0.891                  │ │ Perf:   25% │ │              │
│  │ • app: 0.502                    │ └─────────────┘ │              │
│  │ • model: 0.248                  │ Total: /100     │              │
│  │ • data: 0.208                   └─────────────────┘              │
│  └─────────────────────────────────────────────────────────────────┤
│                                                                     │
│  File Structure Analysis                                         │
│  ┌─────────────────────────────────────────────────────────────────┤
│  │ Structure Patterns:                                             │
│  │ • src/ organization    → Well structured                        │
│  │ • tests/ directory     → Test coverage                          │
│  │ • docs/ folder         → Documentation                          │
│  │ • requirements.txt     → Dependency management                  │
│  │ • .gitignore           → Version control                        │
│  │ • Multiple languages   → Polyglot project                       │
│  └─────────────────────────────────────────────────────────────────┘
└─────────────────────────────────────────────────────────────────────┘
```

## Contributing

1. Fork the repo and create a branch for your feature.
2. Run and add tests for new behavior (`pytest`).
3. Submit a PR with a clear description and any design notes.
