# Code Documentation Auto-TaggerTF-IDF Concept Map Generator



Automatically analyze codebases and suggest relevant tags based on actual code content, not just what developers claim.Overview



## FeaturesTurn a long document into an interactive concept graph of the most important terms and how they co-occur.



- **Multi-source analysis**: Extracts from docstrings, comments, README files, variable names, and importsFeatures

- **Language-specific TF-IDF**: Custom weighting for Python, JavaScript, Java, C++, and more- Upload or paste text

- **Smart tag suggestions**: Technology stack, domain, and purpose classification- TF-IDF top-k keywords per paragraph

- **Confidence scoring**: Shows reasoning behind each tag suggestion- Build co-occurrence network from per-paragraph keywords

- **Stack Overflow validation**: Cross-references with SO tag taxonomy- Optional KMeans clustering of keywords

- Visualize interactive graph (pyvis) embedded inside Streamlit

## Quick Start

Quickstart

1. Install dependencies:

```bash1. Create a virtual environment and install dependencies:

pip install -r requirements.txt

``````powershell

python -m venv .venv; .\.venv\Scripts\Activate.ps1

2. Run the analyzer:pip install -r requirements.txt

```bash```

streamlit run app.py

```2. Run the Streamlit app:



3. Upload a codebase (zip file or folder) and get instant tag suggestions!```powershell

streamlit run app.py

## How It Works```



1. **Code Analysis**: Parses source files to extract meaningful text contentNotes

2. **Language Detection**: Identifies programming languages and applies specific term weighting- The app uses TF-IDF from scikit-learn to extract top keywords per paragraph and constructs an undirected co-occurrence graph.

3. **TF-IDF Processing**: Analyzes term frequency patterns across different code elements- Clustering is optional and uses KMeans over keyword vectors built from paragraph TF-IDF columns.

4. **Tag Classification**: Maps analysis results to hierarchical tag categories

5. **Confidence Scoring**: Provides reasoning and confidence levels for each suggestionFiles

- `app.py` - Streamlit UI

## Sample Analysis- `src/concept_map/analysis.py` - TF-IDF and graph construction

- `src/concept_map/visualize.py` - pyvis integration

Upload any GitHub repository and see tags like:- `tests/test_analysis.py` - simple unit tests

- `machine-learning` (detected TensorFlow imports + ML terminology)

- `web-framework` (React/Vue patterns + REST API code)Next steps

- `well-documented` (high doc-to-code ratio)- Improve keyword extraction (use RAKE or spaCy noun chunks)

- `performance-critical` (optimization comments + complexity metrics)- Add weighting by TF-IDF score when building edges

- Add export (PNG/SVG/GraphML)

Perfect for GitHub repo tagging, project discovery, and code quality assessment.