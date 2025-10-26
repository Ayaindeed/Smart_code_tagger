# Code Documentation Auto-Tagger - Architecture Diagram

## Simple Flow Diagram (for draw.io, Lucidchart, etc.)

```
User Input
   ↓
[Upload Code/GitHub URL]
   ↓
Code Analyzer
   ↓
TF-IDF Processor
   ↓
Tag Engine
   ↓
Streamlit UI
```

## Detailed Architecture (Copy-paste friendly for diagram tools)

### Components:
1. **Frontend**: Streamlit Web UI
2. **Code Analyzer**: AST Parser + Regex Patterns  
3. **TF-IDF Engine**: scikit-learn + Custom Domain Logic
4. **Tag Suggestion**: ML + Stack Overflow Validation
5. **Data Export**: CSV/JSON/Markdown

### Data Flow:
```
GitHub Repo/Zip File → Code Analyzer → Language Detection → 
TF-IDF Processing → Pattern Analysis → Tag Generation → 
Confidence Scoring → UI Display → Export Options
```

## Tech Stack Summary:
- **Backend**: Python 3.x
- **Web Framework**: Streamlit
- **ML/NLP**: scikit-learn (TF-IDF)
- **Code Analysis**: AST, Pygments, Regex
- **Data Processing**: Pandas
- **API Integration**: GitHub API, Stack Overflow
- **Export**: Multiple formats support

## Key Features:
✅ Multi-language code analysis  
✅ GitHub integration  
✅ AI-powered tag suggestions  
✅ Confidence scoring  
✅ Professional UI with custom CSS  
✅ Real-time analysis  
✅ Export functionality