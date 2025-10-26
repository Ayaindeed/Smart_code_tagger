// Mermaid Diagram Code (for GitHub, GitLab, or online Mermaid editors)
// Copy this code into any Mermaid-supported platform

graph TD
    A[User Upload] --> B[Code Analyzer]
    B --> C[Language Detection]
    C --> D[TF-IDF Processing]
    D --> E[Tag Generation]
    E --> F[Confidence Scoring]
    F --> G[Streamlit Display]
    G --> H[Export Results]
    
    I[GitHub URL] --> B
    J[Zip File] --> B
    
    B --> K[AST Parser]
    B --> L[Regex Patterns]
    
    D --> M[scikit-learn]
    E --> N[Stack Overflow API]
    
    style A fill:#e1f5fe
    style G fill:#f3e5f5
    style H fill:#e8f5e8