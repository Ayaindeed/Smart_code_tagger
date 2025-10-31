import streamlit as st
import pandas as pd
import tempfile
import os
import zipfile
import requests
import shutil
from pathlib import Path
from urllib.parse import urlparse
import streamlit.components.v1 as components

from src.code_tagger.analyzer import CodeAnalyzer
from src.code_tagger.tfidf_processor import LanguageSpecificTFIDF
from src.code_tagger.tag_engine import TagSuggestionEngine


def download_github_repo(repo_url: str) -> str:
    """Download a GitHub repository as a ZIP file."""
    try:
        # Parse the GitHub URL
        parsed_url = urlparse(repo_url)
        if 'github.com' not in parsed_url.netloc:
            raise ValueError("URL must be a GitHub repository")
        
        # Extract owner and repo name from URL
        path_parts = parsed_url.path.strip('/').split('/')
        if len(path_parts) < 2:
            raise ValueError("Invalid GitHub repository URL")
        
        owner, repo_name = path_parts[0], path_parts[1]
        
        # Construct download URL for ZIP file
        download_url = f"https://github.com/{owner}/{repo_name}/archive/refs/heads/main.zip"
        
        # Try main branch first, then master
        response = requests.get(download_url, timeout=30)
        if response.status_code == 404:
            download_url = f"https://github.com/{owner}/{repo_name}/archive/refs/heads/master.zip"
            response = requests.get(download_url, timeout=30)
        
        if response.status_code != 200:
            raise ValueError(f"Failed to download repository: HTTP {response.status_code}")
        
        # Save the ZIP file
        temp_dir = Path("temp_repos")
        temp_dir.mkdir(exist_ok=True)
        zip_path = temp_dir / f"{owner}_{repo_name}.zip"
        
        with open(zip_path, 'wb') as f:
            f.write(response.content)
        
        return str(zip_path)
    
    except Exception as e:
        st.error(f"Error downloading repository: {str(e)}")
        return None


def create_sample_repository(repo_type: str) -> str:
    """Create sample repository for demonstration."""
    
    sample_dir = Path(f"sample_repos/{repo_type}")
    sample_dir.mkdir(parents=True, exist_ok=True)
    
    if repo_type == "sample_ml_project":
        # Create ML project structure
        (sample_dir / "train.py").write_text("""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt

def load_data(filename):
    \"\"\"Load and preprocess the dataset.
    
    Args:
        filename (str): Path to the CSV data file
        
    Returns:
        pd.DataFrame: Preprocessed dataset
    \"\"\"
    data = pd.read_csv(filename)
    # Remove missing values
    data = data.dropna()
    return data

def train_model(X_train, y_train):
    \"\"\"Train a Random Forest classifier.
    
    Args:
        X_train: Training features
        y_train: Training labels
        
    Returns:
        sklearn.ensemble.RandomForestClassifier: Trained model
    \"\"\"
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test):
    \"\"\"Evaluate model performance.\"\"\"
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    print(f"Model accuracy: {accuracy:.2f}")
    return accuracy

if __name__ == "__main__":
    # Load data
    data = load_data("dataset.csv")
    
    # Prepare features and labels
    X = data.drop('target', axis=1)
    y = data['target']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Train and evaluate
    model = train_model(X_train, y_train)
    accuracy = evaluate_model(model, X_test, y_test)
""")
        
        (sample_dir / "README.md").write_text("""
# Machine Learning Classification Project

This project implements a Random Forest classifier for binary classification tasks.

## Features

- Data preprocessing and cleaning
- Model training with cross-validation  
- Performance evaluation with multiple metrics
- Visualization of results

## Requirements

- pandas
- numpy
- scikit-learn
- matplotlib

## Usage

```python
python train.py
```

## Model Performance

The Random Forest classifier achieves >85% accuracy on the test dataset.
""")

    elif repo_type == "sample_web_api":
        # Create web API project
        (sample_dir / "app.py").write_text("""
from flask import Flask, request, jsonify
from flask_cors import CORS
import sqlite3
import jwt
import hashlib
from datetime import datetime, timedelta

app = Flask(__name__)
CORS(app)
app.config['SECRET_KEY'] = 'your-secret-key'

def init_database():
    \"\"\"Initialize SQLite database with user table.\"\"\"
    conn = sqlite3.connect('app.db')
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            email TEXT UNIQUE NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    conn.commit()
    conn.close()

@app.route('/api/auth/register', methods=['POST'])
def register_user():
    \"\"\"Register a new user account.\"\"\"
    data = request.get_json()
    
    if not data or not data.get('username') or not data.get('password'):
        return jsonify({'error': 'Missing username or password'}), 400
    
    # Hash password for security
    password_hash = hashlib.sha256(data['password'].encode()).hexdigest()
    
    try:
        conn = sqlite3.connect('app.db')
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO users (username, password_hash, email)
            VALUES (?, ?, ?)
        ''', (data['username'], password_hash, data.get('email', '')))
        conn.commit()
        
        return jsonify({'message': 'User registered successfully'}), 201
    
    except sqlite3.IntegrityError:
        return jsonify({'error': 'Username already exists'}), 409
    finally:
        conn.close()

@app.route('/api/auth/login', methods=['POST'])
def login_user():
    \"\"\"Authenticate user and return JWT token.\"\"\"
    data = request.get_json()
    
    if not data or not data.get('username') or not data.get('password'):
        return jsonify({'error': 'Missing credentials'}), 400
    
    # Verify password
    password_hash = hashlib.sha256(data['password'].encode()).hexdigest()
    
    conn = sqlite3.connect('app.db')
    cursor = conn.cursor()
    cursor.execute('''
        SELECT id FROM users 
        WHERE username = ? AND password_hash = ?
    ''', (data['username'], password_hash))
    
    user = cursor.fetchone()
    conn.close()
    
    if user:
        # Generate JWT token
        token = jwt.encode({
            'user_id': user[0],
            'exp': datetime.utcnow() + timedelta(hours=24)
        }, app.config['SECRET_KEY'], algorithm='HS256')
        
        return jsonify({'token': token}), 200
    else:
        return jsonify({'error': 'Invalid credentials'}), 401

@app.route('/api/users/<int:user_id>', methods=['GET'])
def get_user(user_id):
    \"\"\"Get user information by ID.\"\"\"
    # TODO: Add JWT token validation
    
    conn = sqlite3.connect('app.db')
    cursor = conn.cursor()
    cursor.execute('''
        SELECT id, username, email, created_at 
        FROM users WHERE id = ?
    ''', (user_id,))
    
    user = cursor.fetchone()
    conn.close()
    
    if user:
        return jsonify({
            'id': user[0],
            'username': user[1], 
            'email': user[2],
            'created_at': user[3]
        })
    else:
        return jsonify({'error': 'User not found'}), 404

if __name__ == '__main__':
    init_database()
    app.run(debug=True, host='0.0.0.0', port=5000)
""")
        
        (sample_dir / "requirements.txt").write_text("""
flask
flask-cors
pyjwt
sqlite3
""")

    elif repo_type == "sample_react_app":
        # Create React project structure
        (sample_dir / "App.js").write_text("""
import React, { useState, useEffect } from 'react';
import axios from 'axios';
import './App.css';

function App() {
  const [todos, setTodos] = useState([]);
  const [newTodo, setNewTodo] = useState('');
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    fetchTodos();
  }, []);

  const fetchTodos = async () => {
    try {
      setLoading(true);
      const response = await axios.get('/api/todos');
      setTodos(response.data);
    } catch (error) {
      console.error('Error fetching todos:', error);
    } finally {
      setLoading(false);
    }
  };

  const addTodo = async (e) => {
    e.preventDefault();
    if (!newTodo.trim()) return;

    try {
      const response = await axios.post('/api/todos', {
        title: newTodo,
        completed: false
      });
      setTodos([...todos, response.data]);
      setNewTodo('');
    } catch (error) {
      console.error('Error adding todo:', error);
    }
  };

  const toggleTodo = async (id) => {
    try {
      const todo = todos.find(t => t.id === id);
      const response = await axios.put(`/api/todos/${id}`, {
        ...todo,
        completed: !todo.completed
      });
      setTodos(todos.map(t => t.id === id ? response.data : t));
    } catch (error) {
      console.error('Error updating todo:', error);
    }
  };

  const deleteTodo = async (id) => {
    try {
      await axios.delete(`/api/todos/${id}`);
      setTodos(todos.filter(t => t.id !== id));
    } catch (error) {
      console.error('Error deleting todo:', error);
    }
  };

  if (loading) {
    return <div className="loading">Loading todos...</div>;
  }

  return (
    <div className="App">
      <header className="App-header">
        <h1>Todo List Manager</h1>
      </header>
      
      <main className="container">
        <form onSubmit={addTodo} className="todo-form">
          <input
            type="text"
            value={newTodo}
            onChange={(e) => setNewTodo(e.target.value)}
            placeholder="Add a new todo..."
            className="todo-input"
          />
          <button type="submit" className="add-button">
            Add Todo
          </button>
        </form>

        <div className="todo-list">
          {todos.length === 0 ? (
            <p className="empty-message">No todos yet. Add one above!</p>
          ) : (
            todos.map(todo => (
              <div key={todo.id} className={`todo-item ${todo.completed ? 'completed' : ''}`}>
                <input
                  type="checkbox"
                  checked={todo.completed}
                  onChange={() => toggleTodo(todo.id)}
                  className="todo-checkbox"
                />
                <span className="todo-title">{todo.title}</span>
                <button 
                  onClick={() => deleteTodo(todo.id)}
                  className="delete-button"
                >
                  Delete
                </button>
              </div>
            ))
          )}
        </div>
      </main>
    </div>
  );
}

export default App;
""")
        
        (sample_dir / "package.json").write_text("""
{
  "name": "todo-list-app",
  "version": "1.0.0",
  "description": "A modern React todo list application",
  "main": "index.js",
  "scripts": {
    "start": "react-scripts start",
    "build": "react-scripts build",
    "test": "react-scripts test",
    "eject": "react-scripts eject"
  },
  "dependencies": {
    "react": "^18.2.0",
    "react-dom": "^18.2.0",
    "react-scripts": "5.0.1",
    "axios": "^1.4.0"
  },
  "devDependencies": {
    "@testing-library/jest-dom": "^5.16.4",
    "@testing-library/react": "^13.3.0"
  }
}
""")
    
    return str(sample_dir)


def display_analysis_results(analysis_results, tfidf_results, tag_suggestions, 
                           min_confidence, max_tags, show_reasoning,
                           include_language, include_technology, include_domain, include_quality):
    """Display comprehensive analysis results with enhanced UI."""
    
    # Filter suggestions based on user preferences
    filtered_suggestions = []
    category_filters = {
        'language': include_language,
        'technology': include_technology, 
        'domain': include_domain,
        'quality': include_quality
    }
    
    for suggestion in tag_suggestions:
        if suggestion['confidence'] >= min_confidence:
            category = suggestion['category']
            if category_filters.get(category, True):
                filtered_suggestions.append(suggestion)
    
    filtered_suggestions = filtered_suggestions[:max_tags]
    
    # Quick stats at the top
    col_stats1, col_stats2, col_stats3, col_stats4 = st.columns(4)
    
    with col_stats1:
        st.metric("📊 Total Tags", len(filtered_suggestions))
    
    with col_stats2:
        high_conf_tags = len([s for s in filtered_suggestions if s['confidence'] >= 0.7])
        st.metric("⭐ High Confidence", high_conf_tags)
    
    with col_stats3:
        languages = analysis_results.get('languages', {})
        st.metric("🔤 Languages", len(languages))
    
    with col_stats4:
        total_files = sum(languages.values()) if languages else 0
        st.metric("📁 Files Analyzed", total_files)
    
    # Main results with tabs
    tab1, tab2, tab3 = st.tabs(["🏷️ Tag Suggestions", "📊 Codebase Insights", "🔍 Detailed Analysis"])
    
    with tab1:
        if not filtered_suggestions:
            st.warning("🚫 No tags found matching your criteria. Try lowering the confidence threshold or adjusting category filters.")
            return
        
        # Group tags by category for better organization
        tags_by_category = {}
        for suggestion in filtered_suggestions:
            category = suggestion['category']
            if category not in tags_by_category:
                tags_by_category[category] = []
            tags_by_category[category].append(suggestion)
        
        # Display tags by category
        category_icons = {
            'language': '🔤',
            'technology': '⚙️', 
            'domain': '🎯',
            'quality': '✅'
        }
        
        for category, tags in tags_by_category.items():
            with st.expander(f"{category_icons.get(category, '🏷️')} {category.title()} Tags ({len(tags)})", expanded=True):
                
                # Create a wider grid layout for better space utilization
                cols = st.columns(3)  # Changed from 2 to 3 columns
                
                for i, suggestion in enumerate(tags):
                    with cols[i % 3]:  # Changed from 2 to 3
                        # Create a card-like display for each tag
                        confidence_color = get_confidence_color(suggestion['confidence'])
                        confidence_emoji = "🟢" if suggestion['confidence'] >= 0.8 else "🟡" if suggestion['confidence'] >= 0.6 else "🟡" if suggestion['confidence'] >= 0.4 else "🔴"
                        
                        st.markdown(f"""
                        <div style="background: white; padding: 1rem; border-radius: 8px; border-left: 4px solid {confidence_color}; margin: 0.5rem 0; box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
                            <h4 style="margin: 0; color: #1f1f1f;">{confidence_emoji} {suggestion['tag']}</h4>
                            <p style="margin: 0.5rem 0; color: #666; font-size: 0.9rem;">
                                <strong>Confidence:</strong> {suggestion['confidence']:.1%} ({suggestion['confidence_level']})
                            </p>
                        """, unsafe_allow_html=True)
                        
                        # Validation badges
                        if suggestion.get('so_validated'):
                            st.markdown("✅ **Stack Overflow Validated**")
                        elif suggestion.get('so_similar'):
                            st.markdown(f"🔍 **Similar:** `{suggestion['so_similar']}`")
                        
                        # Show reasoning if enabled
                        if show_reasoning and suggestion.get('reasons'):
                            with st.expander("💡 Why this tag?", expanded=False):
                                for reason in suggestion['reasons']:
                                    st.markdown(f"• {reason}")
                        
                        st.markdown("</div>", unsafe_allow_html=True)
    
    with tab2:
        # Use full width for better visibility
        st.markdown("### 📈 Language Distribution")
        
        if languages:
            # Create two columns but give more space
            col_chart, col_table = st.columns([3, 2])
            
            with col_chart:
                # Create a more visual language display
                df_lang = pd.DataFrame(list(languages.items()), columns=['Language', 'Files'])
                df_lang = df_lang.sort_values('Files', ascending=False)
                
                # Bar chart for languages
                st.bar_chart(df_lang.set_index('Language'))
            
            with col_table:
                # Detailed table
                st.dataframe(df_lang, use_container_width=True, hide_index=True)
        else:
            st.info("No language information available")
        
        st.markdown("---")  # Add separator
        
        # Quality metrics section with full width
        st.markdown("### ✅ Quality Metrics")
        
        quality_metrics = tfidf_results.get('quality_metrics', {})
        
        # Quality score visualization
        doc_ratio = quality_metrics.get('documentation_ratio', 0)
        has_tests = quality_metrics.get('has_tests', False)
        has_readme = quality_metrics.get('has_readme', False)
        perf_focus = quality_metrics.get('performance_focused', False)
        
        # Calculate overall quality score
        quality_score = 0
        if doc_ratio > 0.1: quality_score += 25
        if has_tests: quality_score += 25
        if has_readme: quality_score += 25
        if perf_focus: quality_score += 25
        
        # Create metrics layout
        col_score, col_metrics1, col_metrics2 = st.columns([1, 1, 1])
        
        with col_score:
            st.metric("🎯 Overall Quality Score", f"{quality_score}/100")
        
        with col_metrics1:
            st.metric("📖 Documentation", f"{doc_ratio:.1%}")
            st.metric("📋 Has README", "✅" if has_readme else "❌")
        
        with col_metrics2:
            st.metric("🧪 Has Tests", "✅" if has_tests else "❌")
            st.metric("⚡ Performance Focus", "✅" if perf_focus else "❌")
    
    with tab3:
        # Advanced analysis with multiple sections
        detail_tab1, detail_tab2, detail_tab3 = st.tabs(["🔤 Top Terms", "🎯 Domain Signals", "📁 File Structure"])
        
        with detail_tab1:
            st.markdown("### Most Important Terms (TF-IDF)")
            top_terms = tfidf_results.get('tfidf_features', {}).get('top_terms', [])[:20]
            if top_terms:
                df_terms = pd.DataFrame(top_terms, columns=['Term', 'TF-IDF Score'])
                df_terms['TF-IDF Score'] = df_terms['TF-IDF Score'].round(3)
                
                # Create a bar chart of top terms
                st.bar_chart(df_terms.set_index('Term'))
                
                # Detailed table
                st.dataframe(df_terms, use_container_width=True, hide_index=True)
            else:
                st.info("No term analysis available")
        
        with detail_tab2:
            st.markdown("### Domain-Specific Signals")
            domain_signals = tfidf_results.get('domain_signals', {})
            
            if domain_signals:
                for domain, signals in domain_signals.items():
                    with st.expander(f"🏷️ {domain.replace('_', ' ').title()}", expanded=True):
                        for signal in signals[:5]:  # Show top 5
                            st.markdown(f"• **{signal}**")
            else:
                st.info("No specific domain signals detected")
        
        with detail_tab3:
            st.markdown("### File Structure Analysis")
            
            # Use wider layout for file analysis
            col_stats, col_analysis = st.columns([2, 3])
            
            with col_stats:
                st.markdown("**📊 File Statistics:**")
                total_files = sum(analysis_results.get('languages', {}).values())
                doc_count = len(analysis_results.get('docstrings', []))
                comment_count = len(analysis_results.get('comments', []))
                
                # Display metrics in a more compact way
                metric_col1, metric_col2, metric_col3 = st.columns(3)
                with metric_col1:
                    st.metric("Files", total_files)
                with metric_col2:
                    st.metric("Docstrings", doc_count)
                with metric_col3:
                    st.metric("Comments", comment_count)
            
            with col_analysis:
                st.markdown("**📋 Content Quality Analysis:**")
                
                # Create a more visual quality assessment
                if doc_count > 0:
                    doc_ratio = doc_count / max(total_files, 1)
                    if doc_ratio > 0.5:
                        st.success(f"📖 Excellent documentation coverage ({doc_count} docstrings, {doc_ratio:.1%} of files)")
                    elif doc_ratio > 0.2:
                        st.info(f"📖 Good documentation ({doc_count} docstrings, {doc_ratio:.1%} of files)")
                    else:
                        st.warning(f"� Limited documentation ({doc_count} docstrings, {doc_ratio:.1%} of files)")
                else:
                    st.warning("📝 No docstrings found - consider adding documentation")
                
                if comment_count > 0:
                    comment_ratio = comment_count / max(total_files, 1)
                    if comment_ratio > 2:
                        st.success(f"💬 Well commented code ({comment_count} comments)")
                    else:
                        st.info(f"💬 Some comments present ({comment_count} comments)")
                else:
                    st.info("💭 Consider adding more explanatory comments")
    
    # Export functionality
    st.markdown("---")
    st.markdown("### 📤 Export Results")
    
    col_export1, col_export2, col_export3 = st.columns(3)
    
    with col_export1:
        # Create CSV export
        if filtered_suggestions:
            csv_data = pd.DataFrame([{
                'Tag': s['tag'],
                'Category': s['category'],
                'Confidence': s['confidence'],
                'Level': s['confidence_level'],
                'SO_Validated': s.get('so_validated', False),
                'Reasons': '; '.join(s.get('reasons', []))
            } for s in filtered_suggestions])
            
            csv = csv_data.to_csv(index=False)
            st.download_button(
                "📊 Download CSV",
                csv,
                "code_tags.csv",
                "text/csv",
                help="Download tag suggestions as CSV file"
            )
    
    with col_export2:
        # Create JSON export
        if filtered_suggestions:
            import json
            json_data = json.dumps(filtered_suggestions, indent=2)
            st.download_button(
                "📋 Download JSON",
                json_data,
                "code_tags.json",
                "application/json",
                help="Download detailed results as JSON"
            )
    
    with col_export3:
        # Create markdown export
        if filtered_suggestions:
            md_content = "# Code Tags Analysis\n\n"
            for suggestion in filtered_suggestions:
                md_content += f"## {suggestion['tag']}\n"
                md_content += f"- **Category:** {suggestion['category']}\n"
                md_content += f"- **Confidence:** {suggestion['confidence']:.1%}\n"
                if suggestion.get('so_validated'):
                    md_content += "- **Stack Overflow Validated:** ✅\n"
                md_content += "\n"
            
            st.download_button(
                "📝 Download Markdown",
                md_content,
                "code_tags.md",
                "text/markdown",
                help="Download as Markdown file"
            )


def get_confidence_color(confidence):
    """Return color based on confidence level."""
    if confidence >= 0.8:
        return "green"
    elif confidence >= 0.6:
        return "orange" 
    elif confidence >= 0.4:
        return "yellow"
    else:
        return "red"


def perform_analysis(codebase_path, repo_name, min_confidence, max_tags, show_reasoning,
                    include_language, include_technology, include_domain, include_quality):
    """Perform the complete analysis and display results."""
    # Enhanced progress tracking with better UX
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    try:
        # Initialize components
        status_text.text("🔧 Initializing AI components...")
        progress_bar.progress(10)
        
        analyzer = CodeAnalyzer()
        tfidf_processor = LanguageSpecificTFIDF()
        tag_engine = TagSuggestionEngine()
        
        # Step 1: Analyze codebase structure and content
        status_text.text("📁 Scanning codebase structure...")
        progress_bar.progress(25)
        analysis_results = analyzer.analyze_codebase(codebase_path)
        
        # Step 2: Process with TF-IDF
        status_text.text("🔤 Processing code patterns with TF-IDF...")
        progress_bar.progress(60)
        tfidf_results = tfidf_processor.process_codebase_analysis(analysis_results)
        
        # Step 3: Generate tag suggestions
        status_text.text("🏷️ Generating intelligent tag suggestions...")
        progress_bar.progress(85)
        tag_suggestions = tag_engine.suggest_tags(tfidf_results, analysis_results)
        
        # Complete
        status_text.text("✅ Analysis complete!")
        progress_bar.progress(100)
        
        # Clear progress indicators
        import time
        time.sleep(1)
        progress_bar.empty()
        status_text.empty()
        
        # Display results with centered layout for better readability
        st.markdown("---")
        
        # Create a centered container for analytics - FULL WIDTH
        analytics_col1, analytics_main, analytics_col2 = st.columns([0.5, 7, 0.5])
        
        with analytics_main:
            st.markdown("## 📊 Analysis Results")
            display_analysis_results(analysis_results, tfidf_results, tag_suggestions,
                                   min_confidence, max_tags, show_reasoning,
                                   include_language, include_technology, include_domain, include_quality)
            st.markdown('</div>', unsafe_allow_html=True)
        
        return True
        
    except Exception as e:
        st.error(f"❌ Analysis failed: {str(e)}")
        with st.expander("🔍 Error Details"):
            st.exception(e)
        return False
    
    finally:
        # Clean up temporary file
        if codebase_path and (codebase_path.endswith('.zip') or 'temp_repos' in str(codebase_path)):
            try:
                os.unlink(codebase_path)
            except:
                pass


# Page configuration
st.set_page_config(
    layout="wide", 
    page_title="Smart Code Tagger", 
    page_icon="🏷️",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
.main-header {
    text-align: center;
    padding: 1rem 0;
    background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
    color: white;
    border-radius: 10px;
    margin-bottom: 2rem;
}
.step-card {
    background: #f8f9fa;
    padding: 1.5rem;
    border-radius: 10px;
    border-left: 4px solid #667eea;
    margin: 1rem 0;
}
.upload-section {
    background: #f8f9fa;
    padding: 2rem;
    border-radius: 15px;
    box-shadow: 0 2px 10px rgba(0,0,0,0.1);
    margin: 1rem 0;
    border: 1px solid #e9ecef;
}
.upload-section h4 {
    color: #495057 !important;
    margin-bottom: 1rem;
}
.upload-section p {
    color: #6c757d !important;
    line-height: 1.5;
}
.demo-card {
    background: #e3f2fd;
    padding: 1rem;
    border-radius: 8px;
    margin: 0.5rem 0;
    border: 1px solid #90caf9;
}
.demo-card strong {
    color: #1565c0 !important;
}
.demo-card small {
    color: #424242 !important;
}
.success-banner {
    background: #d4edda;
    border: 1px solid #c3e6cb;
    border-radius: 8px;
    padding: 1rem;
    color: #155724;
    margin: 1rem 0;
}
.analytics-container {
    background: #ffffff;
    padding: 2rem;
    border-radius: 15px;
    box-shadow: 0 4px 20px rgba(0,0,0,0.08);
    margin: 1rem 0;
    border: 1px solid #e9ecef;
}
.analytics-container h2 {
    text-align: center;
    color: #495057;
    margin-bottom: 2rem;
}
.analytics-container h3 {
    color: #6c757d;
    margin-top: 2rem;
    margin-bottom: 1rem;
}
.tag-card {
    background: #f8f9fa;
    padding: 1rem;
    border-radius: 8px;
    margin: 0.5rem 0;
    border-left: 4px solid #667eea;
}
.quality-metric {
    background: #fff3cd;
    padding: 1rem;
    border-radius: 8px;
    margin: 0.5rem 0;
    border-left: 4px solid #ffc107;
}
</style>
""", unsafe_allow_html=True)

# Main header
st.markdown("""
<div class="main-header">
    <h1>🏷️ Smart Code Tagger</h1>
    <p>AI-Powered Code Documentation & Tag Suggestion Tool</p>
</div>
""", unsafe_allow_html=True)

# Introduction with clear value proposition
st.markdown("""
### What This Tool Does

Transform your codebase analysis with intelligent tag suggestions powered by **TF-IDF** and **machine learning**:

- **Code Analysis**: Automatically scans your code for patterns, libraries, and frameworks
- **Smart Tags**: Generates relevant tags based on actual code content (not just filenames)
- **Stack Overflow Validation**: Cross-references suggestions with real developer tags
- **Confidence Scoring**: Shows how confident the AI is about each suggestion
""")

# Sidebar for options
with st.sidebar:
    st.markdown("### Analysis Settings")
    
    with st.expander("Confidence & Display", expanded=True):
        min_confidence = st.slider(
            "Minimum confidence threshold", 
            0.0, 1.0, 0.3, 0.1,
            help="Only show tags with confidence above this threshold"
        )
        max_tags = st.number_input(
            "Maximum tags to display", 
            1, 50, 15,
            help="Limit the number of tag suggestions shown"
        )
        show_reasoning = st.checkbox(
            "Show detailed reasoning", 
            value=True,
            help="Display why each tag was suggested"
        )
    
    with st.expander("📂 Tag Categories", expanded=True):
        st.markdown("Select which types of tags to include:")
        include_language = st.checkbox("🔤 Programming Languages", value=True)
        include_technology = st.checkbox("⚙️ Technology Stack", value=True)  
        include_domain = st.checkbox("🎯 Domain/Purpose", value=True)
        include_quality = st.checkbox("✅ Quality Indicators", value=True)
    
    st.markdown("---")
    st.markdown("""
    ### 💡 How It Works
    1. **Upload** your code (ZIP, GitHub link, or try samples)
    2. **AI analyzes** code patterns, imports, and documentation
    3. **Get suggestions** with confidence scores and reasoning
    4. **Export results** for documentation or project planning
    """)

# Main interface with improved layout
st.markdown("### 📤 Choose Your Codebase")

# Create tabs for different input methods
tab1, tab2, tab3 = st.tabs(["📁 Upload ZIP", "🔗 GitHub Repository", "🧪 Try Samples"])

with tab1:
    st.markdown("""
    <div class="upload-section">
    <h4>📁 Upload ZIP File</h4>
    <p>Upload a ZIP file containing your entire codebase. The tool will analyze all supported files inside.</p>
    </div>
    """, unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader(
        "Choose a ZIP file containing your codebase",
        type=['zip'],
        help="Supported formats: .zip containing source code files"
    )
    
    if uploaded_file is not None:
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix='.zip') as tmp_file:
            tmp_file.write(uploaded_file.read())
            codebase_path = tmp_file.name
        
        repo_name = uploaded_file.name.replace('.zip', '')
        st.markdown(f"""
        <div class="success-banner">
        ✅ <strong>File uploaded successfully!</strong><br>
        📁 {uploaded_file.name} ({uploaded_file.size:,} bytes) - Starting analysis...
        </div>
        """, unsafe_allow_html=True)
        
        # Automatically start analysis
        perform_analysis(codebase_path, repo_name, min_confidence, max_tags, show_reasoning,
                       include_language, include_technology, include_domain, include_quality)

with tab2:
    st.markdown("""
    <div class="upload-section">
    <h4>🔗 GitHub Repository</h4>
    <p>Paste any public GitHub repository URL and we'll download and analyze it for you.</p>
    </div>
    """, unsafe_allow_html=True)
    
    github_url = st.text_input(
        "GitHub Repository URL",
        placeholder="https://github.com/username/repository-name",
        help="Enter the full URL to any public GitHub repository"
    )
    
    col_btn1, col_btn2 = st.columns([1, 3])
    with col_btn1:
        download_button = st.button("📥 Download & Analyze", type="secondary")
    
    if download_button and github_url:
        if 'github.com' in github_url:
            with st.spinner("Downloading repository from GitHub..."):
                codebase_path = download_github_repo(github_url)
                if codebase_path:
                    # Extract repo name from URL for display
                    repo_name = github_url.split('/')[-1] if github_url.endswith('/') else github_url.split('/')[-1]
                    st.markdown(f"""
                    <div class="success-banner">
                    ✅ <strong>Repository downloaded successfully!</strong><br>
                    🔗 {github_url}<br>
                    📁 Starting analysis...
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Automatically start analysis
                    perform_analysis(codebase_path, repo_name, min_confidence, max_tags, show_reasoning,
                                   include_language, include_technology, include_domain, include_quality)
        else:
            st.error("Please enter a valid GitHub repository URL")

with tab3:
    st.markdown("""
    <div class="upload-section">
    <h4>🧪 Try Sample Projects</h4>
    <p>Test the tool with pre-built sample projects to see how it works.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sample repositories with better descriptions
    sample_repos = {
        "🤖 Machine Learning Project": {
            "key": "sample_ml_project",
            "description": "Python ML project with scikit-learn, pandas, data preprocessing"
        },
        "🌐 Web API Server": {
            "key": "sample_web_api", 
            "description": "Flask REST API with authentication, database, CORS"
        },
        "⚛️ React Frontend": {
            "key": "sample_react_app",
            "description": "Modern React app with hooks, API calls, component structure"
        }
    }
    
    for name, info in sample_repos.items():
        with st.container():
            st.markdown(f"""
            <div class="demo-card">
            <strong>{name}</strong><br>
            <small>{info['description']}</small>
            </div>
            """, unsafe_allow_html=True)
            
            col_sample1, col_sample2 = st.columns([1, 3])
            with col_sample1:
                if st.button(f"Create {name.split()[1]}", key=f"sample_{info['key']}"):
                    with st.spinner(f"Creating {name}..."):
                        sample_path = create_sample_repository(info['key'])
                        if sample_path:
                            repo_name = name
                            st.markdown(f"""
                            <div class="success-banner">
                            ✅ <strong>Sample created!</strong><br> 
                            📁 {name} - Starting analysis...
                            </div>
                            """, unsafe_allow_html=True)
                            
                            # Automatically start analysis
                            perform_analysis(sample_path, repo_name, min_confidence, max_tags, show_reasoning,
                                           include_language, include_technology, include_domain, include_quality)

# Show helpful information when no analysis is running
st.markdown("---")
st.markdown("### 🚀 Getting Started")

col_info1, col_info2, col_info3 = st.columns(3)

with col_info1:
    st.markdown("""
    <div class="step-card">
    <h4>1️⃣ Choose Input</h4>
    <p>Upload ZIP file, paste GitHub URL, or try a sample project</p>
    </div>
                <style>
    .step-card {
        background-color: #6495ED;
    }
    </style>
    """, unsafe_allow_html=True)

with col_info2:
    st.markdown("""
    <div class="step-card">
    <h4>2️⃣ Configure</h4>
    <p>Adjust settings in the sidebar for personalized results</p>
    </div>
    """, unsafe_allow_html=True)

with col_info3:
    st.markdown("""
    <div class="step-card">
    <h4>3️⃣ Analyze</h4>
    <p>Get AI-powered tag suggestions with confidence scores</p>
    </div>
    """, unsafe_allow_html=True)

# About section
st.sidebar.markdown("---")
st.sidebar.markdown("### 💡 About Smart Code Tagger")
st.sidebar.markdown("""
**AI-powered code analysis** that suggests relevant tags based on:

🔍 **Deep Code Analysis**
- Programming languages & frameworks
- Library imports & dependencies  
- Code patterns & architecture
- Documentation quality

🏷️ **Smart Tag Generation**
- TF-IDF weighted analysis
- Stack Overflow validation
- Confidence scoring
- Category classification

📊 **Rich Insights**  
- Visual analytics
- Export capabilities
- Quality metrics
- Detailed reasoning
-----------
    Built  by Aya R. with ❤️ 
    using Streamlit & SL
""")