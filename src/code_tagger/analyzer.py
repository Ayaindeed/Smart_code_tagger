import os
import ast
import re
import zipfile
import tempfile
from typing import Dict, List, Set, Tuple, Optional
from pathlib import Path
from collections import defaultdict

from pygments.lexers import get_lexer_for_filename, guess_lexer
from pygments.util import ClassNotFound


class CodeAnalyzer:
    """Extract meaningful text content from source code files."""
    
    # File extensions for different languages
    LANGUAGE_EXTENSIONS = {
        'python': {'.py', '.pyw'},
        'javascript': {'.js', '.jsx', '.ts', '.tsx', '.mjs'},
        'java': {'.java'},
        'cpp': {'.cpp', '.cxx', '.cc', '.c', '.hpp', '.h'},
        'csharp': {'.cs'},
        'go': {'.go'},
        'rust': {'.rs'},
        'php': {'.php'},
        'ruby': {'.rb'},
        'swift': {'.swift'},
        'kotlin': {'.kt'},
        'scala': {'.scala'}
    }
    
    # Programming language specific stop words
    LANGUAGE_STOPWORDS = {
        'python': {'def', 'class', 'import', 'from', 'if', 'else', 'elif', 'for', 'while', 'try', 'except', 'return', 'yield', 'with', 'as', 'lambda', 'pass', 'break', 'continue'},
        'javascript': {'function', 'var', 'let', 'const', 'if', 'else', 'for', 'while', 'do', 'switch', 'case', 'return', 'try', 'catch', 'finally', 'async', 'await'},
        'java': {'class', 'public', 'private', 'protected', 'static', 'void', 'int', 'String', 'boolean', 'if', 'else', 'for', 'while', 'try', 'catch', 'return', 'new', 'this'},
        'common': {'true', 'false', 'null', 'undefined', 'self', 'this', 'super', 'main', 'test', 'get', 'set', 'add', 'remove', 'update', 'create', 'delete'}
    }
    
    def __init__(self):
        self.supported_extensions = set()
        for exts in self.LANGUAGE_EXTENSIONS.values():
            self.supported_extensions.update(exts)
    
    def analyze_codebase(self, path: str) -> Dict[str, any]:
        """Main entry point: analyze a codebase directory or zip file."""
        if path.endswith('.zip'):
            return self._analyze_zip(path)
        else:
            return self._analyze_directory(path)
    
    def _analyze_zip(self, zip_path: str) -> Dict[str, any]:
        """Extract and analyze a zip file."""
        with tempfile.TemporaryDirectory() as temp_dir:
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(temp_dir)
            return self._analyze_directory(temp_dir)
    
    def _analyze_directory(self, dir_path: str) -> Dict[str, any]:
        """Analyze all relevant files in a directory."""
        results = {
            'languages': {},
            'docstrings': [],
            'comments': [],
            'identifiers': [],
            'imports': [],
            'readme_content': '',
            'file_structure': {},
            'complexity_metrics': {}
        }
        
        path = Path(dir_path)
        
        # Find all source files
        source_files = []
        for file_path in path.rglob('*'):
            if file_path.is_file() and file_path.suffix in self.supported_extensions:
                if not self._should_ignore_file(file_path):
                    source_files.append(file_path)
        
        # Analyze each file
        language_counts = defaultdict(int)
        
        for file_path in source_files:
            try:
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
                
                # Detect language
                language = self._detect_language(file_path, content)
                language_counts[language] += 1
                
                # Extract various components
                file_analysis = self._analyze_file(content, language, file_path)
                
                # Merge results
                results['docstrings'].extend(file_analysis['docstrings'])
                results['comments'].extend(file_analysis['comments'])
                results['identifiers'].extend(file_analysis['identifiers'])
                results['imports'].extend(file_analysis['imports'])
                
            except Exception as e:
                print(f"Error analyzing {file_path}: {e}")
                continue
        
        # Find README files
        readme_files = list(path.glob('README*')) + list(path.glob('readme*'))
        for readme_file in readme_files:
            try:
                with open(readme_file, 'r', encoding='utf-8', errors='ignore') as f:
                    results['readme_content'] += f.read() + '\n'
            except Exception:
                pass
        
        # Set primary language and language distribution
        results['languages'] = dict(language_counts)
        results['primary_language'] = max(language_counts.items(), key=lambda x: x[1])[0] if language_counts else 'unknown'
        
        return results
    
    def _should_ignore_file(self, file_path: Path) -> bool:
        """Check if file should be ignored (node_modules, .git, etc.)."""
        ignore_patterns = {
            'node_modules', '.git', '__pycache__', '.pytest_cache',
            'venv', '.venv', 'env', '.env', 'build', 'dist',
            '.idea', '.vscode', 'target', 'bin', 'obj'
        }
        
        for part in file_path.parts:
            if part in ignore_patterns or part.startswith('.'):
                return True
        return False
    
    def _detect_language(self, file_path: Path, content: str) -> str:
        """Detect programming language from file extension and content."""
        extension = file_path.suffix.lower()
        
        # Check extension mapping first
        for lang, exts in self.LANGUAGE_EXTENSIONS.items():
            if extension in exts:
                return lang
        
        # Fallback to pygments detection
        try:
            lexer = get_lexer_for_filename(str(file_path))
            return lexer.name.lower()
        except ClassNotFound:
            try:
                lexer = guess_lexer(content)
                return lexer.name.lower()
            except ClassNotFound:
                return 'unknown'
    
    def _analyze_file(self, content: str, language: str, file_path: Path) -> Dict[str, List[str]]:
        """Extract docstrings, comments, identifiers, and imports from a file."""
        result = {
            'docstrings': [],
            'comments': [],
            'identifiers': [],
            'imports': []
        }
        
        if language == 'python':
            result.update(self._analyze_python_file(content))
        elif language in ['javascript', 'typescript']:
            result.update(self._analyze_javascript_file(content))
        elif language == 'java':
            result.update(self._analyze_java_file(content))
        else:
            # Generic analysis for other languages
            result.update(self._analyze_generic_file(content, language))
        
        return result
    
    def _analyze_python_file(self, content: str) -> Dict[str, List[str]]:
        """Python-specific analysis using AST."""
        result = {
            'docstrings': [],
            'comments': [],
            'identifiers': [],
            'imports': []
        }
        
        try:
            tree = ast.parse(content)
            
            for node in ast.walk(tree):
                # Extract docstrings
                if isinstance(node, (ast.FunctionDef, ast.ClassDef, ast.AsyncFunctionDef)):
                    if (ast.get_docstring(node)):
                        result['docstrings'].append(ast.get_docstring(node))
                    
                    # Function/class names as identifiers
                    result['identifiers'].append(node.name)
                
                # Extract imports
                elif isinstance(node, ast.Import):
                    for alias in node.names:
                        result['imports'].append(alias.name)
                
                elif isinstance(node, ast.ImportFrom):
                    if node.module:
                        result['imports'].append(node.module)
                    for alias in node.names:
                        result['imports'].append(alias.name)
                
                # Variable names
                elif isinstance(node, ast.Name):
                    if len(node.id) > 2:  # Skip very short names
                        result['identifiers'].append(node.id)
        
        except SyntaxError:
            pass  # Invalid Python syntax, skip AST parsing
        
        # Extract comments using regex
        comments = re.findall(r'#\s*(.+)', content)
        result['comments'].extend(comments)
        
        return result
    
    def _analyze_javascript_file(self, content: str) -> Dict[str, List[str]]:
        """JavaScript/TypeScript analysis using regex patterns."""
        result = {
            'docstrings': [],
            'comments': [],
            'identifiers': [],
            'imports': []
        }
        
        # Extract single-line comments
        single_comments = re.findall(r'//\s*(.+)', content)
        result['comments'].extend(single_comments)
        
        # Extract multi-line comments and JSDoc
        multi_comments = re.findall(r'/\*\*?(.*?)\*/', content, re.DOTALL)
        for comment in multi_comments:
            clean_comment = re.sub(r'\s*\*\s*', ' ', comment).strip()
            if clean_comment:
                result['docstrings'].append(clean_comment)
        
        # Extract imports
        imports = re.findall(r'(?:import|require)\s*(?:\{[^}]+\}|\w+)?\s*from\s*[\'"]([^\'"]+)[\'"]', content)
        result['imports'].extend(imports)
        
        # Extract function names
        functions = re.findall(r'(?:function\s+(\w+)|(\w+)\s*[=:]\s*(?:function|\([^)]*\)\s*=>))', content)
        for match in functions:
            name = match[0] or match[1]
            if name and len(name) > 2:
                result['identifiers'].append(name)
        
        # Extract class names
        classes = re.findall(r'class\s+(\w+)', content)
        result['identifiers'].extend(classes)
        
        return result
    
    def _analyze_java_file(self, content: str) -> Dict[str, List[str]]:
        """Java analysis using regex patterns."""
        result = {
            'docstrings': [],
            'comments': [],
            'identifiers': [],
            'imports': []
        }
        
        # Extract JavaDoc comments
        javadocs = re.findall(r'/\*\*(.*?)\*/', content, re.DOTALL)
        for javadoc in javadocs:
            clean_javadoc = re.sub(r'\s*\*\s*', ' ', javadoc).strip()
            if clean_javadoc:
                result['docstrings'].append(clean_javadoc)
        
        # Extract single-line comments
        comments = re.findall(r'//\s*(.+)', content)
        result['comments'].extend(comments)
        
        # Extract imports
        imports = re.findall(r'import\s+(?:static\s+)?([^;]+);', content)
        result['imports'].extend(imports)
        
        # Extract class names
        classes = re.findall(r'(?:public\s+|private\s+|protected\s+)?class\s+(\w+)', content)
        result['identifiers'].extend(classes)
        
        # Extract method names
        methods = re.findall(r'(?:public\s+|private\s+|protected\s+)?(?:static\s+)?(?:\w+\s+)*(\w+)\s*\([^)]*\)\s*\{', content)
        result['identifiers'].extend([m for m in methods if len(m) > 2])
        
        return result
    
    def _analyze_generic_file(self, content: str, language: str) -> Dict[str, List[str]]:
        """Generic analysis for languages without specific parsers."""
        result = {
            'docstrings': [],
            'comments': [],
            'identifiers': [],
            'imports': []
        }
        
        # Extract common comment patterns
        # Single-line comments (// or #)
        single_comments = re.findall(r'(?://|#)\s*(.+)', content)
        result['comments'].extend(single_comments)
        
        # Multi-line comments (/* */ or """ """)
        multi_comments = re.findall(r'(?:/\*.*?\*/|""".*?""")', content, re.DOTALL)
        result['docstrings'].extend(multi_comments)
        
        # Extract identifiers (camelCase, snake_case)
        identifiers = re.findall(r'\b[a-zA-Z_][a-zA-Z0-9_]{2,}\b', content)
        result['identifiers'].extend(identifiers)
        
        return result