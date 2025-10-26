import re
import numpy as np
from typing import Dict, List, Tuple, Set
from collections import Counter, defaultdict
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class LanguageSpecificTFIDF:
    """TF-IDF processor with language-specific term weighting and domain knowledge."""
    
    # Technology-specific terms with weights
    DOMAIN_WEIGHTS = {
        'machine_learning': {
            'sklearn', 'tensorflow', 'pytorch', 'keras', 'numpy', 'pandas', 'scipy',
            'neural', 'network', 'model', 'training', 'prediction', 'regression',
            'classification', 'clustering', 'feature', 'dataset', 'algorithm'
        },
        'web_development': {
            'react', 'vue', 'angular', 'express', 'flask', 'django', 'fastapi',
            'html', 'css', 'javascript', 'typescript', 'node', 'server', 'client',
            'api', 'rest', 'graphql', 'http', 'request', 'response', 'router'
        },
        'data_science': {
            'matplotlib', 'seaborn', 'plotly', 'jupyter', 'notebook', 'analysis',
            'visualization', 'statistics', 'correlation', 'distribution', 'data'
        },
        'game_development': {
            'unity', 'unreal', 'engine', 'sprite', 'collision', 'physics',
            'render', 'shader', 'texture', 'mesh', 'animation', 'game'
        },
        'mobile_development': {
            'android', 'ios', 'react-native', 'flutter', 'swift', 'kotlin',
            'mobile', 'app', 'native', 'platform', 'device'
        },
        'devops': {
            'docker', 'kubernetes', 'ci', 'cd', 'jenkins', 'github-actions',
            'aws', 'azure', 'gcp', 'deployment', 'infrastructure', 'monitoring'
        },
        'security': {
            'encryption', 'authentication', 'authorization', 'security', 'crypto',
            'hash', 'ssl', 'tls', 'vulnerability', 'sanitize', 'validate'
        },
        'database': {
            'sql', 'nosql', 'mongodb', 'postgresql', 'mysql', 'redis',
            'database', 'query', 'schema', 'migration', 'orm', 'index'
        }
    }
    
    # Quality indicators
    QUALITY_INDICATORS = {
        'well_documented': {
            'docstring', 'comment', 'readme', 'documentation', 'doc', 'example',
            'tutorial', 'guide', 'explanation', 'description'
        },
        'tested': {
            'test', 'unittest', 'pytest', 'jest', 'mocha', 'spec', 'assertion',
            'mock', 'fixture', 'coverage'
        },
        'performance': {
            'optimization', 'performance', 'speed', 'fast', 'efficient', 'cache',
            'memory', 'cpu', 'benchmark', 'profiling', 'async', 'parallel'
        },
        'error_handling': {
            'exception', 'error', 'try', 'catch', 'handling', 'validation',
            'logging', 'debug', 'traceback', 'recovery'
        }
    }
    
    # Language-specific framework patterns
    LANGUAGE_FRAMEWORKS = {
        'python': {
            'django', 'flask', 'fastapi', 'pandas', 'numpy', 'scipy',
            'sklearn', 'tensorflow', 'pytorch', 'requests', 'sqlalchemy'
        },
        'javascript': {
            'react', 'vue', 'angular', 'express', 'node', 'webpack',
            'babel', 'typescript', 'jest', 'lodash', 'axios'
        },
        'java': {
            'spring', 'hibernate', 'junit', 'maven', 'gradle', 'jackson',
            'apache', 'servlet', 'jpa', 'jdbc'
        },
        'cpp': {
            'boost', 'qt', 'opencv', 'eigen', 'cmake', 'stl',
            'thread', 'memory', 'pointer', 'template'
        }
    }
    
    def __init__(self):
        self.vectorizer = None
        self.domain_weights = self._flatten_domain_weights()
        self.quality_weights = self._flatten_quality_weights()
    
    def _flatten_domain_weights(self) -> Dict[str, str]:
        """Create term -> domain mapping."""
        mapping = {}
        for domain, terms in self.DOMAIN_WEIGHTS.items():
            for term in terms:
                mapping[term] = domain
        return mapping
    
    def _flatten_quality_weights(self) -> Dict[str, str]:
        """Create term -> quality mapping."""
        mapping = {}
        for quality, terms in self.QUALITY_INDICATORS.items():
            for term in terms:
                mapping[term] = quality
        return mapping
    
    def process_codebase_analysis(self, analysis_results: Dict) -> Dict[str, any]:
        """Process code analysis results and generate weighted TF-IDF features."""
        
        # Combine all text sources
        all_text = self._combine_text_sources(analysis_results)
        
        # Create custom TF-IDF with language-specific processing
        language = analysis_results.get('primary_language', 'unknown')
        features = self._extract_weighted_features(all_text, language, analysis_results)
        
        return {
            'tfidf_features': features,
            'text_sources': all_text,
            'language_weights': self._calculate_language_weights(analysis_results),
            'domain_signals': self._detect_domain_signals(analysis_results),
            'quality_metrics': self._calculate_quality_metrics(analysis_results)
        }
    
    def _combine_text_sources(self, analysis_results: Dict) -> Dict[str, str]:
        """Combine different text sources with appropriate preprocessing."""
        sources = {}
        
        # Docstrings and comments (high weight)
        docstrings = ' '.join(analysis_results.get('docstrings', []))
        comments = ' '.join(analysis_results.get('comments', []))
        sources['documentation'] = self._clean_text(docstrings + ' ' + comments)
        
        # README content (medium weight)
        readme = analysis_results.get('readme_content', '')
        sources['readme'] = self._clean_text(readme)
        
        # Identifiers (processed as keywords)
        identifiers = analysis_results.get('identifiers', [])
        sources['identifiers'] = ' '.join(self._process_identifiers(identifiers))
        
        # Import statements (high weight for technology detection)
        imports = analysis_results.get('imports', [])
        sources['imports'] = ' '.join(self._process_imports(imports))
        
        return sources
    
    def _clean_text(self, text: str) -> str:
        """Clean and normalize text content."""
        # Remove code-specific symbols and normalize
        text = re.sub(r'[^\w\s\-]', ' ', text)  # Keep alphanumeric, spaces, hyphens
        text = re.sub(r'\s+', ' ', text)  # Normalize whitespace
        text = text.lower().strip()
        return text
    
    def _process_identifiers(self, identifiers: List[str]) -> List[str]:
        """Process variable/function names to extract meaningful terms."""
        processed = []
        
        for identifier in identifiers:
            # Split camelCase and snake_case
            words = re.sub(r'([a-z])([A-Z])', r'\1 \2', identifier)  # camelCase
            words = words.replace('_', ' ')  # snake_case
            words = words.lower().split()
            
            # Filter out common programming terms and short words
            filtered_words = [
                word for word in words 
                if len(word) > 2 and word not in {'get', 'set', 'add', 'new', 'old', 'tmp', 'temp'}
            ]
            
            processed.extend(filtered_words)
        
        return processed
    
    def _process_imports(self, imports: List[str]) -> List[str]:
        """Extract meaningful library/framework names from imports."""
        processed = []
        
        for imp in imports:
            # Extract main library name (e.g., 'pandas.core' -> 'pandas')
            parts = imp.split('.')
            main_lib = parts[0] if parts else imp
            
            # Clean up common patterns
            main_lib = main_lib.strip('\'"')
            if main_lib and len(main_lib) > 1:
                processed.append(main_lib.lower())
        
        return processed
    
    def _extract_weighted_features(self, text_sources: Dict[str, str], language: str, analysis_results: Dict) -> Dict:
        """Extract TF-IDF features with custom weighting."""
        
        # Combine all text with source-specific weights
        weighted_text = []
        
        # Weight documentation higher
        doc_text = text_sources.get('documentation', '')
        if doc_text:
            weighted_text.extend([doc_text] * 3)  # 3x weight
        
        # Weight imports very high for technology detection
        import_text = text_sources.get('imports', '')
        if import_text:
            weighted_text.extend([import_text] * 5)  # 5x weight
        
        # README gets standard weight
        readme_text = text_sources.get('readme', '')
        if readme_text:
            weighted_text.append(readme_text)
        
        # Identifiers get medium weight
        id_text = text_sources.get('identifiers', '')
        if id_text:
            weighted_text.extend([id_text] * 2)  # 2x weight
        
        if not weighted_text:
            return {'terms': {}, 'top_terms': []}
        
        # Create custom TF-IDF
        vectorizer = TfidfVectorizer(
            max_features=1000,
            ngram_range=(1, 2),
            stop_words='english',
            min_df=1,
            max_df=0.95
        )
        
        try:
            tfidf_matrix = vectorizer.fit_transform(weighted_text)
            feature_names = vectorizer.get_feature_names_out()
            
            # Calculate mean TF-IDF scores
            mean_scores = np.mean(tfidf_matrix.toarray(), axis=0)
            
            # Create term -> score mapping
            term_scores = dict(zip(feature_names, mean_scores))
            
            # Apply domain-specific boosting
            boosted_scores = self._apply_domain_boosting(term_scores, language)
            
            # Get top terms
            top_terms = sorted(boosted_scores.items(), key=lambda x: x[1], reverse=True)[:50]
            
            return {
                'terms': boosted_scores,
                'top_terms': top_terms,
                'vectorizer': vectorizer
            }
            
        except Exception as e:
            print(f"TF-IDF extraction failed: {e}")
            return {'terms': {}, 'top_terms': []}
    
    def _apply_domain_boosting(self, term_scores: Dict[str, float], language: str) -> Dict[str, float]:
        """Apply domain-specific and language-specific boosting to TF-IDF scores."""
        boosted = term_scores.copy()
        
        for term, score in term_scores.items():
            boost_factor = 1.0
            
            # Domain-specific boosting
            if term in self.domain_weights:
                boost_factor *= 2.0
            
            # Quality indicator boosting
            if term in self.quality_weights:
                boost_factor *= 1.5
            
            # Language-specific framework boosting
            if language in self.LANGUAGE_FRAMEWORKS:
                if term in self.LANGUAGE_FRAMEWORKS[language]:
                    boost_factor *= 2.5
            
            boosted[term] = score * boost_factor
        
        return boosted
    
    def _calculate_language_weights(self, analysis_results: Dict) -> Dict[str, float]:
        """Calculate confidence weights for detected languages."""
        languages = analysis_results.get('languages', {})
        total_files = sum(languages.values())
        
        if total_files == 0:
            return {}
        
        # Normalize to percentages
        weights = {lang: count / total_files for lang, count in languages.items()}
        return weights
    
    def _detect_domain_signals(self, analysis_results: Dict) -> Dict[str, List[str]]:
        """Detect domain-specific signals from the codebase."""
        signals = defaultdict(list)
        
        # Check imports for domain signals
        imports = analysis_results.get('imports', [])
        for imp in imports:
            imp_lower = imp.lower()
            for domain, terms in self.DOMAIN_WEIGHTS.items():
                if any(term in imp_lower for term in terms):
                    signals[domain].append(f"import: {imp}")
        
        # Check text content for domain signals
        all_text = ' '.join(
            analysis_results.get('docstrings', []) +
            analysis_results.get('comments', []) +
            [analysis_results.get('readme_content', '')]
        ).lower()
        
        for domain, terms in self.DOMAIN_WEIGHTS.items():
            found_terms = [term for term in terms if term in all_text]
            if found_terms:
                signals[domain].extend([f"term: {term}" for term in found_terms[:3]])
        
        return dict(signals)
    
    def _calculate_quality_metrics(self, analysis_results: Dict) -> Dict[str, any]:
        """Calculate code quality metrics."""
        metrics = {}
        
        # Documentation ratio
        total_files = sum(analysis_results.get('languages', {}).values())
        doc_items = len(analysis_results.get('docstrings', []))
        comment_count = len(analysis_results.get('comments', []))
        
        metrics['documentation_ratio'] = (doc_items + comment_count) / max(total_files, 1)
        metrics['has_readme'] = bool(analysis_results.get('readme_content', '').strip())
        
        # Test indicators
        all_text = ' '.join(
            analysis_results.get('identifiers', []) +
            analysis_results.get('imports', [])
        ).lower()
        
        test_keywords = {'test', 'spec', 'unittest', 'pytest', 'jest', 'mocha'}
        metrics['has_tests'] = any(keyword in all_text for keyword in test_keywords)
        
        # Performance indicators
        perf_keywords = {'async', 'cache', 'performance', 'optimization', 'parallel'}
        metrics['performance_focused'] = any(keyword in all_text for keyword in perf_keywords)
        
        return metrics