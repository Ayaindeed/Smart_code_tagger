import requests
import json
from typing import Dict, List, Tuple, Set
from collections import defaultdict, Counter
import re


class TagSuggestionEngine:
    """Generate tag suggestions with confidence scoring and Stack Overflow validation."""
    
    # Hierarchical tag categories
    TAG_CATEGORIES = {
        'technology_stack': {
            'frontend': ['react', 'vue', 'angular', 'html', 'css', 'javascript', 'typescript', 'web-development'],
            'backend': ['node.js', 'django', 'flask', 'express', 'spring', 'api', 'server'],
            'mobile': ['android', 'ios', 'react-native', 'flutter', 'swift', 'kotlin', 'mobile'],
            'desktop': ['electron', 'qt', 'tkinter', 'wpf', 'desktop-application'],
            'database': ['sql', 'nosql', 'mongodb', 'postgresql', 'mysql', 'redis', 'database'],
            'devops': ['docker', 'kubernetes', 'ci-cd', 'aws', 'azure', 'deployment', 'infrastructure']
        },
        'domain': {
            'machine-learning': ['scikit-learn', 'tensorflow', 'pytorch', 'neural-network', 'deep-learning', 'ai'],
            'data-science': ['pandas', 'numpy', 'matplotlib', 'data-analysis', 'visualization', 'statistics'],
            'web-development': ['web', 'http', 'rest-api', 'web-framework', 'frontend', 'backend'],
            'game-development': ['game', 'unity', 'unreal-engine', 'graphics', 'simulation'],
            'security': ['cryptography', 'authentication', 'cybersecurity', 'encryption', 'security'],
            'finance': ['fintech', 'trading', 'blockchain', 'cryptocurrency', 'financial-modeling']
        },
        'purpose': {
            'library': ['library', 'package', 'module', 'framework', 'sdk'],
            'application': ['application', 'app', 'software', 'tool', 'utility'],
            'tutorial': ['tutorial', 'example', 'demo', 'learning', 'educational'],
            'research': ['research', 'academic', 'paper', 'experiment', 'prototype']
        },
        'quality': {
            'well-documented': ['documentation', 'readme', 'examples', 'tutorial'],
            'tested': ['unit-tests', 'testing', 'test-coverage', 'quality-assurance'],
            'performance': ['optimization', 'high-performance', 'efficient', 'fast'],
            'beginner-friendly': ['beginner', 'simple', 'easy', 'tutorial', 'learning'],
            'production-ready': ['production', 'stable', 'reliable', 'enterprise']
        }
    }
    
    # Stack Overflow popular tags (top 100)
    POPULAR_SO_TAGS = {
        'javascript', 'python', 'java', 'reactjs', 'html', 'css', 'node.js', 'c++',
        'typescript', 'angular', 'php', 'c#', 'vue.js', 'sql', 'mysql', 'django',
        'flask', 'express', 'mongodb', 'postgresql', 'git', 'docker', 'kubernetes',
        'aws', 'azure', 'machine-learning', 'tensorflow', 'pytorch', 'pandas',
        'numpy', 'matplotlib', 'scikit-learn', 'android', 'ios', 'swift', 'kotlin',
        'react-native', 'flutter', 'unity', 'web-scraping', 'api', 'rest',
        'json', 'xml', 'oauth', 'jwt', 'redis', 'elasticsearch', 'nginx',
        'apache', 'linux', 'ubuntu', 'windows', 'macos', 'bash', 'powershell'
    }
    
    def __init__(self):
        self.so_tags_cache = None
        self._load_so_tags()
    
    def suggest_tags(self, tfidf_results: Dict, analysis_results: Dict) -> List[Dict]:
        """Generate comprehensive tag suggestions with confidence scores."""
        
        suggestions = []
        
        # 1. Technology stack tags
        tech_tags = self._suggest_technology_tags(tfidf_results, analysis_results)
        suggestions.extend(tech_tags)
        
        # 2. Domain/purpose tags
        domain_tags = self._suggest_domain_tags(tfidf_results, analysis_results)
        suggestions.extend(domain_tags)
        
        # 3. Quality/characteristic tags
        quality_tags = self._suggest_quality_tags(tfidf_results, analysis_results)
        suggestions.extend(quality_tags)
        
        # 4. Language-specific tags
        language_tags = self._suggest_language_tags(analysis_results)
        suggestions.extend(language_tags)
        
        # 5. Remove duplicates and sort by confidence
        unique_suggestions = self._deduplicate_suggestions(suggestions)
        
        # 6. Validate against Stack Overflow tags
        validated_suggestions = self._validate_with_stackoverflow(unique_suggestions)
        
        # 7. Add reasoning and final scoring
        final_suggestions = self._add_detailed_reasoning(validated_suggestions, tfidf_results, analysis_results)
        
        return sorted(final_suggestions, key=lambda x: x['confidence'], reverse=True)[:20]
    
    def _suggest_technology_tags(self, tfidf_results: Dict, analysis_results: Dict) -> List[Dict]:
        """Suggest technology stack tags based on imports and frameworks."""
        suggestions = []
        
        # Check imports for framework detection
        imports = analysis_results.get('imports', [])
        top_terms = dict(tfidf_results.get('top_terms', []))
        
        # Framework mapping
        framework_mapping = {
            'react': 'reactjs',
            'vue': 'vue.js',
            'angular': 'angular',
            'django': 'django',
            'flask': 'flask',
            'express': 'express',
            'tensorflow': 'tensorflow',
            'pytorch': 'pytorch',
            'pandas': 'pandas',
            'numpy': 'numpy',
            'scikit-learn': 'machine-learning',
            'sklearn': 'machine-learning',
            'mongodb': 'mongodb',
            'postgresql': 'postgresql',
            'mysql': 'mysql',
            'redis': 'redis',
            'docker': 'docker',
            'kubernetes': 'kubernetes'
        }
        
        for framework, tag in framework_mapping.items():
            confidence = 0.0
            reasons = []
            
            # Check imports
            if any(framework in imp.lower() for imp in imports):
                confidence += 0.8
                reasons.append(f"Direct import detected: {framework}")
            
            # Check TF-IDF terms
            if framework in top_terms:
                confidence += top_terms[framework] * 0.5
                reasons.append(f"High frequency in code: {framework}")
            
            # Check variations
            variations = [framework.replace('-', ''), framework.replace('_', '')]
            for var in variations:
                if var in top_terms:
                    confidence += top_terms[var] * 0.3
                    reasons.append(f"Related term found: {var}")
            
            if confidence > 0.2:
                suggestions.append({
                    'tag': tag,
                    'confidence': min(confidence, 1.0),
                    'category': 'technology',
                    'reasons': reasons
                })
        
        return suggestions
    
    def _suggest_domain_tags(self, tfidf_results: Dict, analysis_results: Dict) -> List[Dict]:
        """Suggest domain-specific tags based on content analysis."""
        suggestions = []
        
        domain_signals = tfidf_results.get('domain_signals', {})
        top_terms = dict(tfidf_results.get('top_terms', []))
        
        # Domain keyword mapping
        domain_keywords = {
            'machine-learning': ['model', 'training', 'prediction', 'neural', 'algorithm', 'dataset'],
            'web-development': ['server', 'client', 'api', 'http', 'request', 'response', 'route'],
            'data-science': ['analysis', 'visualization', 'statistics', 'chart', 'graph', 'correlation'],
            'game-development': ['game', 'player', 'sprite', 'collision', 'physics', 'render'],
            'mobile-development': ['mobile', 'app', 'native', 'platform', 'device', 'screen'],
            'security': ['security', 'encryption', 'authentication', 'password', 'token', 'crypto'],
            'database': ['query', 'table', 'schema', 'migration', 'index', 'relationship'],
            'devops': ['deployment', 'infrastructure', 'monitoring', 'pipeline', 'automation']
        }
        
        for domain, keywords in domain_keywords.items():
            confidence = 0.0
            reasons = []
            
            # Check direct domain signals
            if domain.replace('-', '_') in domain_signals:
                confidence += 0.6
                reasons.extend(domain_signals[domain.replace('-', '_')][:2])
            
            # Check keyword presence in top terms
            keyword_matches = 0
            for keyword in keywords:
                if keyword in top_terms:
                    keyword_matches += 1
                    confidence += top_terms[keyword] * 0.2
            
            if keyword_matches > 0:
                reasons.append(f"Domain keywords found: {keyword_matches}/{len(keywords)}")
            
            if confidence > 0.3:
                suggestions.append({
                    'tag': domain,
                    'confidence': min(confidence, 1.0),
                    'category': 'domain',
                    'reasons': reasons
                })
        
        return suggestions
    
    def _suggest_quality_tags(self, tfidf_results: Dict, analysis_results: Dict) -> List[Dict]:
        """Suggest quality and characteristic tags."""
        suggestions = []
        
        quality_metrics = tfidf_results.get('quality_metrics', {})
        
        # Documentation quality
        if quality_metrics.get('documentation_ratio', 0) > 0.5:
            suggestions.append({
                'tag': 'well-documented',
                'confidence': min(quality_metrics['documentation_ratio'], 1.0),
                'category': 'quality',
                'reasons': [f"High documentation ratio: {quality_metrics['documentation_ratio']:.2f}"]
            })
        
        # README presence
        if quality_metrics.get('has_readme', False):
            suggestions.append({
                'tag': 'documentation',
                'confidence': 0.6,
                'category': 'quality',
                'reasons': ['README file present']
            })
        
        # Test presence
        if quality_metrics.get('has_tests', False):
            suggestions.append({
                'tag': 'testing',
                'confidence': 0.7,
                'category': 'quality',
                'reasons': ['Test files detected']
            })
        
        # Performance focus
        if quality_metrics.get('performance_focused', False):
            suggestions.append({
                'tag': 'performance',
                'confidence': 0.6,
                'category': 'quality',
                'reasons': ['Performance-related terms found']
            })
        
        return suggestions
    
    def _suggest_language_tags(self, analysis_results: Dict) -> List[Dict]:
        """Suggest programming language tags."""
        suggestions = []
        
        languages = analysis_results.get('languages', {})
        total_files = sum(languages.values())
        
        # Language tag mapping
        language_mapping = {
            'python': 'python',
            'javascript': 'javascript',
            'java': 'java',
            'cpp': 'c++',
            'csharp': 'c#',
            'go': 'go',
            'rust': 'rust',
            'swift': 'swift',
            'kotlin': 'kotlin'
        }
        
        for lang, count in languages.items():
            if lang in language_mapping and count > 0:
                confidence = count / total_files
                
                suggestions.append({
                    'tag': language_mapping[lang],
                    'confidence': confidence,
                    'category': 'language',
                    'reasons': [f'{count} {lang} files ({confidence*100:.1f}% of codebase)']
                })
        
        return suggestions
    
    def _deduplicate_suggestions(self, suggestions: List[Dict]) -> List[Dict]:
        """Remove duplicate suggestions and merge similar ones."""
        seen_tags = {}
        
        for suggestion in suggestions:
            tag = suggestion['tag']
            if tag in seen_tags:
                # Merge with existing suggestion
                existing = seen_tags[tag]
                existing['confidence'] = max(existing['confidence'], suggestion['confidence'])
                existing['reasons'].extend(suggestion['reasons'])
            else:
                seen_tags[tag] = suggestion
        
        return list(seen_tags.values())
    
    def _validate_with_stackoverflow(self, suggestions: List[Dict]) -> List[Dict]:
        """Validate and boost tags that exist on Stack Overflow."""
        for suggestion in suggestions:
            tag = suggestion['tag']
            
            # Check against popular SO tags
            if tag in self.POPULAR_SO_TAGS:
                suggestion['confidence'] *= 1.2  # Boost popular tags
                suggestion['so_validated'] = True
                suggestion['reasons'].append('Popular Stack Overflow tag')
            else:
                # Check if similar tag exists
                similar_tag = self._find_similar_so_tag(tag)
                if similar_tag:
                    suggestion['so_similar'] = similar_tag
                    suggestion['reasons'].append(f'Similar to SO tag: {similar_tag}')
                else:
                    suggestion['confidence'] *= 0.8  # Slight penalty for unknown tags
                    suggestion['so_validated'] = False
        
        return suggestions
    
    def _find_similar_so_tag(self, tag: str) -> str:
        """Find similar Stack Overflow tags."""
        # Simple similarity check
        tag_lower = tag.lower()
        
        for so_tag in self.POPULAR_SO_TAGS:
            if tag_lower in so_tag or so_tag in tag_lower:
                return so_tag
            
            # Check for partial matches
            if len(tag_lower) > 3 and len(so_tag) > 3:
                if tag_lower[:4] == so_tag[:4] or tag_lower[-4:] == so_tag[-4:]:
                    return so_tag
        
        return None
    
    def _add_detailed_reasoning(self, suggestions: List[Dict], tfidf_results: Dict, analysis_results: Dict) -> List[Dict]:
        """Add detailed reasoning and final confidence adjustment."""
        
        for suggestion in suggestions:
            # Add context information
            suggestion['context'] = {
                'primary_language': analysis_results.get('primary_language', 'unknown'),
                'total_files': sum(analysis_results.get('languages', {}).values()),
                'has_readme': bool(analysis_results.get('readme_content', '').strip())
            }
            
            # Adjust confidence based on context
            if suggestion['category'] == 'language':
                # Language tags get high confidence if they're the primary language
                if suggestion['tag'].lower() in analysis_results.get('primary_language', '').lower():
                    suggestion['confidence'] *= 1.3
            
            elif suggestion['category'] == 'technology':
                # Technology tags get boosted if multiple signals support them
                if len(suggestion['reasons']) > 2:
                    suggestion['confidence'] *= 1.2
            
            # Ensure confidence stays in valid range
            suggestion['confidence'] = min(max(suggestion['confidence'], 0.0), 1.0)
            
            # Add confidence level description
            conf = suggestion['confidence']
            if conf >= 0.8:
                suggestion['confidence_level'] = 'Very High'
            elif conf >= 0.6:
                suggestion['confidence_level'] = 'High'
            elif conf >= 0.4:
                suggestion['confidence_level'] = 'Medium'
            elif conf >= 0.2:
                suggestion['confidence_level'] = 'Low'
            else:
                suggestion['confidence_level'] = 'Very Low'
        
        return suggestions
    
    def _load_so_tags(self):
        """Load Stack Overflow tags (cached for performance)."""
        # In a real implementation, you might fetch this from SO API
        # For now, we use the predefined popular tags
        self.so_tags_cache = self.POPULAR_SO_TAGS
    
    def get_tag_explanation(self, tag: str) -> str:
        """Get explanation for what a tag represents."""
        explanations = {
            'machine-learning': 'Projects involving algorithms that learn from data',
            'web-development': 'Applications or libraries for web-based software',
            'data-science': 'Projects focused on data analysis and insights',
            'well-documented': 'Code with comprehensive documentation and examples',
            'testing': 'Projects with automated tests and quality assurance',
            'performance': 'Code optimized for speed and efficiency',
            'beginner-friendly': 'Projects suitable for learning and new developers'
        }
        
        return explanations.get(tag, f'Tag representing {tag} technology or concept')