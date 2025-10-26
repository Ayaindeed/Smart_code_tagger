from src.code_tagger.analyzer import CodeAnalyzer
from src.code_tagger.tfidf_processor import LanguageSpecificTFIDF
from src.code_tagger.tag_engine import TagSuggestionEngine


def test_basic_analysis_flow():
    """Test the basic analysis pipeline with a simple Python file."""
    
    # Create a temporary test directory
    import tempfile
    import os
    from pathlib import Path
    
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create a simple Python file
        test_file = Path(temp_dir) / "test.py"
        test_file.write_text("""
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier

def train_model(data):
    '''Train a machine learning model on the provided data.
    
    Args:
        data: Training dataset
        
    Returns:
        Trained model
    '''
    model = RandomForestClassifier()
    # TODO: Add data preprocessing
    model.fit(data.drop('target', axis=1), data['target'])
    return model

class DataProcessor:
    '''Process and clean training data.'''
    
    def __init__(self):
        self.scaler = None
    
    def normalize_features(self, features):
        '''Normalize feature values for better model performance.'''
        return features / features.max()
""")
        
        # Test analyzer
        analyzer = CodeAnalyzer()
        analysis_results = analyzer.analyze_codebase(temp_dir)
        
        # Basic checks
        assert analysis_results['primary_language'] == 'python'
        assert 'pandas' in analysis_results['imports']
        assert 'sklearn' in str(analysis_results['imports'])
        assert len(analysis_results['docstrings']) > 0
        
        # Test TF-IDF processor
        tfidf_processor = LanguageSpecificTFIDF()
        tfidf_results = tfidf_processor.process_codebase_analysis(analysis_results)
        
        assert 'tfidf_features' in tfidf_results
        assert 'domain_signals' in tfidf_results
        
        # Test tag engine
        tag_engine = TagSuggestionEngine()
        tag_suggestions = tag_engine.suggest_tags(tfidf_results, analysis_results)
        
        # Should suggest machine learning related tags
        suggested_tags = [s['tag'] for s in tag_suggestions]
        assert any('machine' in tag.lower() or 'ml' in tag.lower() or 'python' in tag.lower() 
                  for tag in suggested_tags)
        
        print("✅ Basic analysis flow test passed!")
        print(f"Detected language: {analysis_results['primary_language']}")
        print(f"Found imports: {analysis_results['imports'][:5]}")
        print(f"Top suggested tags: {suggested_tags[:5]}")


if __name__ == "__main__":
    test_basic_analysis_flow()