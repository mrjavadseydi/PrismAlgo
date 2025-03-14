import logging
from src.patterns.gartley import GartleyPattern
from src.patterns.butterfly import ButterflyPattern
from src.patterns.bat import BatPattern
from src.patterns.crab import CrabPattern
from src.patterns.shark import SharkPattern
from src.patterns.cypher import CypherPattern

logger = logging.getLogger('PatternFactory')

class PatternFactory:
    """
    Factory class for creating and managing harmonic pattern detectors.
    """
    
    def __init__(self):
        """
        Initialize the pattern factory.
        """
        self.available_patterns = {
            'gartley': GartleyPattern,
            'butterfly': ButterflyPattern,
            'bat': BatPattern,
            'crab': CrabPattern,
            'shark': SharkPattern,
            'cypher': CypherPattern,
            # Add more patterns here as they are implemented
        }
        logger.info(f"Pattern factory initialized with {len(self.available_patterns)} patterns")
    
    def get_pattern(self, pattern_name, tolerance=0.05):
        """
        Get a pattern detector instance by name.
        
        Args:
            pattern_name (str): Name of the pattern (case-insensitive)
            tolerance (float): Tolerance for Fibonacci ratio matching
            
        Returns:
            BaseHarmonicPattern: Pattern detector instance
        """
        pattern_name = pattern_name.lower()
        
        if pattern_name not in self.available_patterns:
            raise ValueError(f"Pattern '{pattern_name}' not found. Available patterns: {', '.join(self.available_patterns.keys())}")
        
        pattern_class = self.available_patterns[pattern_name]
        return pattern_class(tolerance=tolerance)
    
    def get_all_patterns(self, tolerance=0.05):
        """
        Get instances of all available pattern detectors.
        
        Args:
            tolerance (float): Tolerance for Fibonacci ratio matching
            
        Returns:
            dict: Dictionary of pattern name to pattern detector instance
        """
        return {name: pattern_class(tolerance=tolerance) for name, pattern_class in self.available_patterns.items()}
    
    def list_available_patterns(self):
        """
        List all available pattern names.
        
        Returns:
            list: List of available pattern names
        """
        return list(self.available_patterns.keys()) 