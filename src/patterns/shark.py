import numpy as np
import pandas as pd
import logging
from src.patterns.base_pattern import BaseHarmonicPattern

logger = logging.getLogger('SharkPattern')

class SharkPattern(BaseHarmonicPattern):
    """
    Implementation of the Shark harmonic pattern.
    
    The Shark pattern is characterized by these Fibonacci ratios:
    - XA to AB: AB should be a 0.5-0.618 retracement of XA
    - AB to BC: BC should be a 1.13-1.618 extension of AB
    - BC to CD: CD should be a 1.618-2.24 extension of BC
    - XA to AD: AD should be a 0.886-1.13 retracement/extension of XA
    """
    
    def __init__(self, tolerance=0.05):
        """
        Initialize the Shark pattern detector.
        
        Args:
            tolerance (float): Tolerance for Fibonacci ratio matching (default: 0.05)
        """
        super().__init__("Shark", tolerance)
    
    def get_pattern_ratios(self):
        """
        Get the ideal Fibonacci ratios for the Shark pattern.
        
        Returns:
            dict: Dictionary of ratios for each leg of the pattern
        """
        return {
            'AB_XA': [0.5, 0.618],  # AB should be between 0.5 and 0.618 retracement of XA
            'BC_AB': [1.13, 1.618],  # BC should be between 1.13 and 1.618 extension of AB
            'CD_BC': [1.618, 2.24],  # CD should be between 1.618 and 2.24 extension of BC
            'AD_XA': [0.886, 1.13]  # AD should be between 0.886 retracement and 1.13 extension of XA
        }
    
    def find_patterns(self, df, swing_window=5):
        """
        Find Shark patterns in the given price data.
        
        Args:
            df (pandas.DataFrame): DataFrame with OHLCV data
            swing_window (int): Window size for detecting swings
            
        Returns:
            list: List of dictionaries containing pattern information
        """
        self.patterns_found = []
        
        # Find swing highs and lows
        swings = self.find_swings(df, window=swing_window)
        if len(swings) < 4:
            logger.info(f"Not enough swing points found. Need at least 4, got {len(swings)}")
            return self.patterns_found
        
        # Get pattern ratios
        pattern_ratios = self.get_pattern_ratios()
        
        # Iterate through swing points to find patterns
        for i in range(len(swings) - 3):
            # Define points X, A, B, C, D
            x_idx, x_price, x_type = swings[i]
            a_idx, a_price, a_type = swings[i+1]
            
            # Skip if X and A are the same type (both highs or both lows)
            if x_type == a_type:
                continue
            
            # Iterate through potential B points
            for j in range(i+2, len(swings) - 2):
                b_idx, b_price, b_type = swings[j]
                
                # Skip if A and B are the same type
                if a_type == b_type:
                    continue
                
                # Calculate XA and AB price movements
                xa_move = abs(a_price - x_price)
                ab_move = abs(b_price - a_price)
                
                # Calculate AB/XA ratio
                ab_xa_ratio = ab_move / xa_move if xa_move != 0 else 0
                
                # Check if AB/XA ratio matches the pattern
                ab_xa_valid = False
                if isinstance(pattern_ratios['AB_XA'], list):
                    ab_xa_valid = (
                        self.is_ratio_valid(ab_xa_ratio, pattern_ratios['AB_XA'][0]) or
                        self.is_ratio_valid(ab_xa_ratio, pattern_ratios['AB_XA'][1]) or
                        (ab_xa_ratio > pattern_ratios['AB_XA'][0] - self.tolerance and
                         ab_xa_ratio < pattern_ratios['AB_XA'][1] + self.tolerance)
                    )
                else:
                    ab_xa_valid = self.is_ratio_valid(ab_xa_ratio, pattern_ratios['AB_XA'])
                
                if not ab_xa_valid:
                    continue
                
                # Iterate through potential C points
                for k in range(j+1, len(swings) - 1):
                    c_idx, c_price, c_type = swings[k]
                    
                    # Skip if B and C are the same type
                    if b_type == c_type:
                        continue
                    
                    # Calculate BC price movement
                    bc_move = abs(c_price - b_price)
                    
                    # Calculate BC/AB ratio
                    bc_ab_ratio = bc_move / ab_move if ab_move != 0 else 0
                    
                    # Check if BC/AB ratio matches the pattern
                    bc_ab_valid = False
                    if isinstance(pattern_ratios['BC_AB'], list):
                        bc_ab_valid = (
                            self.is_ratio_valid(bc_ab_ratio, pattern_ratios['BC_AB'][0]) or
                            self.is_ratio_valid(bc_ab_ratio, pattern_ratios['BC_AB'][1]) or
                            (bc_ab_ratio > pattern_ratios['BC_AB'][0] - self.tolerance and
                             bc_ab_ratio < pattern_ratios['BC_AB'][1] + self.tolerance)
                        )
                    else:
                        bc_ab_valid = self.is_ratio_valid(bc_ab_ratio, pattern_ratios['BC_AB'])
                    
                    if not bc_ab_valid:
                        continue
                    
                    # Iterate through potential D points
                    for l in range(k+1, len(swings)):
                        d_idx, d_price, d_type = swings[l]
                        
                        # Skip if C and D are the same type
                        if c_type == d_type:
                            continue
                        
                        # Calculate CD and AD price movements
                        cd_move = abs(d_price - c_price)
                        ad_move = abs(d_price - a_price)
                        
                        # Calculate CD/BC and AD/XA ratios
                        cd_bc_ratio = cd_move / bc_move if bc_move != 0 else 0
                        ad_xa_ratio = ad_move / xa_move if xa_move != 0 else 0
                        
                        # Check if CD/BC ratio matches the pattern
                        cd_bc_valid = False
                        if isinstance(pattern_ratios['CD_BC'], list):
                            cd_bc_valid = (
                                self.is_ratio_valid(cd_bc_ratio, pattern_ratios['CD_BC'][0]) or
                                self.is_ratio_valid(cd_bc_ratio, pattern_ratios['CD_BC'][1]) or
                                (cd_bc_ratio > pattern_ratios['CD_BC'][0] - self.tolerance and
                                 cd_bc_ratio < pattern_ratios['CD_BC'][1] + self.tolerance)
                            )
                        else:
                            cd_bc_valid = self.is_ratio_valid(cd_bc_ratio, pattern_ratios['CD_BC'])
                        
                        # Check if AD/XA ratio matches the pattern
                        ad_xa_valid = False
                        if isinstance(pattern_ratios['AD_XA'], list):
                            ad_xa_valid = (
                                self.is_ratio_valid(ad_xa_ratio, pattern_ratios['AD_XA'][0]) or
                                self.is_ratio_valid(ad_xa_ratio, pattern_ratios['AD_XA'][1]) or
                                (ad_xa_ratio > pattern_ratios['AD_XA'][0] - self.tolerance and
                                 ad_xa_ratio < pattern_ratios['AD_XA'][1] + self.tolerance)
                            )
                        else:
                            ad_xa_valid = self.is_ratio_valid(ad_xa_ratio, pattern_ratios['AD_XA'])
                        
                        # If all ratios are valid, we found a pattern
                        if cd_bc_valid and ad_xa_valid:
                            pattern = {
                                'type': self.name,
                                'direction': 'bullish' if x_type == 'low' else 'bearish',
                                'points': {
                                    'X': {'idx': x_idx, 'price': x_price, 'date': df.index[x_idx]},
                                    'A': {'idx': a_idx, 'price': a_price, 'date': df.index[a_idx]},
                                    'B': {'idx': b_idx, 'price': b_price, 'date': df.index[b_idx]},
                                    'C': {'idx': c_idx, 'price': c_price, 'date': df.index[c_idx]},
                                    'D': {'idx': d_idx, 'price': d_price, 'date': df.index[d_idx]}
                                },
                                'ratios': {
                                    'AB/XA': ab_xa_ratio,
                                    'BC/AB': bc_ab_ratio,
                                    'CD/BC': cd_bc_ratio,
                                    'AD/XA': ad_xa_ratio
                                },
                                'completion_date': df.index[d_idx],
                                'target_price': self.calculate_target_price(x_price, a_price, d_price)
                            }
                            
                            self.patterns_found.append(pattern)
                            logger.info(f"Found {self.name} pattern: {pattern['direction']} at {pattern['completion_date']}")
        
        return self.patterns_found
    
    def calculate_target_price(self, x_price, a_price, d_price):
        """
        Calculate the target price for the pattern.
        
        Args:
            x_price (float): Price at point X
            a_price (float): Price at point A
            d_price (float): Price at point D
            
        Returns:
            float: Target price
        """
        # For Shark pattern, target is often the 0.5 retracement of XA from D
        xa_move = abs(a_price - x_price)
        target_extension = 0.5
        
        if a_price > x_price:  # Bullish pattern
            return d_price + (xa_move * target_extension)
        else:  # Bearish pattern
            return d_price - (xa_move * target_extension) 