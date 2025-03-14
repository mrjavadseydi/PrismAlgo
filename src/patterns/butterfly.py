import numpy as np
import pandas as pd
import logging
from src.patterns.base_pattern import BaseHarmonicPattern

logger = logging.getLogger('ButterflyPattern')

class ButterflyPattern(BaseHarmonicPattern):
    """
    Implementation of the Butterfly harmonic pattern.
    
    The Butterfly pattern is characterized by these Fibonacci ratios:
    - XA to AB: AB should be a 0.786 retracement of XA
    - AB to BC: BC should be a 0.382-0.886 retracement of AB
    - BC to CD: CD should be a 1.618-2.618 extension of BC
    - XA to AD: AD should be a 1.27-1.618 extension of XA
    """
    
    def __init__(self, tolerance=0.05):
        """
        Initialize the Butterfly pattern detector.
        
        Args:
            tolerance (float): Tolerance for Fibonacci ratio matching (default: 0.05)
        """
        super().__init__("Butterfly", tolerance)
    
    def get_pattern_ratios(self):
        """
        Get the ideal Fibonacci ratios for the Butterfly pattern.
        
        Returns:
            dict: Dictionary of ratios for each leg of the pattern
        """
        return {
            'AB_XA': 0.786,  # AB should be a 0.786 retracement of XA
            'BC_AB': [0.382, 0.886],  # BC should be between 0.382 and 0.886 retracement of AB
            'CD_BC': [1.618, 2.618],  # CD should be between 1.618 and 2.618 extension of BC
            'AD_XA': [1.27, 1.618]  # AD should be between 1.27 and 1.618 extension of XA
        }
    
    def find_patterns(self, df, swing_window=5):
        """
        Find Butterfly patterns in the given price data.
        
        Args:
            df (pandas.DataFrame): DataFrame with OHLCV data
            swing_window (int): Window size for detecting swings
            
        Returns:
            list: List of detected Butterfly patterns with their details
        """
        # Get swing highs and lows
        swing_highs, swing_lows = self.find_swings(df, swing_window)
        
        if len(swing_highs) < 2 or len(swing_lows) < 2:
            logger.warning("Not enough swing points to find patterns")
            return []
        
        patterns = []
        pattern_id = 0
        
        # Get pattern ratios
        ratios = self.get_pattern_ratios()
        
        # Look for bullish Butterfly (X:low, A:high, B:low, C:high, D:low)
        for i in range(len(swing_lows) - 1):  # X point (low)
            x_idx = swing_lows.iloc[i]['index']
            x_price = swing_lows.iloc[i]['price']
            
            # Find potential A points (highs) after X
            for j in range(len(swing_highs)):
                a_idx = swing_highs.iloc[j]['index']
                
                if a_idx <= x_idx:
                    continue  # A must be after X
                
                a_price = swing_highs.iloc[j]['price']
                xa_diff = a_price - x_price
                
                if xa_diff <= 0:
                    continue  # XA must be upward for bullish pattern
                
                # Find potential B points (lows) after A
                for k in range(i + 1, len(swing_lows)):
                    b_idx = swing_lows.iloc[k]['index']
                    
                    if b_idx <= a_idx:
                        continue  # B must be after A
                    
                    b_price = swing_lows.iloc[k]['price']
                    ab_diff = b_price - a_price
                    
                    if ab_diff >= 0:
                        continue  # AB must be downward for bullish pattern
                    
                    # Check AB/XA ratio
                    ab_xa_ratio = self.calculate_ratio(a_price, b_price, x_price)
                    
                    if not self.is_ratio_valid(ab_xa_ratio, ratios['AB_XA']):
                        continue
                    
                    # Find potential C points (highs) after B
                    for l in range(j + 1, len(swing_highs)):
                        c_idx = swing_highs.iloc[l]['index']
                        
                        if c_idx <= b_idx:
                            continue  # C must be after B
                        
                        c_price = swing_highs.iloc[l]['price']
                        bc_diff = c_price - b_price
                        
                        if bc_diff <= 0:
                            continue  # BC must be upward for bullish pattern
                        
                        # Check BC/AB ratio
                        bc_ab_ratio = self.calculate_ratio(b_price, c_price, a_price)
                        
                        if isinstance(ratios['BC_AB'], list):
                            if not (ratios['BC_AB'][0] <= bc_ab_ratio <= ratios['BC_AB'][1]):
                                continue
                        elif not self.is_ratio_valid(bc_ab_ratio, ratios['BC_AB']):
                            continue
                        
                        # Find potential D points (lows) after C
                        for m in range(k + 1, len(swing_lows)):
                            d_idx = swing_lows.iloc[m]['index']
                            
                            if d_idx <= c_idx:
                                continue  # D must be after C
                            
                            d_price = swing_lows.iloc[m]['price']
                            cd_diff = d_price - c_price
                            
                            if cd_diff >= 0:
                                continue  # CD must be downward for bullish pattern
                            
                            # Check CD/BC ratio
                            cd_bc_ratio = self.calculate_ratio(c_price, d_price, b_price)
                            
                            if isinstance(ratios['CD_BC'], list):
                                if not (ratios['CD_BC'][0] <= cd_bc_ratio <= ratios['CD_BC'][1]):
                                    continue
                            elif not self.is_ratio_valid(cd_bc_ratio, ratios['CD_BC']):
                                continue
                            
                            # Check AD/XA ratio
                            ad_xa_ratio = self.calculate_ratio(a_price, d_price, x_price)
                            
                            if isinstance(ratios['AD_XA'], list):
                                if not (ratios['AD_XA'][0] <= ad_xa_ratio <= ratios['AD_XA'][1]):
                                    continue
                            elif not self.is_ratio_valid(ad_xa_ratio, ratios['AD_XA']):
                                continue
                            
                            # All ratios match, we found a bullish Butterfly pattern
                            pattern_id += 1
                            patterns.append({
                                'id': pattern_id,
                                'type': 'Butterfly',
                                'direction': 'bullish',
                                'X_idx': x_idx,
                                'A_idx': a_idx,
                                'B_idx': b_idx,
                                'C_idx': c_idx,
                                'D_idx': d_idx,
                                'X_price': x_price,
                                'A_price': a_price,
                                'B_price': b_price,
                                'C_price': c_price,
                                'D_price': d_price,
                                'X_date': df.index[x_idx],
                                'A_date': df.index[a_idx],
                                'B_date': df.index[b_idx],
                                'C_date': df.index[c_idx],
                                'D_date': df.index[d_idx],
                                'AB_XA_ratio': ab_xa_ratio,
                                'BC_AB_ratio': bc_ab_ratio,
                                'CD_BC_ratio': cd_bc_ratio,
                                'AD_XA_ratio': ad_xa_ratio,
                                'D_index': d_idx
                            })
        
        # Look for bearish Butterfly (X:high, A:low, B:high, C:low, D:high)
        for i in range(len(swing_highs) - 1):  # X point (high)
            x_idx = swing_highs.iloc[i]['index']
            x_price = swing_highs.iloc[i]['price']
            
            # Find potential A points (lows) after X
            for j in range(len(swing_lows)):
                a_idx = swing_lows.iloc[j]['index']
                
                if a_idx <= x_idx:
                    continue  # A must be after X
                
                a_price = swing_lows.iloc[j]['price']
                xa_diff = a_price - x_price
                
                if xa_diff >= 0:
                    continue  # XA must be downward for bearish pattern
                
                # Find potential B points (highs) after A
                for k in range(i + 1, len(swing_highs)):
                    b_idx = swing_highs.iloc[k]['index']
                    
                    if b_idx <= a_idx:
                        continue  # B must be after A
                    
                    b_price = swing_highs.iloc[k]['price']
                    ab_diff = b_price - a_price
                    
                    if ab_diff <= 0:
                        continue  # AB must be upward for bearish pattern
                    
                    # Check AB/XA ratio
                    ab_xa_ratio = self.calculate_ratio(a_price, b_price, x_price)
                    
                    if not self.is_ratio_valid(ab_xa_ratio, ratios['AB_XA']):
                        continue
                    
                    # Find potential C points (lows) after B
                    for l in range(j + 1, len(swing_lows)):
                        c_idx = swing_lows.iloc[l]['index']
                        
                        if c_idx <= b_idx:
                            continue  # C must be after B
                        
                        c_price = swing_lows.iloc[l]['price']
                        bc_diff = c_price - b_price
                        
                        if bc_diff >= 0:
                            continue  # BC must be downward for bearish pattern
                        
                        # Check BC/AB ratio
                        bc_ab_ratio = self.calculate_ratio(b_price, c_price, a_price)
                        
                        if isinstance(ratios['BC_AB'], list):
                            if not (ratios['BC_AB'][0] <= bc_ab_ratio <= ratios['BC_AB'][1]):
                                continue
                        elif not self.is_ratio_valid(bc_ab_ratio, ratios['BC_AB']):
                            continue
                        
                        # Find potential D points (highs) after C
                        for m in range(k + 1, len(swing_highs)):
                            d_idx = swing_highs.iloc[m]['index']
                            
                            if d_idx <= c_idx:
                                continue  # D must be after C
                            
                            d_price = swing_highs.iloc[m]['price']
                            cd_diff = d_price - c_price
                            
                            if cd_diff <= 0:
                                continue  # CD must be upward for bearish pattern
                            
                            # Check CD/BC ratio
                            cd_bc_ratio = self.calculate_ratio(c_price, d_price, b_price)
                            
                            if isinstance(ratios['CD_BC'], list):
                                if not (ratios['CD_BC'][0] <= cd_bc_ratio <= ratios['CD_BC'][1]):
                                    continue
                            elif not self.is_ratio_valid(cd_bc_ratio, ratios['CD_BC']):
                                continue
                            
                            # Check AD/XA ratio
                            ad_xa_ratio = self.calculate_ratio(a_price, d_price, x_price)
                            
                            if isinstance(ratios['AD_XA'], list):
                                if not (ratios['AD_XA'][0] <= ad_xa_ratio <= ratios['AD_XA'][1]):
                                    continue
                            elif not self.is_ratio_valid(ad_xa_ratio, ratios['AD_XA']):
                                continue
                            
                            # All ratios match, we found a bearish Butterfly pattern
                            pattern_id += 1
                            patterns.append({
                                'id': pattern_id,
                                'type': 'Butterfly',
                                'direction': 'bearish',
                                'X_idx': x_idx,
                                'A_idx': a_idx,
                                'B_idx': b_idx,
                                'C_idx': c_idx,
                                'D_idx': d_idx,
                                'X_price': x_price,
                                'A_price': a_price,
                                'B_price': b_price,
                                'C_price': c_price,
                                'D_price': d_price,
                                'X_date': df.index[x_idx],
                                'A_date': df.index[a_idx],
                                'B_date': df.index[b_idx],
                                'C_date': df.index[c_idx],
                                'D_date': df.index[d_idx],
                                'AB_XA_ratio': ab_xa_ratio,
                                'BC_AB_ratio': bc_ab_ratio,
                                'CD_BC_ratio': cd_bc_ratio,
                                'AD_XA_ratio': ad_xa_ratio,
                                'D_index': d_idx
                            })
        
        logger.info(f"Found {len(patterns)} Butterfly patterns")
        return patterns 