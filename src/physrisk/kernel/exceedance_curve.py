import numpy as np
import physrisk.kernel.curve as cv
from typing import List, Union

class ExceedanceCurve:
    """A point on an exceedance curve comprises an value, v, and a probability, p.
    p is the probability that an event occurs with event value (e.g. intensity) >= v."""
    
    __slots__ = ["probs", "values"]
    
    def __init__(self,
        probs: Union[List[float], np.ndarray],   
        values: Union[List[float], np.ndarray]):
        """Create a new asset event distribution.
        Args:
            probs: exceedance probabilities (must be sorted and decreasing)
            values: values (must be sorted and non-decreasing)
        """

        # probabilities must be sorted and decreasing
        # values must be sorted and non-decreasing (intens[i + 1] >= intens[i])
        if len(probs) != len(values):
            raise ValueError('same number of probabilities and values expected')
        if not np.all(np.diff(probs) < 0):
            raise ValueError('probs must be sorted and decreasing')
        if not np.all(np.diff(values) >= 0):
            raise ValueError('values must be sorted and non-decreasing')

        self.probs = np.array(probs)
        self.values = np.array(values) 

    def add_value_point(self, value):
        """Add a point to the curve with specified value and exceedance probability determined from existing curve by linear interpolation."""
        values, probs = cv.add_x_value_to_curve(value, self.values, self.probs)
        return ExceedanceCurve(probs, values)

    def get_probability_bins(self):
        r"""Convert from exceedance (cumulative) probability to bins of constant probability.
        This is equivalent to the assumption of linear interpolation of exceedance points.

        .. math::
            p^\text{b}_i = p^\text{e}_{i + 1} - p^\text{e}_i 
        
        Returns:
            value_bins (ndarray), probs: the contiguous bin lower and upper values, probabilities of each bin
            If value_bins is of lenth n then ther are n-1 bins and n-1 probabilities
        
        """
        # value bins are contiguous 
        value_bins = self.values[:]
        probs = self.probs[:-1] - self.probs[1:] 
        return value_bins, probs
        

