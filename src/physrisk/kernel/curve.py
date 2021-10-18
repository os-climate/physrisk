import numpy as np
from physrisk.kernel import exceedance_curve
from physrisk.kernel.exceedance_curve import ExceedanceCurve
    
def add_x_value_to_curve(x, curve_x, curve_y):
    """Add an x value to a curve, interpolated from the existing curve. curve_x and curve_y are the curve x and y values.
    curve_x is sorted non-decreasing. This function may be used to align curves and bins.
    """
    # note some care needed as multiple identical curve_x are permitted: cannot simply use np.interp
    # this might be a candidate for numba optimization

    i = np.searchsorted(curve_x, x)

    if i == len(curve_y):
        curve_y = np.insert(curve_y, i, curve_y[i - 1]) # flat extrapolation
        curve_x = np.insert(curve_x, i, x)    
    elif x == curve_x[i]:
        # point already exists; nothing to do
        return curve_x, curve_y
    elif (i == 0):
        curve_y = np.insert(curve_y, 0, curve_y[0]) # flat extrapolation
        curve_x = np.insert(curve_x, 0, x)       
    else:
        pl, pu = curve_y[i - 1], curve_y[i]
        il, iu = curve_x[i - 1], curve_x[i]
        # linear interpolation; quadratic interpolation (linear in probability density) may also be of interest
        prob = pl + (x - il) * (pu - pl) / (iu - il) 
        curve_y = np.insert(curve_y, i, prob) 
        curve_x = np.insert(curve_x, i, x)
    
    return curve_x, curve_y

def to_exceedance_curve(bin_edges, probs):
    """An exceedance curve gives the probability that the random variable is greater than the value,
        a type of cumulative probability.   
    """
    nz = np.asarray(probs > 0).nonzero()
    fnz = nz[0][0] if len(nz[0]) > 0 else 0
    nz_values = bin_edges[fnz:]
    nz_probs = probs[fnz:]
    cum_prob = np.insert(np.cumsum(nz_probs[::-1]), 0, 0.0)[::-1]
    return ExceedanceCurve(cum_prob, nz_values)