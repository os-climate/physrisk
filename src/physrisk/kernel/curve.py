import numpy as np
    
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

