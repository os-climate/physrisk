import collections.abc


def get_iterable(x):
    if isinstance(x, collections.abc.Iterable):
        return x
    else:
        return (x,)
