import collections


def get_iterable(x):
    if isinstance(x, collections.Iterable):
        return x
    else:
        return (x,)
