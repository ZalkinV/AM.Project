import math



def first_function(x):
    return math.e**(1 / (1 + x))


def second_function(x, y, z):
    # (x | y) -> (y -> z)
    # !(x and y) -> (!y or z)
    # (x and y) or (!y or z)
    # x and y or !y or z
    # x or !y or z
    return int(x or not y or z)


def third_function(x, y):
    return int((y and x) or (not y and not x) or y)
