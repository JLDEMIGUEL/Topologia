import math

import numpy as np


def calcRadio(points):
    if len(points) <= 1:
        return 0
    maximum = 0
    for x in points:
        for y in [z for z in points if not np.array_equal(x, z)]:
            dist = math.dist(x, y)
            if maximum < dist:
                maximum = dist
    return maximum
