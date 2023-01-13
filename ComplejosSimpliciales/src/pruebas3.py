import time
from multiprocessing import freeze_support

import numpy as np

from VietorisRipsComplex import Vietoris_RipsComplex

if __name__ == '__main__':
    freeze_support()

    points = np.array([(0.38021546727456423, 0.46419202339598786), (0.7951628297672293, 0.49263630135869474),
                       (0.566623772375203, 0.038325621649018426), (0.3369306814864865, 0.7103735061134965),
                       (0.08272837815822842, 0.2263273314352896), (0.5180166301873989, 0.6271769943824689),
                       (0.33691411899985035, 0.8402045183219995), (0.33244488399729255, 0.4524636520475205),
                       (0.11778991601260325, 0.6657734204021165), (0.9384303415747769, 0.2313873874340855)])

    points = np.random.rand(20, 2)
    print("hola")

    t0 = time.time()
    alpha = Vietoris_RipsComplex(points)
    #print(alpha.faces)
    t1 = time.time()

    print(time.strftime("%H:%M:%S", time.gmtime(t1 - t0)))
