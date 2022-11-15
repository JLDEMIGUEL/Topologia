from ComplejosSimpliciales.src.SimplicialComplex import SimplicialComplex

tetraedro = SimplicialComplex({(0, 1, 2, 3)})
tetraedro_borde = tetraedro.skeleton(2)

figura = SimplicialComplex({(0, 1), (1, 2, 3, 4), (4, 5), (5, 6), (4, 6), (6, 7, 8), (8, 9)})

anillo = SimplicialComplex({(1, 2, 4), (1, 3, 6), (1, 4, 6), (2, 3, 5), (2, 4, 5), (3, 5, 6)})

toro = SimplicialComplex(
    {(1, 2, 4), (2, 4, 5), (2, 3, 5), (3, 5, 6), (1, 3, 6), (1, 4, 6), (4, 5, 7), (5, 7, 8), (5, 6, 8), (6, 8, 9),
     (4, 6, 9), (4, 7, 9), (1, 7, 8), (1, 2, 8), (2, 8, 9), (2, 3, 9), (3, 7, 9), (1, 3, 7)})

plano_proyectivo = SimplicialComplex({(1, 2, 6), (2, 3, 4), (1, 3, 4), (1, 2, 5), (2, 3, 5), (1, 3, 6),
                                      (2, 4, 6), (1, 4, 5), (3, 5, 6), (4, 5, 6)})

botella_klein = SimplicialComplex(
    {(0, 1, 7), (0, 6, 7), (1, 7, 8), (1, 2, 8), (2, 3, 8), (0, 2, 3), (3, 4, 6), (4, 6, 7),
     (4, 5, 7), (5, 7, 8), (5, 6, 8), (3, 6, 8), (0, 1, 3), (1, 3, 4), (1, 2, 4), (2, 4, 5), (0, 2, 5), (0, 5, 6)})
