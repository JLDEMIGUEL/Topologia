import numpy as np

from SimplicialComplex import SimplicialComplex

print("\n\n\n###############")
print("EJEMPLO 1: Complejo (0,1,2,3)")
print("###############\n")

complejo = SimplicialComplex({(0, 1, 2, 3)})

print("Representacion del complejo simplicial: ")
print(complejo.face_set())
print("\n")

print("Dimension")
print(complejo.dimension())
print("\n")

print("N_faces(0)")
print(complejo.n_faces(0))
print("\n")

print("N_faces(1)")
print(complejo.n_faces(1))
print("\n")

print("N_faces(2)")
print(complejo.n_faces(2))
print("\n")

print("N_faces(3)")
print(complejo.n_faces(3))
print("\n")

print("Star(0,1)")
print(complejo.star((0, 1)))
print("\n")

print("ClosedStar(0,1)")
print(complejo.closedStar((0, 1)))
print("\n")

print("Link(0,1)")
print(complejo.link((0, 1)))
print("\n")

print("Skeleton(2)")
print(complejo.skeleton(2))
print("\n")

print("Euler Caracateristic sc")
print(complejo.euler_characteristic())
print("\n")

# SC1 SKELETON(2)
print("\n\n\n###############")
print("EJEMPLO 2: SC1 SKELETON(2)")
print("###############\n")

sc1 = SimplicialComplex(complejo.skeleton(2))

print("FaceSet sc1")
print(sc1.face_set())
print("\n")

print("Dimension sc1")
print(sc1.dimension())
print("\n")

print("Star sc1 (0,)")
print(sc1.star((0,)))
print("\n")

print("Link sc1 (0,)")
print(sc1.link((0,)))
print("\n")

print("Euler Caracateristic sc")
print(sc1.euler_characteristic())
print("\n")

# SC2
print("\n\n\n###############")
print("EJEMPLO 3: SC {(0,1),(1,2,3,4),(4,5),(5,6),(4,6),(6,7,8),(8,9)}")
print("###############\n")

sc = SimplicialComplex({(0, 1), (1, 2, 3, 4), (4, 5), (5, 6), (4, 6), (6, 7, 8), (8, 9)})

print("FaceSet sc")
print(sc.face_set())
print("\n")

print("Dimension sc")
print(sc.dimension())
print("\n")

print("Skeleton(1) sc")
print(sc.skeleton(1))
print("\n")

print("Star sc (4,)")
print(sc.star((4,)))
print("\n")

print("Link sc (4,)")
print(sc.link((4,)))
print("\n")

print("Euler Caracateristic sc")
print(sc.euler_characteristic())
print("\n")

# EJEMPLO 4
print("\n\n\n###############")
print("EJEMPLO 4:")
print("###############\n")

sc = SimplicialComplex(sc.skeleton(1))

print("FaceSet sc")
print(sc.face_set())
print("\n")

print("Euler Caracateristic sc")
print(sc.euler_characteristic())
print("\n")

# EJEMPLO 5
print("\n\n\n###############")
print("EJEMPLO 5:")
print("###############\n")

sc = SimplicialComplex({(0, 1, 2), (2, 3), (3, 4)})

print("FaceSet sc")
print(sc.face_set())
print("\n")

print("Dimension sc")
print(sc.dimension())
print("\n")

print("Skeleton(1) sc")
print(sc.skeleton(1))
print("\n")

print("Star((2,)) sc")
print(sc.star((2,)))
print("\n")

print("Link((2,)) sc")
print(sc.link((2,)))
print("\n")

# EJEMPLO 6
print("\n\n\n###############")
print("EJEMPLO 6:")
print("###############\n")

sc = SimplicialComplex({(1, 2, 4), (1, 3, 6), (1, 4, 6), (2, 3, 5), (2, 4, 5), (3, 5, 6)})

print("FaceSet sc")
print(sc.face_set())
print("\n")

print("Dimension sc")
print(sc.dimension())
print("\n")

print("Skeleton(1) sc")
print(sc.skeleton(1))
print("\n")

print("Star((1,4)) sc")
print(sc.star((1, 4)))
print("\n")

print("Link((1,4)) sc")
print(sc.link((1, 4)))
print("\n")

print("Euler Caracateristic sc")
print(sc.euler_characteristic())
print("\n")

print("Connected components sc")
print(sc.connected_components())
print("\n")

# EJEMPLO 7
print("\n\n\n###############")
print("EJEMPLO 7:")
print("###############\n")

sc = SimplicialComplex(sc.skeleton(1))

print("Euler Caracateristic sc")
print(sc.euler_characteristic())
print("\n")

# EJEMPLO 8
print("\n\n\n###############")
print("EJEMPLO 8 (Toro):")
print("###############\n")

sc = SimplicialComplex({(1, 2, 4), (2, 4, 5), (2, 3, 5), (3, 5, 6), (1, 3, 6), (1, 4, 6), (4, 5, 7),
                        (5, 7, 8), (5, 6, 8), (6, 8, 9), (4, 6, 9), (4, 7, 9), (1, 7, 8), (1, 2, 8), (2, 8, 9),
                        (2, 3, 9), (3, 7, 9),
                        (1, 3, 7)})

print("Dimension sc")
print(sc.dimension())
print("\n")

print("N_faces(1)")
print(sc.n_faces(1))
print("\n")

print("Star((1,)) sc")
print(sc.star((1,)))
print("\n")

print("Link((1,)) sc")
print(sc.link((1,)))
print("\n")

print("Euler Caracateristic sc")
print(sc.euler_characteristic())
print("\n")

# EJEMPLO 8.1
print("\n\n\n###############")
print("EJEMPLO 8.1 (skeleton 1 Toro):")
print("###############\n")

sc = SimplicialComplex(sc.skeleton(1))

print("Euler Caracateristic sc")
print(sc.euler_characteristic())
print("\n")

# EJEMPLO 9
print("\n\n\n###############")
print("EJEMPLO 9 (Plano proyectivo):")
print("###############\n")

sc = SimplicialComplex({(1, 2, 6), (2, 3, 4), (1, 3, 4), (1, 2, 5), (2, 3, 5), (1, 3, 6),
                        (2, 4, 6), (1, 4, 5), (2, 3, 5), (4, 5, 6)})

print("Dimension sc")
print(sc.dimension())
print("\n")

print("N_faces(1)")
print(sc.n_faces(1))
print("\n")

print("Star((1,)) sc")
print(sc.star((1,)))
print("\n")

print("Link((1,)) sc")
print(sc.link((1,)))
print("\n")

print("Euler Caracateristic sc")
print(sc.euler_characteristic())
print("\n")

print("Connected components sc")
print(sc.connected_components())
print("\n")

# EJEMPLO 10
print("\n\n\n###############")
print("EJEMPLO 10 (skeleton 1 Plano proyectivo):")
print("###############\n")

sc = SimplicialComplex(sc.skeleton(1))

print("Euler Caracateristic sc")
print(sc.euler_characteristic())
print("\n")

# EJEMPLO 11
print("\n\n\n###############")
print("EJEMPLO 11:")
print("###############\n")

sc = SimplicialComplex({(0,), (1,), (2, 3), (4, 5), (5, 6), (4, 6), (6, 7, 8, 9)})

print("FaceSet sc")
print(sc.face_set())
print("\n")

print("Dimension sc")
print(sc.dimension())
print("\n")

print("Skeleton(1) sc")
print(sc.skeleton(1))
print("\n")

print("Star((6,)) sc")
print(sc.star((6,)))
print("\n")

print("Link((6,)) sc")
print(sc.link((6,)))
print("\n")

print("Euler Caracateristic sc")
print(sc.euler_characteristic())
print("\n")

print("Connected components sc")
print(sc.connected_components())
print("\n")

sc.add([(99, 100)])

sc.add([(9, 10)])
sc.add([(999, 1000)])

print(sc.orderByFloat())
print(sc.filterByFloat(2))

points=np.array([(0.38021546727456423, 0.46419202339598786), (0.7951628297672293, 0.49263630135869474), (0.566623772375203, 0.038325621649018426), (0.3369306814864865, 0.7103735061134965), (0.08272837815822842, 0.2263273314352896), (0.5180166301873989, 0.6271769943824689), (0.33691411899985035, 0.8402045183219995), (0.33244488399729255, 0.4524636520475205), (0.11778991601260325, 0.6657734204021165), (0.9384303415747769, 0.2313873874340855)])

sc.AlphaComplex(points)