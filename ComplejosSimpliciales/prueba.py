from AlphaComplex import AlphaComplex

sc = AlphaComplex([[-3, 0], [0, 1], [3, 0], [-2, -2], [2, -2], [0, -4]])

#print(np.matrix(sc.matriz_borde_generalizado()))
print(sc.algoritmo_matriz(sc.matriz_borde_generalizado()))

# M = np.matrix([[1, 0, 0, 1], [1, 1, 0, 1], [1, 0, 0, 0], [0, 0, 1, 1]])
# print(sc.algoritmo_matriz(M))

lista = [(0,), (1,), (2,), (3,), (4,), (5,), (0, 3), (2, 4), (3, 5), (4, 5), (1, 2), (0, 1), (1, 3), (1, 4), (3, 4), (0, 1, 3), (1, 2, 4), (3, 4, 5), (1, 3, 4)]
