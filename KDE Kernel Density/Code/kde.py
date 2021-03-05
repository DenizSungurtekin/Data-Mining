import numpy as np
# When comments are in french, its for myself

def normalKernel1D(X):

    return 1 / np.sqrt(2 * np.pi) * np.exp(-0.5 * (X ** 2)) #Formule 3 du gaussian kernel dans le pdf pour un seul x ( Univariate )


def normalKernelMultiD(X, d):
        
    return (1 / ((2 * np.pi) ** (d / 2))) * np.exp(-0.5 * np.dot(X.T, X))  #Formule 4 du pdf: cas multivariate, où d est le nombre d'element de x

def deltaProd(X, c, h, d):

    result = 1
    if d > 1:
        for i in range(d):
            result = result * normalKernel1D((X[i] - c[i]) / h)

    else:
        result = result * normalKernel1D((X - c) / h)
                                                            #  Mais ici pour rendre le kernel plus smooth on utilise le gaussian kernel
                                                            #  La partie produit de la formule 6 du pdf
    return result


def deltaMultiD(X, c, h, d):

    return normalKernelMultiD((X - c) / h, d)


# Case with one element in c
def densityEstimation(X,c,h,d):

    Vr = h**d
    scalar = 1 / Vr # n = 1


    resultat = [scalar * deltaProd(X[j],c,h,d) for j in range(X.shape[0])]

    resultat = np.asarray(resultat)


    return resultat         # Formule 5 (sans la somme ,car probabilité inconditionel) du pdf, un vecteur de taille [900,1] convertit en 30x30 lors du plot



# Case with multiple element in c
def densityEstimation2(X,c,h,d):  # X[j]: un point de testSet, c: centre de l'hypercube, h: taille de l'hypercube, d: dimension des points

    n = len(c)
    Vr = h**d
    scalar = 1 / (Vr * n)
    resultat = np.zeros(len(X))

    for j in range(X.shape[0]):
        somme = 0
        for i in range(n):
            somme = somme + deltaProd(X[j],c[i],h,d)

        resultat[j] = scalar * somme

    return resultat         # Formule 6 (avec la somme ,car probabilité sans conditionel) du pdf, un vecteur de taille [900,1] convertit en 30x30 lors du plot


# Naives Bayes necessary functions without mean_std
def train(X, y):

    unique_y = np.unique(y)
    points_by_class = [[x for x, t in zip (X, y) if t == c] for c in unique_y]
    points_by_class = np.asarray(points_by_class)

    prior = [len(points_by_class[i])/len(X) for i in range(len(unique_y))]

    return prior
