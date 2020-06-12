import numpy as np
import math
from math import gamma
from math import log
from math import exp

z_base = 0.0


def ichol(L, D):
    return np.dot(L, np.dot(np.linalg.inv(D), L.T))


def extract_parents(DAG, U, i):
    if U[np.ix_(DAG[:, i] == 1, DAG[:, i] == 1)].shape[0] == 0:
        # print("vertex",i+1,"has no parents")
        return [[1]]
    return U[np.ix_(DAG[:, i] == 1, DAG[:, i] == 1)]


def extract_parents_and_self(DAG, U, i):
    return U[np.ix_(DAG[:, i] > 0, DAG[:, i] > 0)]


def extract_parents_vector(DAG, U, i):
    return U[np.ix_(DAG[:, i] == 1, [i])]


def num_of_parents(DAG):
    p = DAG.shape[0]
    v = np.zeros(p)     # number of parents
    for i in range(p):
        for j in range(p):
            if DAG[i, j] == 1:
                v[j] += 1
    return v


def num_of_edges(DAG):
    p = DAG.shape[0]
    edges = 0
    for i in range(p):
        for j in range(p):
            if DAG[i, j] == 1:
                edges += 1
    return edges


def set_alpha(v, c=1, d=10):    # alpha = c * v + d
    alpha = c * v + d
    return alpha


def set_U(p):
    return np.eye(p)


def schur_complement(U):  # 0列0行を除いた部分行列に対するSchur complement
    return U[0, 0] - np.dot(U[1:, 0:1].T,
                            np.dot(np.linalg.inv(U[1:, 1:]), U[1:, 0:1]))


def logzval(DAG, U, alpha):
    z = 0.0
    p = alpha.size
    v = num_of_parents(DAG)
    for i in range(p):
        expn = alpha[i] / 2.0 - v[i] / 2.0 - 1
        if expn <= 0:
            print("cannot be normalized at the vertex", i)
        #   print(extract_parents(DAG,U,i))
        #   print(extract_parents_and_self(DAG,U,i))
        #   print(gamma(expn))
        #   print(2.0 ** (alpha[i]/2.0 - 1))
        #   print(math.pi ** (v[i]/2.0))
        #   print(np.linalg.det(extract_parents(DAG,U,i)) ** (expn - 1.0/2.0) )
        #   print(np.linalg.det(extract_parents_and_self(DAG,U,i)) ** expn )
        # mul = gamma(expn) * (2.0 ** (alpha[i]/2.0 - 1) ) * (math.pi ** (v[i]/2.0)) * (np.linalg.det(extract_parents(DAG,U,i)) ** (expn - 1.0/2.0) ) / ( np.linalg.det(extract_parents_and_self(DAG,U,i) ) ** expn )
        z += log(gamma(expn)) + (alpha[i] / 2.0 - 1) * log(2.0) + (v[i] / 2.0) * log(math.pi) + (expn - 1.0 / 2.0) * log(
            np.linalg.det(extract_parents(DAG, U, i))) - expn * log(np.linalg.det(extract_parents_and_self(DAG, U, i)))
    # print(z)
    return z


def permutate(S, perm):
    p = perm.size
    P = np.zeros((p, p))
    for i in range(p):
        P[perm[i]][i] = 1
    S_P = np.dot(P.T, np.dot(S, P))
    return S_P


def inverse_permutate(S_P, perm):
    p = perm.size
    P = np.zeros((p, p))
    for i in range(p):
        P[perm[i]][i] = 1
    S = np.dot(P, np.dot(S_P, P.T))
    return S


def posterior_DAG_log_probability(DAG, S_P, n):
    v = num_of_parents(DAG)
    alpha = set_alpha(v)
    U = set_U(DAG.shape[0])
    global z_base
    if z_base == 0.0:
        z_base = logzval(DAG, U + n * S_P, n + alpha) - logzval(DAG, U, alpha)
        # print("z_base set to ",z_base)
    return logzval(DAG, U + n * S_P, n + alpha) - \
        logzval(DAG, U, alpha) - z_base


def thresholding(L, threshold):
    for i in range(L.shape[0]):
        for j in range(L.shape[1]):
            if abs(L[i][j]) <= threshold:
                L[i][j] = 0.0
    return L


def generate_initial_DAG(S_P, threshold=0.15):
    L = np.linalg.cholesky(S_P + 0.1 * np.eye(S_P.shape[0]))
    # print("initial L_P\n",L)
    L = thresholding(L, threshold)
    # print("L_P after thresholding:\n",L)
    num = 0
    DAG = np.eye(L.shape[0]) * 2
    for i in range(L.shape[0]):
        for j in range(i):
            if L[i][j] != 0.0:
                DAG[i][j] = 1
                num += 1
    # print("initial_DAG_P:\n",DAG)
    # print("edge_num = ",num)
    return DAG


'''
def shotgun_search(S_P,n,DAG):
    p = S_P.shape[0]
    search_width= p
    search_depth = (int)(p * (p-1) / 2)

    max_posterior_probability = exp (posterior_DAG_log_probability(DAG,S_P,n) )
    max_posterior_DAG = DAG.copy()
    for depth in range(search_depth):
        if (num_of_parents(DAG) == np.zeros(p)).all() :
            print("not anymore")
            break
        newDAGs = []
        probs = []
        probs_sum = 0.0
        remain = search_width
        while  remain > 0:
            i = np.random.randint(0,p)
            j = np.random.randint(0,p)
            if i == j : continue
            if i < j : i, j = j, i
            if DAG[i][j] == 1 :
                remain-=1
                newDAG = DAG.copy()
                newDAG[i][j] = 0
                newDAGs.append(newDAG)
                prob = exp( posterior_DAG_log_probability(newDAG,S_P,n) )
                probs.append(prob)
                probs_sum += prob
                #print(newDAG)
                #print(prob)
                if max_posterior_probability < prob :
                    max_posterior_probability = prob
                    max_posterior_DAG = newDAG.copy()
                    #print("max posterior DAG is updated to the following:\n",max_posterior_DAG)
                    #print("max_posterior_probability is now",max_posterior_probability,"\n")
        #print("\n")
        #print("Below are current_candidate DAGs and there relative probability")
        for i in range(len(probs)):
            probs[i]/=probs_sum
            #print(newDAGs[i])
            #print(probs[i])
        #print("Above are current_candidate DAGs and there relative probability\n")
        choice = np.random.choice(len(probs), p=probs)
        DAG = newDAGs[choice]
        #print("selected DAG:\n",DAG,"\n\n")
    #print("final DAG\n",max_posterior_DAG)
    return max_posterior_DAG , max_posterior_probability

'''


def shotgun_search(S_P, n, DAG, seed_num=32):
    np.random.seed(seed=seed_num)

    p = S_P.shape[0]

    search_width = p * 3
    search_depth = p * 3

    # DAG = generate_initial_DAG(S_P)
    max_posterior_probability = exp(posterior_DAG_log_probability(DAG, S_P, n))
    max_posterior_DAG = DAG.copy()

    for depth in range(search_depth):
        newDAGs = []
        probs = []
        probs_sum = 0.0
        remain = search_width
        while remain > 0:
            dice = np.random.randint(0, 3)
            if dice == 0:  # one edge away
                if (num_of_parents(DAG) == np.zeros(p)).all():
                    continue
                while True:
                    i = np.random.randint(0, p)
                    j = np.random.randint(0, p)
                    if i == j:
                        continue
                    if i < j:
                        i, j = j, i
                    if DAG[i, j] == 0:
                        continue
                    remain -= 1
                    newDAG = DAG.copy()
                    newDAG[i, j] = 0
                    newDAGs.append(newDAG)
                    prob = exp(posterior_DAG_log_probability(newDAG, S_P, n))
                    probs.append(prob)
                    probs_sum += prob
                    # print(newDAG)
                    # print(prob)
                    if max_posterior_probability < prob:
                        max_posterior_probability = prob
                        max_posterior_DAG = newDAG.copy()
                        # print("max posterior DAG is updated to the following:\n",max_posterior_DAG)
                        # print("max_posterior_probability is now",max_posterior_probability,"\n")
                    break
            elif dice == 1:  # one edge added
                if num_of_edges(DAG) == (p - 1) * p / 2:
                    continue
                while True:
                    i = np.random.randint(0, p)
                    j = np.random.randint(0, p)
                    if i == j:
                        continue
                    if i < j:
                        i, j = j, i
                    if DAG[i, j] == 1:
                        continue
                    remain -= 1
                    newDAG = DAG.copy()
                    newDAG[i, j] = 1
                    newDAGs.append(newDAG)
                    prob = exp(posterior_DAG_log_probability(newDAG, S_P, n))
                    probs.append(prob)
                    probs_sum += prob
                    # print(newDAG)
                    # print(prob)
                    if max_posterior_probability < prob:
                        max_posterior_probability = prob
                        max_posterior_DAG = newDAG.copy()
                        # print("max posterior DAG is updated to the following:\n",max_posterior_DAG)
                        # print("max_posterior_probability is now",max_posterior_probability,"\n")
                    break
            else:  # one edge replaced
                if (num_of_parents(DAG) == np.zeros(p)).all():
                    continue
                if num_of_edges(DAG) == (p - 1) * p / 2:
                    continue
                while True:
                    i = np.random.randint(0, p)
                    j = np.random.randint(0, p)
                    if i == j:
                        continue
                    if i < j:
                        i, j = j, i
                    if DAG[i, j] == 0:
                        break
                while True:
                    i2 = np.random.randint(0, p)
                    j2 = np.random.randint(0, p)
                    if i2 == j2:
                        continue
                    if i2 < j2:
                        i2, j2 = j2, i2
                    if DAG[i2, j2] == 1:
                        break
                remain -= 1
                newDAG = DAG.copy()
                newDAG[i, j] = 1
                newDAG[i2, j2] = 0
                newDAGs.append(newDAG)
                prob = exp(posterior_DAG_log_probability(newDAG, S_P, n))
                probs.append(prob)
                probs_sum += prob
                # print(newDAG)
                # print(prob)
                if max_posterior_probability < prob:
                    max_posterior_probability = prob
                    max_posterior_DAG = newDAG.copy()
                    # print("max posterior DAG is updated to the following:\n",max_posterior_DAG)
                    # print("max_posterior_probability is now",max_posterior_probability,"\n")
        for i in range(len(probs)):
            probs[i] /= probs_sum
            # print(newDAGs[i])
            # print(probs[i])
        # print("Above are current_candidate DAGs and there relative probability\n")
        choice = np.random.choice(len(probs), p=probs)
        DAG = newDAGs[choice]
        # print("selected DAG:\n",DAG,"\n\n")
    # print("final DAG\n",max_posterior_DAG)
    return max_posterior_DAG, max_posterior_probability


def map_estimate(DAG, S_P, n):
    p = DAG.shape[0]
    v = num_of_parents(DAG)
    alpha = set_alpha(v)
    U = set_U(p)
    alpha_post = alpha + n
    U_post = U + n * S_P
    D = np.zeros((p, p))
    L = np.eye(p)
    for i in range(p):
        if v[i] == 0:
            D[i, i] = U_post[i, i] / alpha_post[i]
            continue
        D[i, i] = schur_complement(
            extract_parents_and_self(DAG, U_post, i)) / alpha_post[i]
        # print(np.linalg.inv(extract_parents(DAG,U_post,i)))
        # print(extract_parents_vector(DAG,U_post,i))
        l_i = - np.dot(np.linalg.inv(extract_parents(DAG, U_post, i)),
                       extract_parents_vector(DAG, U_post, i))
        # print(l_i)
        index = 0
        for j in range(i + 1, p):
            if DAG[j, i] == 1:
                L[j, i] = l_i[index, 0]
                index += 1
        assert index == l_i.shape[0], "dimension of l_i and number of parents don't coincide"
    return L, D


def map_estimate2(DAG, S_P, n):
    p = DAG.shape[0]
    v = num_of_parents(DAG)
    alpha = set_alpha(v)
    U = set_U(p)
    alpha_post = alpha + n
    U_post = U + n * S_P
    D = np.zeros((p, p))
    L = np.eye(p)
    for i in range(p):
        if v[i] == 0:
            D[i, i] = U_post[i, i] / alpha_post[i]
            continue
        D[i, i] = schur_complement(
            extract_parents_and_self(DAG, U_post, i)) / (alpha_post[i] - v[i])
        # print(np.linalg.inv(extract_parents(DAG,U_post,i)))
        # print(extract_parents_vector(DAG,U_post,i))
        l_i = - np.dot(np.linalg.inv(extract_parents(DAG, U_post, i)),
                       extract_parents_vector(DAG, U_post, i))
        # print(l_i)
        index = 0
        for j in range(i + 1, p):
            if DAG[j, i] == 1:
                L[j, i] = l_i[index, 0]
                index += 1
        assert index == l_i.shape[0], "dimension of l_i and number of parents don't coincide"
    return L, D


def bayes_estimate(DAG, S_P, n):
    p = DAG.shape[0]
    v = num_of_parents(DAG)
    alpha = set_alpha(v)
    U = set_U(p)
    alpha_post = alpha + n
    U_post = U + n * S_P
    D = np.zeros((p, p))
    L = np.eye(p)
    for i in range(p):
        if v[i] == 0:
            D[i, i] = U_post[i, i] / alpha_post[i]
            continue
        D[i, i] = schur_complement(extract_parents_and_self(
            DAG, U_post, i)) / (alpha_post[i] - v[i] - 4)
        # print(np.linalg.inv(extract_parents(DAG,U_post,i)))
        # print(extract_parents_vector(DAG,U_post,i))
        l_i = - np.dot(np.linalg.inv(extract_parents(DAG, U_post, i)),
                       extract_parents_vector(DAG, U_post, i))
        # print(l_i)
        index = 0
        for j in range(i + 1, p):
            if DAG[j, i] == 1:
                L[j, i] = l_i[index, 0]
                index += 1
        assert index == l_i.shape[0], "dimension of l_i and number of parents don't coincide"
    return L, D


def l1(Omega, trueSigma):  # Stein's loss
    A = np.dot(Omega, trueSigma)
    if np.linalg.det(A) == 0:
        return 10101010101010
    return np.trace(A) - log(np.linalg.det(A)) - Omega.shape[0]


def l1_inv(Omega, trueSigma):  # Stein's loss
    Sigma = np.linalg.inv(Omega)
    trueOmega = np.linalg.inv(trueSigma)
    A = np.dot(trueOmega, Sigma)
    if np.linalg.det(A) == 0:
        return 10101010101010
    return np.trace(A) - log(np.linalg.det(A)) - Omega.shape[0]


def l4(Omega, trueOmega):
    c = 0.0
    p = Omega.shape[0]
    for i in range(p):
        for j in range(p):
            c += abs(Omega[i, j] - trueOmega[i, j])
    return c


def l5(Omega, trueOmega):
    c = 0.0
    p = Omega.shape[0]
    for i in range(p):
        for j in range(p):
            c += abs(Omega[i, j] - trueOmega[i, j]) ** 2
    return c


def TPR(DAG, trueDAG):
    p = DAG.shape[0]
    bunbo = 0.0
    bunshi = 0.0
    for i in range(p):
        for j in range(i):
            if trueDAG[i, j] == 1:
                bunbo += 1.0
            if trueDAG[i, j] == 1 and DAG[i, j] == 1:
                bunshi += 1.0
    return bunshi / bunbo


def FPR(DAG, trueDAG):
    p = DAG.shape[0]
    bunbo = 0.0
    bunshi = 0.0
    for i in range(p):
        for j in range(i):
            if trueDAG[i, j] == 0:
                bunbo += 1.0
            if trueDAG[i, j] == 0 and DAG[i, j] == 1:
                bunshi += 1.0
    return bunshi / bunbo

#
# n = 10      #観測数
# U = np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]])
# alpha = np.array([10,10,10,10])

# L0 = np.array([[1.0,0,0,0],[0.6,1.0,0,0],[0.1,0.6,1.0,0],[0.1,0.6,0.1,1.0]])
# S = np.dot(L0,L0.T)
# perm = np.array([0,1,2,3])
# FINALDAG = shotgun_search(S,perm,U,alpha,n)
# (L_P,D_P) = map_estimate(FINALDAG,U,alpha,permutate(S,perm),n,perm)
# L = inverse_permutate(L_P,perm)
# D = inverse_permutate(D_P,perm)
# Omega = np.dot(L,np.dot(np.linalg.inv(D),L.T))
# print(Omega)
# print(np.linalg.cholesky(Omega+0.0001*np.eye(Omega.shape[0])))


# U=np.array([[4,1,1,1],[1,4,2,1],[1,2,4,3],[1,1,3,4]])
# DAG=np.array([[2,0,0,0],[1,2,0,0],[1,1,2,0],[1,0,1,2]])
# U = np.array([[1,2,3,4],[5,6,7,8],[9,10,11,12],[13,14,15,16]])
# S = np.array([[2,1,1,1],[1,3,1,1],[1,1,4,1],[1,1,1,5]])
