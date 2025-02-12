import numpy as np

def exp_upper(alpha, eps, LB, UB) :

    #prova = np.zeros_like(alpha)

    comp =np.zeros_like(alpha )

    cond1 = np.array(alpha - eps <= LB)
    cond2 = np.array(alpha + eps <= UB )

    c2 = 1/6*(alpha+eps-LB)**3
    comp[ cond1 * cond2 ] = c2  [ cond1 * cond2 ] 
    #prova[ cond1 * cond2 ] = 2

    c3 = 1/6*((alpha - LB + eps)**3 - (alpha - UB + eps)**3)
    comp[ cond1 *~cond2 ] = c3 [ cond1 *~cond2 ]
    #prova[ cond1 *~cond2 ] = 3

    c1 = 1/6*((alpha - LB + eps)**3 - (alpha - LB - eps)**3)
    comp[~cond1 * cond2 ] = c1 [~cond1 * cond2 ]
    #prova[~cond1 * cond2 ] = 1

    c4 = 2*eps*(alpha *(alpha - eps -LB) + 1/2 *(LB**2 - (alpha-eps)**2)) - 1/6 *( (alpha -UB + eps)**3 -8*eps**3)
    comp[~cond1 *~cond2 ] = c4 [~cond1 *~cond2 ]
    #prova[~cond1 *~cond2] = 4

    c5 = 2*eps*(alpha *(UB -LB) + 1/2 *(LB**2 - UB**2))

    #print(prova)

    cond3 = np.array(alpha - eps >= UB)
    comp[cond3] = c5[cond3]
    #prova[cond3] = 5

    cond4 = np.array(alpha + eps <= LB)
    comp[cond4] = 0
    #prova[cond4] = 6

    #print(prova)

    res = np.sum(comp, axis = 1)

    return res
    
def exp_lower(beta, eps, LB, UB) :

    comp =np.zeros_like(beta )
    
    cond1 = np.array(beta - eps <= LB)
    cond2 = np.array(beta + eps <= UB )

    c2 = 1/6*(beta-eps-UB)**3
    comp[ cond1 * cond2 ] = c2  [ cond1 * cond2 ] 

    c3 = 1/6*((beta - UB - eps)**3 - (beta -LB - eps)**3)
    comp[ cond1 *~cond2 ] = c3 [ cond1 *~cond2 ]

    c1 = 1/6*((beta - UB - eps)**3 - (beta - UB + eps)**3)
    comp[~cond1 * cond2 ] = c1 [~cond1 * cond2 ]

    c4 = 2*eps*(beta *(UB - beta - eps ) + 1/2 *((beta+eps)**2 - UB**2)) - 1/6 * ( (beta -LB - eps)**3 + 8 * eps**3 )
    comp[~cond1 *~cond2 ] = c4 [~cond1 *~cond2 ]

    c5 = 2*eps*(beta *(UB -LB) + 1/2 *(LB**2 - UB **2))

    comp[beta + eps <= LB] = c5[beta + eps <= LB]
    comp[beta - eps >= UB] = 0


    comp = comp/(2*np.reshape(eps, (1, -1)) * (UB-LB))
    
    res = np.sum(comp, axis = 1)

    return res

def test_EER(eps, p, q, lips):
    _, LB, UB = lips.predict(p, return_bds=True)
    _, LBq, UBq = lips.predict(q, return_bds=True)

    dist = np.linalg.norm(p-q, ord = 2, axis=1, keepdims=True)
    L = lips.L

    alpha = UBq - L * dist - eps

    EUI = exp_upper(alpha, eps, LB,  UB)

    beta = - LBq - L * dist - eps

    ELI = - exp_upper(beta,eps,-UB, -LB)

    #print(EUI.mean(), ELI.mean())
    return EUI - ELI

def grad_exp_upper(alpha, eps, LB, UB) :

    #prova = np.zeros_like(alpha)

    n_samples = len(alpha)
    n_out = len(alpha[0])

    dalpha = np.zeros( (n_samples, n_out) )
    deps = np.zeros( (n_samples, n_out))
    dLB = np.zeros( (n_samples, n_out) )
    dUB = np.zeros( (n_samples, n_out))

    cond1 = np.array(alpha - eps <= LB)
    cond2 = np.array(alpha + eps <= UB )

    # c2 = 1/6*(alpha+eps-LB)**3
    der = 1/2*(alpha+eps-LB)**2  
    dalpha[ cond1 * cond2 ] = der [ cond1 * cond2 ] 
    der = 1/2*(alpha+eps-LB)**2
    deps[ cond1 * cond2 ] = der [ cond1 * cond2 ] 
    der = - 1/2*(alpha+eps-LB)**2
    dLB[ cond1 * cond2 ] = der [ cond1 * cond2 ] 
    dUB[ cond1 * cond2 ] = 0
    # prova[ cond1 * cond2 ] = 2

    # c3 = 1/6*((alpha - LB + eps)**3 - (alpha - UB + eps)**3)
    der = 1/2*((alpha - LB + eps)**2 - (alpha - UB + eps)**2)
    dalpha[ cond1 *~cond2 ] =  der [ cond1 *~cond2 ] 
    der = 1/2*((alpha - LB + eps)**2 - (alpha - UB + eps)**2) 
    deps[ cond1 *~cond2 ] = der [ cond1 *~cond2 ] 
    der = - 1/2*(alpha + eps - LB)**2
    dLB[ cond1 *~cond2 ] = der  [ cond1 *~cond2 ] 
    der = 1/2*(alpha + eps - UB)**2
    dUB[ cond1 *~cond2 ] = der  [ cond1 *~cond2 ] 
    # prova[ cond1 *~cond2 ] = 3

    # c1 = 1/6*((alpha - LB + eps)**3 - (alpha - LB - eps)**3)
    der = 1/2*((alpha - LB + eps)**2 - (alpha - LB - eps)**2)
    dalpha[~cond1 * cond2 ] =  der [~cond1 * cond2 ] 
    der = 1/2*((alpha - LB + eps)**2 + (alpha - LB - eps)**2) 
    deps[~cond1 * cond2 ] = der [~cond1 * cond2 ] 
    der = - 1/2*((alpha - LB + eps)**2 - (alpha - LB - eps)**2)
    dLB[~cond1 * cond2 ] = der  [~cond1 * cond2 ] 
    dUB[~cond1 * cond2 ] = 0
    # prova[~cond1 * cond2 ] = 1

    # c4 = 2*eps*(alpha *(alpha - eps -LB) + 1/2 *(LB**2 - (alpha-eps)**2)) - 1/6 *( (alpha -UB + eps)**3 -8*eps**3)
    der = 2*eps*( alpha -LB ) - 1/2 * (alpha -UB + eps)**2
    dalpha[~cond1 *~cond2 ] =  der  [~cond1 *~cond2 ] 
    der = 2*(alpha *(alpha - eps -LB) + 1/2 *(LB**2 - (alpha-eps)**2)) + 2*eps**2 - 1/2 *( (alpha -UB + eps)**2  - 8*eps**2)
    deps[~cond1 *~cond2 ] = der  [~cond1 *~cond2 ] 
    der = 2*eps*( alpha + LB )
    dLB[~cond1 *~cond2 ] = der  [~cond1 *~cond2 ] 
    der = 1/2 * (alpha -UB + eps)**2
    dUB[~cond1 *~cond2 ] = der [~cond1 *~cond2 ] 
    # prova[~cond1 *~cond2 ] = 4

    # print(prova)

    cond3 = np.array(alpha - eps >= UB)
    if np.any(cond3) :
    # c5 = 2*eps*(alpha *(UB -LB) + 1/2 *(LB**2 - UB**2))
        der = np.ones((len(cond3), 1) ) *2*eps *(UB -LB)
        dalpha[cond3] =  der  [cond3]
        der = 2*(alpha *(UB -LB) + 1/2 *(LB**2 - UB**2))
        deps[cond3] = der [cond3] 
        der = 2*eps*(- alpha + LB )
        dLB[cond3] = der [cond3] 
        der = 2*eps*( alpha - UB )
        dUB[cond3] = der [cond3] 
    # prova[cond3] = 5

    cond4 = np.array(alpha + eps <= LB)
    if np.any(cond4) :
        dalpha[cond4] = 0
        deps[cond4] = 0
        dLB[cond4] = 0
        dUB[cond4] = 0
    # prova[cond4] = 6
    # print(prova)

    return dalpha, deps, dLB, dUB

def grad_exp_lower(beta, eps, LB, UB) :

    # prova = np.zeros_like(beta)

    n_samples = len(beta)
    n_out = len(beta[0])

    dbeta = np.zeros( (n_samples, n_out) )
    deps = np.zeros( (n_samples, n_out))
    dLB = np.zeros( (n_samples, n_out) )
    dUB = np.zeros( (n_samples, n_out))

    cond1 = np.array(beta - eps <= LB)
    cond2 = np.array(beta + eps <= UB )

    # c2 = 1/6*(beta-eps-UB)**3
    der = 1/2*(beta-eps-UB)**2  
    dbeta[ cond1 * cond2 ] = der [ cond1 * cond2 ] 
    der = 1/2*(beta-eps-UB)**2
    deps[ cond1 * cond2 ] = der [ cond1 * cond2 ] 
    dLB[ cond1 * cond2 ] = 0
    der = - 1/2*(beta-eps-UB)**2
    dUB[ cond1 * cond2 ] = der [ cond1 * cond2 ] 
    # prova[ cond1 * cond2 ] = 2

    # c3 = 1/6*((beta - UB - eps)**3 - (beta -LB - eps)**3)
    der = 1/2*((beta - UB - eps)**2 - (beta - LB - eps)**2)
    dbeta[ cond1 *~cond2 ] =  der [ cond1 *~cond2 ] 
    der = 1/2*((beta - UB - eps)**2 - (beta - LB - eps)**2) 
    deps[ cond1 *~cond2 ] = der [ cond1 *~cond2 ] 
    der = 1/2*(beta - eps - LB)**2
    dLB[ cond1 *~cond2 ] = der  [ cond1 *~cond2 ] 
    der = - 1/2*(beta - eps - UB)**2
    dUB[ cond1 *~cond2 ] = der  [ cond1 *~cond2 ] 
    # prova[ cond1 *~cond2 ] = 3
    
    # c1 = 1/6*((beta - UB - eps)**3 - (beta - UB + eps)**3)
    der = 1/2*((beta - UB - eps)**2 - (beta - UB + eps)**2)
    dbeta[~cond1 * cond2 ] =  der [~cond1 * cond2 ] 
    der = 1/2*((beta - UB - eps)**2 + (beta - UB + eps)**2) 
    deps[~cond1 * cond2 ] = der [~cond1 * cond2 ] 
    dLB[~cond1 * cond2 ] = 0
    der = - 1/2*((beta - UB - eps)**2 - (beta - UB + eps)**2)
    dUB[~cond1 * cond2 ] = der  [~cond1 * cond2 ] 
    # prova[~cond1 * cond2 ] = 1

    # c4 = 2*eps*(beta *(UB - beta - eps ) + 1/2 *((beta+eps)**2 - UB**2)) - 1/6 * ( (beta -LB - eps)**3 + 8 * eps**3 )
    der = 2*eps*( UB - beta ) - 1/2 * (beta -LB - eps)**2
    dbeta[~cond1 *~cond2 ] =  der  [~cond1 *~cond2 ] 
    der = 2*(beta *(UB - beta - eps ) + 1/2 *((beta+eps)**2 - UB**2)) + 2*eps**2 - 1/2 *( - (beta -LB - eps)**2  + 8*eps**2)
    deps[~cond1 *~cond2 ] = der  [~cond1 *~cond2 ] 
    der = 1/2 * (beta - LB - eps)**2
    dLB[~cond1 *~cond2 ] = der  [~cond1 *~cond2 ] 
    der = 2*eps*( beta - LB )
    dUB[~cond1 *~cond2 ] = der [~cond1 *~cond2 ] 
    # prova[~cond1 *~cond2 ] = 4

    # print(prova)

    cond3 = np.array(beta + eps <= LB)
    if np.any(cond3) :
    # c5 = 2*eps*(beta *(UB -LB) + 1/2 *(LB**2 - UB**2))
        der = np.ones((len(cond3), 1) ) *2*eps *(UB -LB)
        dbeta[cond3] =  der  [cond3]
        der = 2*(beta *(UB -LB) + 1/2 *(LB**2 - UB**2))
        deps[cond3] = der [cond3] 
        der = 2*eps*(- beta + LB )
        dLB[cond3] = der [cond3] 
        der = 2*eps*( beta - UB )
        dUB[cond3] = der [cond3] 
    # prova[cond3] = 5

    cond4 = np.array(beta - eps >= UB)
    if np.any(cond4) :
        dbeta[cond4] = 0
        deps[cond4] = 0
        dLB[cond4] = 0
        dUB[cond4] = 0
    # prova[cond4] = 6
    # print(prova)

    return dbeta, deps, dLB, dUB