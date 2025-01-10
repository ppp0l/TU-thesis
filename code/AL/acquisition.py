import numpy as np

def exp_upper(alpha, eps, LB, UB) :

    comp =np.zeros_like(alpha )

    res = np.zeros(len(alpha))

    cond1 = np.array(alpha - eps <= LB)
    cond2 = np.array(alpha + eps <= UB )

    c2 = 1/6*(alpha+eps-LB)**3
    comp[ cond1 * cond2 ] = c2  [ cond1 * cond2 ] 

    c3 = 1/6*((alpha - LB + eps)**3 - (alpha - UB + eps)**3)
    comp[ cond1 *~cond2 ] = c3 [ cond1 *~cond2 ]

    c1 = 1/6*((alpha - LB + eps)**3 - (alpha - LB - eps)**3)
    comp[~cond1 * cond2 ] = c1 [~cond1 * cond2 ]

    c4 = 2*eps*(alpha *(alpha - eps -LB) + 1/2 *(LB**2 - (alpha-eps)**2)) - 1/6 *( (alpha -UB + eps)**3 -8*eps**3)
    comp[~cond1 *~cond2 ] = c4 [~cond1 *~cond2 ]

    c5 = 2*eps*(alpha *(UB -LB) + 1/2 *(LB**2 - UB**2))

    comp[alpha - eps >= UB] = c5[alpha - eps >= UB]

    comp[alpha + eps <= LB] = 0

    res = np.sum(comp, axis = 1)

    return res
    
def exp_lower(beta, eps, LB, UB) :

    comp =np.zeros_like(beta )

    res = np.zeros(len(beta))
    
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

def pos_EER(eps, p, q, lips):
    _, LB, UB = lips.predict(p, return_bds=True)
    _, LBq, UBq = lips.predict(q, return_bds=True)

    dist = np.linalg.norm(p-q, ord = 2, axis=1, keepdims=True)
    L = lips.L

    alpha = UBq - L * dist - eps

    EUI = exp_upper(alpha, eps, LB,  UB)

    beta = LBq + L * dist + eps

    ELI = exp_lower(beta,eps,LB, UB)


    return EUI - ELI