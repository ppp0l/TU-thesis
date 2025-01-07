import numpy as np

def EER(alpha, eps, LB, UB) :

    if alpha + eps <= LB :
        return 0
    if alpha - eps >= UB :
        return 2*eps*(alpha *(UB -LB) + 1/2 *(LB**2 - (UB)**2))
    
    c1 = alpha - eps <= LB
    c2 = alpha + eps <= UB 

    if c1 and c2 :
        return 1/6*(alpha+eps-LB)**3
    elif c1 and not c2 :
        return 1/6*((alpha - LB + eps)**3 - (alpha - UB + eps)**3)
    elif (not c1) and c2 :
        return eps *(LB- alpha)**2 + 2*eps**3/3
    else :
        return 2*eps*(alpha *(alpha - eps -LB) + 1/2 *(LB**2 - (alpha-eps)**2))-1/6 * (alpha -UB + eps)**3
    
def posEER(q,p, lips) :
    
    alpha = UBq - lips.L * np.abs(p-q)-eps

    return EER(alpha, eps, LB, UB)