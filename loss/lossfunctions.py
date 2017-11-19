import numpy as np

'''
    l - landmark
    v - visibility
    l_p - landmark prediction
    v_p - visibility prediction
    n_itr - number of iteration
    len - length of data
'''
def loss_fashion_landmarks(l, v, l_p, v_p, n_itr, len, alpha = 0.5):
        t1 = 2000
        t2 = 400

        loss = 0
        loss_landmarks = 0
        loss_visibilities = 0
        
        if n_itr < t1 :
            alpha = alpha 
        elif t1 <= n_int < t2:
            alpha = alpha*((n_itr-t1)/(t2-t1))
        else:
            alpha = 0

        #Calculate Euclidient Loss for Visibility, Landmark Positions
        for i in range(1, len):
            loss_landmarks += np.sqrt(np.linalg.norm(v_l-l))
            loss_visibilities += np.sqrt(np.linalg.norm(v_p-v))
        
        loss_landmarks = loss_landmarks/(2*len)
        loss_visibilities = loss_visibilities/(2*len)

    loss = loss_landmarks + (alpha*loass_visibilities)
    return loss

def loss_fashion_net:
    pass