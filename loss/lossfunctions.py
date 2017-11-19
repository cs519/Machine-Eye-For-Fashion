'''
    l - landmark
    v - visibility
    l_p - landmark prediction
    v_p - visibility prediction
    n_it - number of iteration
    len - length of data
'''


def get_loss(l, v, l_p, v_p, n_it, len):
        t1 = 2000
        t2 = 400
        
        if n_int < t1 :
            pass #alpha = alpha 
        elif t1 < n_int < t2:
            pass #alpha = alpha*(t-t1)/(t2-t1) 
        else:
            alpha = 0
            betha = 0

        #Calculate Euclidient Loss for Visibility, Landmark Positions and Labels