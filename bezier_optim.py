#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import numpy as np
import matplotlib.pyplot as plt
from math import comb
import scipy.optimize as opt
import time
import scipy.stats as st
    
# from scipy.special import comb


def get_bezier_curve(theta_in,dt):
    """Return a set of coordinates from Bezier control points input."""

    n_pt = int((theta_in.size - 2)/2) # number of points : theta contains weight, sigma, and n_pt 2D points  
    
    points = np.reshape(theta_in[2:], (n_pt, 2))

    bezier = np.zeros(shape=(dt.size,2))
    for i in range(n_pt):
        binom = comb(n_pt-1,i)
        coeff = binom * dt **i * (1-dt)**(n_pt - i-1)
        bezier += points[i]*coeff
    return bezier

def get_segment_curve(theta_in,dt):
    """Return a set of coordinates from control points input, to form a 
    piecewise linear curve. """
    n_pt = int((theta_in.size - 2)/2) # number of points : theta contains weight, sigma, and n_pt 2D points  
    
    points = np.reshape(theta_in[2:], (n_pt, 2))

    curve = np.zeros(shape=(dt.size*(n_pt-1),2))
    
    nt = dt.size

    for i in range(n_pt-1):
        curve[i*nt:(i+1)*nt,0] = np.linspace(points[i,0],points[i+1,0],dt.size)
        curve[i*nt:(i+1)*nt,1] = np.linspace(points[i,1],points[i+1,1],dt.size)
        
    return curve

def is_beyond_limit(theta_in,dt,curve_type,P,Q):
    """Test if the curve foes beyond the image."""
    if curve_type=='bezier':
        bezier = get_bezier_curve(theta_in,dt)
    else:
        bezier = get_segment_curve(theta_in,dt)
    
    beyond_limit = ( (bezier > (P-1)).sum() + (bezier< 0).sum()) > 0
    return beyond_limit

def get_bezier_image(theta_in, dx_flat,dy_flat,dt,P,Q,curve_type='bezier'):
    """Image formation operator.
    """
    width = theta_in[1]
    weight = theta_in[0]
    
    
    if curve_type=='bezier':
        bezier = get_bezier_curve(theta_in,dt)
    else:
        bezier = get_segment_curve(theta_in,dt)

    
    # All distances to the curve
    all_distance = ((dx_flat - bezier[:,1])**2 +(dy_flat - bezier[:,0])**2)**0.5
    # Minimal distance, for any image pixel
    dist_flat = np.min(all_distance,axis=1)
    dist = dist_flat.reshape(P,Q)
    
    # Final image
    shape = weight*np.exp(-0.5 * dist**2 / width**2)
    return shape

# def get_bezier_image_variable(theta_in, dx_flat,dy_flat,dt,P,Q,variable_width):

#     width = theta_in[1]
#     weight = theta_in[0]

#     bezier = get_bezier_curve(theta_in,dt)
    
#     # All distances to the curve
#     all_distance = ((dx_flat - bezier[:,1])**2 +(dy_flat - bezier[:,0])**2)**0.5
    
#     # Minimal distance, for any image pixel
#     dist_flat = np.min(all_distance,axis=1)
#     argmin_dist_flat = np.argmin(all_distance,axis=1)
#     variable_width_here = variable_width[argmin_dist_flat].reshape(P,Q)*width

#     dist = dist_flat.reshape(P,Q)
    
#     # Final image
#     shape = weight*np.exp(-0.5 * dist**2 / variable_width_here**2)
#     return shape


# def generate_synthetic_image(P,Q,n_pt,weight, sigma, width_min,width_max, seed=None ):
#     if seed is not None :
#         np.random.seed(seed)
    
#     dy,dx = np.mgrid[0:P,0:Q]
    
#     dx_flat = dx.flatten().reshape(-1,1)
#     dy_flat = dy.flatten().reshape(-1,1)
#     dt = np.linspace(0,1,1000).reshape(-1,1)
    
#     # variable width as a function of dt
#     shift = 0.3 + np.random.rand()*0.4
#     sine = (1-np.sin(dt*np.pi+shift))
#     sine = (sine - sine.min()) / (sine.max()-sine.min())
#     sine = 1-np.exp(-0.5* (dt-shift)**4 *1000)
#     sine = 1-np.exp(-0.5* (dt-shift)**4 *500)
    
#     variable_width = width_min + (width_max-width_min)*(sine)
    

        
#     Points = np.zeros(shape=(n_pt,2))
#     Points[:,1] = np.linspace(0,P,n_pt)
#     Points[1:-1,1] += np.random.rand(n_pt-2) * 20-10
#     Points[:,0] = np.random.rand(n_pt)*P
#     Points[Points < 0] = 0 ; Points[Points>P] = P-1
    
#     theta_true = np.append( np.array([weight,sigma]).reshape(1,2),Points,axis=0).flatten()

#     ## Image generation
#     shape =  get_bezier_image_variable(theta_true, dx_flat,dy_flat,dt,P,Q,variable_width)
    
#     return shape


def make_complete_theta(theta_in,P_start,P_end):
    """ Link the parameter to optimize (curve without endoints) to the Bezier 
    curve parameter (all control points).
    """
    first_line = np.array([theta_in[0],theta_in[1]]).reshape(1,2)
    second_line =  P_start.reshape(1,2)
    last_line = P_end.reshape(1,2)
    
    
    n_pt =  int((theta_in.size - 2)/2) # number of intermediate points : theta contains weight, sigma, and n_pt 2D points  
    inbetween_lines = theta_in[2:].reshape(n_pt,2)
    
    theta_complete = np.zeros(shape=(n_pt + 3,2))
    theta_complete[0] = first_line
    theta_complete[1] = second_line
    theta_complete[-1] = last_line
    theta_complete[2:-1] = inbetween_lines
    
    return theta_complete.flatten()
    

# def gradient_ana(theta_in,P_start,P_end, observation, dx_flat,dy_flat,dt,P,Q):
    
#     theta_complete = make_complete_theta(theta_in,P_start,P_end)
    
#     width = theta_complete[1]
#     weight = theta_complete[0]
    
#     n_pt =  int((theta_in.size - 2)/2) # nb of intermediate points
#     gradient_tab = np.zeros(shape=(n_pt+1,2))
    

#     bezier = get_bezier_curve(theta_complete,dt)
#     ## All distances to the curve
#     all_distance = ((dx_flat - bezier[:,1])**2 +(dy_flat - bezier[:,0])**2)**0.5
#     ## Minimal distance, for any image pixel
#     dist_flat = np.min(all_distance,axis=1)
#     dist = dist_flat.reshape(P,Q)
#     ## Final image
#     shape = weight*np.exp(-0.5 * dist**2 / width**2)

#     residual = shape - observation # actually that's minus residual

#     # gradient along sigma - the width
#     allgrad_sigma = 1 / width**3 * dist**2 * shape * 2*residual
#     grad_sigma = allgrad_sigma.sum()


#     gradient_tab[0,1] = grad_sigma
#     gradient_tab[0,0] = (2/weight*shape *residual).sum()
    
#     # gradient along P0
#     arg_min = np.argmin(all_distance,axis=1)
#     ts = dt[arg_min.reshape(P,Q)][:,:,0].flatten()

#     B_ts = get_bezier_curve(theta_complete,ts.flatten().reshape(-1,1)).T
#     pixels = np.array([dy_flat,dx_flat])[:,:,0]

#     allgrad_common_u =  - 1/(2*width**2) *  2*(B_ts[0]-pixels[0]) * (shape * 2*residual).flatten()
#     allgrad_common_v =  - 1/(2*width**2) *  2*(B_ts[1]-pixels[1]) * (shape * 2*residual).flatten()

#     n_actual = n_pt+2
#     for i in range(1,n_actual-1):
#         binom = comb(n_actual-1,i)
        
#         coeff = binom * ts **(i) * (1-ts)**(n_actual - i-1)
#         gradient_tab[i,0] = (allgrad_common_u *coeff).sum()
#         gradient_tab[i,1] = (allgrad_common_v *coeff).sum()
        
#     gradient = gradient_tab.flatten()

#     return gradient

def disp_bezier(theta, c1,c2,dx_flat,dy_flat,dt,P,Q,curve_type,extent=None):
    """Helper display function"""
    if curve_type=='bezier':
        bezier = get_bezier_curve(theta,dt)
    else:
        bezier = get_segment_curve(theta,dt)
    #bezier = bo.get_bezier_curve(theta_est,dt)
    shape = get_bezier_image(theta, dx_flat,dy_flat,dt,P,Q,curve_type)
    
    n_pt = int((theta.size - 2)/2)
    
    if extent is None:
        plt.imshow(shape)
    else:
        plt.imshow(shape,extent=extent)
    plt.plot(bezier[:,1],bezier[:,0],'-',c=c1)
    points = np.reshape(theta[2:], (n_pt, 2))
    for i in range(n_pt):
        plt.plot(points[i,1],points[i,0],'o',c=c2)
        
def run_bfgs(theta_start,Y, P_start,P_end,bounds, dx_flat,dy_flat,dt,P,Q,curve_type,jac=None):
    """ BFGS part of the optimization scheme."""

    #print("BFGS...")
    #start=time.time()
    out = opt.minimize(criterion,x0=theta_start,
                       args = (P_start,P_end,Y, dx_flat,dy_flat,dt,P,Q,curve_type),
                       method='L-BFGS-B',
                       jac=jac,
                       bounds=bounds  # L-BFGS-B need bounds
                      )
    #end = time.time()-start
    #print("...end at %.4f s."%end)
    theta_out = out.x
    
    theta_est = make_complete_theta(theta_out,P_start,P_end)
    X_est = get_bezier_image(theta_est, dx_flat,dy_flat,dt,P,Q,curve_type)
    
    return theta_est,X_est


def fit_bounds(theta,bounds):
    """ Truncate the parameter values so as to stay within bounds of interest.
    """
    theta_cut = theta.copy()
    for i in range(theta.size):
        theta_cut[i] = min( max(theta_cut[i], bounds[i,0]),bounds[i,1])
   
    return theta_cut



def criterion(theta_in,P_start,P_end, observation, dx_flat,dy_flat,dt,P,Q,curve_type='bezier'):
    """
    Criterion over which optimization is made.
    """
    theta_complete = make_complete_theta(theta_in,P_start,P_end)
    if is_beyond_limit(theta_in,dt,curve_type,P,Q):
        criterion = 1e10
    else:
        estimated_image = get_bezier_image(theta_complete.flatten(), 
                                           dx_flat,dy_flat,dt,P,Q,curve_type)
        
        criterion = np.linalg.norm( (observation-estimated_image))**2
    #return   np.linalg.norm( (estimated_image - observation))**2 * 1/np.linalg.norm( observation)**2
    return criterion
    #return  1/observation.size *  (np.linalg.norm( (observation-estimated_image))**2 + 10*np.linalg.norm(estimated_image)**2)

def grid_search_3pt(P_start,P_end,Y,dx_flat,dy_flat,P,Q,curve_type,bounds,verbose=False):
    """Find a 3-point bezier curve by grid search.
    Useful to ha ve a correct initializaiton.
    """
    ### Initialization
    sigma_start = 1.05
    weight_start = 2
    n_pt = 3
    
    if curve_type=='segments':
        dt = np.linspace(0,1,int(100/(n_pt-2))).reshape(-1,1)
    else:
        dt = np.linspace(0,1,100).reshape(-1,1)
    #print(dt.size)
    
    ngrid = 40
    
    # Later : to to well, use the "bounds" table
    pmin,pmax = set_pmin_pmax(P)
   
    range_x = np.linspace(pmin,pmax,ngrid)
    range_y = np.linspace(pmin,pmax,ngrid)
    C = np.zeros(shape=(ngrid,ngrid))
    for i in range(ngrid):
        for j in range(ngrid):
            if verbose:print(f"Getting theta_start... { 100*((i) * ngrid + j)/(ngrid)**2} %",end='\r')
            P_init = np.array([range_x[i],range_y[j]]).reshape(1,2)*1.0
            theta_start = np.append( np.array([weight_start,sigma_start]).reshape(1,2),P_init,axis=0).flatten()
    
            C[i,j] = criterion(theta_start, P_start,P_end,Y,dx_flat,dy_flat,dt,P,Q,curve_type)
     
    if verbose:print("")
    imin,jmin = np.where(C==C.min())
    P_init = np.array([range_x[imin],range_y[jmin]]).reshape(1,2)*1.0
    theta_grid = np.append( np.array([weight_start,sigma_start]).reshape(1,2),P_init,axis=0).flatten()
    return theta_grid

def run_metropolis_hastings(theta_start,Y, P_start,P_end,bounds, dx_flat,
                            dy_flat,dt,P,Q,curve_type='bezier',verbose=False,
                            n_iter = 500):
    """ Metropolis-Hastings part of the optimisation.
    """
    n_pt = int((theta_start.size-2)/2)
    # Scales of the Metropolis proposal
    scales = np.zeros(shape=(2*(n_pt) + 2))
    scales[:2] = 0.1
    scales[2:] = 15
    #print(n_pt,theta_start.shape,scales.shape,bounds.shape)

    if verbose:print('Metropolis-Hastings...')
    start = time.time()
    theta_courant = theta_start.copy()
    criterion_courant = criterion(theta_courant, P_start,P_end,Y,dx_flat,dy_flat,dt,P,Q,curve_type)

    acceptance_rate = 0
    for i in range(n_iter):
        random_index = np.random.randint(0,high = theta_start.size)
        mask = np.zeros_like(theta_start)
        mask[random_index]=1
        theta_prop = theta_courant + st.norm.rvs(size=theta_start.size)*scales# *mask
        theta_prop = fit_bounds(theta_prop,bounds)

        criterion_prop = criterion(theta_prop, P_start,P_end,Y,dx_flat,dy_flat,dt,P,Q,curve_type)
        
        if criterion_prop < criterion_courant :
            theta_courant = theta_prop.copy()
            criterion_courant=criterion_prop.copy()
            if verbose:print(i,criterion_courant,end='\r')
            acceptance_rate +=1
        else:
            
            ratio_prob = np.exp(-criterion_prop + criterion_courant)
            p_accept = min(1,ratio_prob)

            if np.random.rand() < p_accept:
        #if criterion_prop < criterion_courant: # changer ça si metropolis-hastings
                theta_courant = theta_prop.copy()
                criterion_courant=criterion_prop.copy()
                if verbose:print(i,criterion_courant,end='\r')
                acceptance_rate +=1
    acceptance_rate /= n_iter
    end = time.time()-start 
    
    if verbose : 
        print('')
        print("...finished in %.2f s, with a %.8f acceptance rate"%(end,acceptance_rate))
    theta_est = make_complete_theta(theta_courant,P_start,P_end)
    X_est = get_bezier_image(theta_est, dx_flat,dy_flat,dt,P,Q,curve_type)
    return theta_est,X_est,acceptance_rate   

def get_bezier_parameters(X, Y, degree=3):
    """ Least square qbezier fit using penrose pseudoinverse.

    Parameters:

    X: array of x data.
    Y: array of y data. Y[0] is the y point for X[0].
    degree: degree of the Bézier curve. 2 for quadratic, 3 for cubic.

    Based on https://stackoverflow.com/questions/12643079/b%C3%A9zier-curve-fitting-with-scipy
    and probably on the 1998 thesis by Tim Andrew Pastva, "Bézier Curve Fitting".
    """
    if degree < 1:
        raise ValueError('degree must be 1 or greater.')

    if len(X) != len(Y):
        raise ValueError('X and Y must be of the same length.')

    if len(X) < degree + 1:
        raise ValueError(f'There must be at least {degree + 1} points to '
                         f'determine the parameters of a degree {degree} curve. '
                         f'Got only {len(X)} points.')

    def bpoly(n, t, k):
        """ Bernstein polynomial when a = 0 and b = 1. """
        return t ** k * (1 - t) ** (n - k) * comb(n, k)
        #return comb(n, i) * ( t**(n-i) ) * (1 - t)**i

    def bmatrix(T):
        """ Bernstein matrix for Bézier curves. """
        return np.matrix([[bpoly(degree, t, k) for k in range(degree + 1)] for t in T])

    def least_square_fit(points, M):
        M_ = np.linalg.pinv(M)
        return M_ * points

    T = np.linspace(0, 1, len(X))
    M = bmatrix(T)
    points = np.array(list(zip(X, Y)))
    
    final = least_square_fit(points, M).tolist()
    final[0] = [X[0], Y[0]]
    final[len(final)-1] = [X[len(X)-1], Y[len(Y)-1]]
    return np.array(final)
def make_bounds(theta_start,P,P_start,P_end):
    """
    Define the search bounds in the parameter space
    """

    
    bounds = np.zeros(shape = (theta_start.size,2))
    bounds = np.zeros(shape = (theta_start.size,2))
    bounds[0,:] = 2,10 # weight = intensity
    bounds[1,:] =0.5,3 # sigma
    pmin,pmax = set_pmin_pmax(P)
    bounds[2:,0] = pmin
    bounds[2:,1] = pmax 
    return bounds



def set_pmin_pmax(P):
        return -P/2,P+P/2

def find_curve(Y,P_start,P_end,n_pt,curve_type,solver,theta_start = None,verbose=False,n_iter = 1000):
    """ 
    Core algorithm. Find a curve between to poinds givan an image, the said
    points, and other parameters.
    """
# Image dimension and grid
    P, Q = Y.shape
    dy,dx = np.mgrid[0:P,0:Q]
    dx_flat = dx.flatten().reshape(-1,1)
    dy_flat = dy.flatten().reshape(-1,1)
    

    ### Initialization
    sigma_start = 1.05
    weight_start = 2

    if curve_type=='segments':
        dt = np.linspace(0,1,int(100/(n_pt-2))).reshape(-1,1)
    else:
        dt = np.linspace(0,1,100).reshape(-1,1)
    #print(dt.size)
    
    if theta_start is None :
        P_init = np.array([np.linspace(P_start[0],P_end[0],n_pt)[1:-1],
                           np.linspace(P_start[1],P_end[1],n_pt)[1:-1]
                          ]).T

        if n_pt==3:
            P_init = 0.5*(P_start+P_end).reshape(1,2)
        theta_start = np.append( np.array([weight_start,sigma_start]).reshape(1,2),P_init,axis=0).flatten()

    # Bounds for the search space
    bounds = make_bounds(theta_start,P,P_start,P_end)

    if solver=='metropolis':
        theta_est,X_est,acceptance_rate = run_metropolis_hastings(theta_start,Y, P_start,P_end,bounds,
                                      dx_flat,dy_flat,dt,P,Q,curve_type,
                                                     verbose=verbose,n_iter=n_iter)
    elif solver=='bfgs':
        theta_est,X_est = run_bfgs(theta_start,Y, P_start,P_end,bounds,
                                      dx_flat,dy_flat,dt,P,Q,curve_type)
        acceptance_rate = 1 # dumb variable
    elif solver == 'mix':
        theta_MH,_,acceptance_rate = run_metropolis_hastings(theta_start,Y,
                                                                  P_start,P_end,bounds,
                                                                  dx_flat,dy_flat,dt,P,Q,curve_type,
                                                                 verbose=verbose,n_iter=n_iter)
        theta_MH_cut = np.zeros_like(theta_start)
        theta_MH_cut[:2] = theta_MH[:2]
        theta_MH_cut[2:] = theta_MH[4:-2]
        theta_est,X_est = run_bfgs(theta_MH_cut,Y, P_start,P_end,bounds,
                              dx_flat,dy_flat,dt,P,Q,curve_type)
        
    else:
        print('Error on the solver !')
    if verbose: print("Estimation : " + str(theta_est))

    return theta_est,X_est,dt,acceptance_rate


def find_curve_greedy(Y,P_start,P_end,n_pt_total,dx_flat,dy_flat,P,Q,curve_type,solver,verbose=False,n_iter = 1000):
    """ Greedy version of the above, presented in the paper.
    The main idea being that if a result is known at N knots, it is easier to
    find the one at N+1 knots.
    """
    
    start = time.time()
    n_pt = 3
    list_of_est = []
    
    # to generate the bounds thing
    
    theta_init = grid_search_3pt(P_start,P_end,Y,dx_flat,dy_flat,P,Q,curve_type,verbose)


    theta_est,X_est,dt,acceptance_rate = find_curve(Y,P_start,P_end,n_pt,curve_type,solver,
                                    theta_start=theta_init,verbose=verbose,n_iter=n_iter)
    list_of_est.append(theta_est)

    #


    for n_pt in range(4,n_pt_total+1): #(4,5,6,7,8,9,10):
        if verbose:
            print("")
            print("Number of points = %.0f .... "%n_pt)

        n_to_search = n_pt - 2
        
        if curve_type=='bezier':
            all_points = get_bezier_curve(theta_est,dt)
            Y_bez,X_bez = all_points[:,1],all_points[:,0]
            out = get_bezier_parameters(X_bez,Y_bez,degree=n_pt-1)
            theta_start = np.zeros(shape=(2+2*(n_to_search)))
            theta_start[:2] = theta_est[:2].copy()
            theta_start[2:] = np.array(out)[1:-1].flatten()
        else:
            #print(theta_est.shape,n_pt)
            points = theta_est[2:].reshape(n_pt-1,2)
            # Theta_est contient les extrémitiés
            # Theta_start ne les contient pas
            
            points2 = np.zeros(shape=(n_pt-2,2))
            #points2[0] = points[0] ; points2[-1] = points[-1]
            points2 = (points[0:-1] + points[1:])/2
            theta_start = np.zeros(shape=(2*(n_pt-2)+2))
            #print(points2.shape,theta_start[2:].shape)
            theta_start[:2] = theta_est[:2].copy()
            theta_start[2:] = points2.flatten()
            

  

        theta_est,X_est,dt,acceptance_rate = find_curve(Y,P_start,P_end,n_pt,curve_type,solver,
                                                        theta_start = theta_start,
                                                        verbose=verbose,
                                                        n_iter = n_iter)
        list_of_est.append(theta_est)

        if verbose: print("Estimation : " + str(theta_est))
    end = time.time() - start
    if verbose:print("Greedy search stopped in %.2f seconds."%end) 
    return list_of_est, theta_est, X_est