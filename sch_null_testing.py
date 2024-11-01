#!/usr/bin/env python

'''
Computes null geodesics. i.e. photon paths, in Schwarzschild spacetime.
The code then creates some basic visualizations of the computed null 
geodesics.

The code follows the conventions/prescriptions in the following two books:
    * Mathematical Theory of Black Holes, by Chandrasekhar.
    * Gravitation, by Misner, Thorne, & Wheeler.

To use the code the user has to specify the following inputs:
r0 = distance between the center of the black hole and the starting point of
     the light ray. This distance is given in units of gravitational radius 
     (GM/c^2) so that, for example, r0=6 means the starting location is 
     6GM/c^2 = 3 Schwarzschild radius away.

delta0 = angle that the light ray's path makes, with respect to the radially
         outward direction, at the starting location r0. delta0 has to be in
         radians. For example, r0=6 and delta0=0 means a light ray starting at
         6GM/c^2 and going directly away from the black hole. Similarly r0=6 
         and delta0 = pi means a light ray starting at 6GM/c^2 but going 
         directly towards the black hole. If r0=6 and delta0=pi/2 then light
         ray at 6GM/c^2 is moving perpendicular to the radia direction. If 
         (r0, delta0) = (6, pi/4) then the ray at 6GM/c^2 is making an angle of
         45-degrees with respect to the radially outward direction.

rMin = stop code if the distance between the ray and the center of the black
       hole becomes less than rMin. This distance is in units of gravitational
       radius (GM/c^2). If the user does not specify an rMin, then the code
       stops just outside the Schwarzschild event horizon (the default value
       of rMin is 2.000001).

rMax = stop code if the distance between the ray and the center of the black
       hole becomes larger than rMax. This distance is in units of
       gravitational radius (GM/c^2). The default value of rMax is 10.

npts = number of integration steps. Default is 10,000. If this is too small 
       then the code will stop before reaching rMin or rMax.


Code output:
r   = array of radial distances of the ray (in units of gravitational radius)
phi = array of azimuthal coordinates of the ray (in radian)
t   = array of Schwarzschild time coordinate of the ray (in units of GM/c^3)


Pre-requisites (beyond basic python):
    * numpy, matplotlib
    * Also needs the file cubic_realroots.py in the same location as this code.

To run code: 
    python sch_null.py

See detailed usage at the bottom of the code.
'''

import sys
import numpy as np
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from cubic_realroots import depressedCubicDistinctRealRoots as cr
from scipy.optimize import fsolve
import numpy.polynomial.polynomial as poly


DEBUGSW = False #True

def rTP(bSq, M):                    # location of turning points
    '''
    When b^2 > 27 M, the turning points are the positive roots of
    R^3 - b^2 R + 2 M b^2 = 0
    '''
    roots = cr(-bSq, 2.*bSq*M)  # one -ve, two +ve roots, in ascending order
    positiveRoots = roots[1:]   # select only the positive roots

    return positiveRoots


def B(r, M):
    return r / np.sqrt(1 - 2*M/r)


def Bminus2 (r, M=1.):              # 1/B^2 from pg 674 of MTW
    return (1. - 2.*M/r)/r/r


def drdl(r, bminus2, sign, M):      # d(r)/d(lambda)
    drdlSq = bminus2 - Bminus2 (r, M)
    return sign*np.sqrt(drdlSq)


def dphidl(r):                      # d(phi)/d(lambda)
    return 1./r/r


def dtdl(r, b, M):                  # d(t)/d(lambda)
    return r/b/(r - 2.*M)


def integrateRadialOutwards (r0, rMax, npts=0, M=1., phi0=0., t0=0.):
    ''' Special case: radially outward till rMax'''
    if DEBUGSW:
        print('radially outward till rMax')

    r = np.linspace(r0, rMax,npts)
    phi = np.full(npts, phi0)

    t_0 = r0 + 2.*M * np.log(r0 - 2.*M)
    t  = r  + 2.*M * np.log(r  - 2.*M) - t_0

    return r, phi, t + t0


def integrateRadialInwards (r0, rMin, npts=0, M=1., phi0=0., t0=0.):
    ''' Special case: radially inward till max(rMin, 2M+0) '''
    if DEBUGSW:
        print('radially inward till greater of rMin or 2M+0')

    rmin = (2 + 1e-6)*M
    if rMin > rmin:
        rmin = rMin

    r = np.linspace(r0, rmin, npts)
    phi = np.full(npts, phi0)

    t_0 = r0 + 2.*M * np.log(r0 - 2.*M)
    t  = t_0 - (r  + 2.*M * np.log(r  - 2.*M))

    return r, phi, t + t0


def integrateFromPeriastron (r, phi, t, r0, rMax, M=1., phi0=0., t0=0., \
        dl=1e-2, npts=0):
    '''
    Special case: r0>3M and delta0=pi/2. In this case r0 is periastron.
    '''
    if DEBUGSW:
        print('starting at periastron')

    # do the first step manually, using a small increase in r to 
    # compute resulting dl, dphi, and dt.
    dr = 1e-6*r[0]
    r[1] = r[0] + dr
    b = B(r0, M)
    bm2 = 1.0/b/b

    Bm2    = (r[1]-2*M)/r[1]/r[1]/r[1]
    dl0    = dr/np.sqrt(bm2 - Bm2)
    dphi   = dl0 * dphidl(r[0])
    phi[1] = phi[0] + dphi
    t[1]   = t[0]   + dl0 * dtdl(r[0], b, M)

    dy = r[1]*np.sin(dphi)
    r1_minus_r0 = np.sqrt(r[0]*r[0] + r[1]*r[1] - \
            2*r[0]*r[1]*np.cos(dphi))
    sindelta = dy/r1_minus_r0

    # now do the rest
    sgn=1
    for i in range(1,npts-1):
        rNew   = r[i]   + dl * drdl(r[i], bm2, sgn, M)
        phiNew = phi[i] + dl * dphidl(r[i])
        tNew   = t[i]   + dl * dtdl(r[i], b, M)
        if rNew > rMax:
            break
        r[i+1], phi[i+1], t[i+1] = rNew, phiNew, tNew
        if ((phiNew == (np.pi / 2)) and (5 <= rNew <= 100)):
            return rNew, phiNew, tNew

    if r[-1] < rMax:
        print('ran out of points before getting to rMax.')

    return 0 

   
def integrateFromApastron (r, phi, t, r0, M=1., phi0=0., t0=0., rMin=2., \
        dl=1e-2, npts=0):
    '''
    Special case: r0<3M and delta0=pi/2. In this case r0 is apastron.
    '''
    if DEBUGSW:
        print('starting at apastron')

    # go to the second  hand, using a small decrease in r to 
    # compute resulting dl, dphi, and dt.
    dr = 1e-6*r[0]
    r[1] = r[0] - dr
    b = B(r0, M)
    bm2 = 1.0/b/b

    Bm2    = (r[1]-2*M)/r[1]/r[1]/r[1]
    dl0    = dr/np.sqrt(bm2 - Bm2)
    dphi   = dl0 * dphidl(r[0])
    phi[1] = phi[0] + dphi
    t[1]   = t[0]   + dl0 * dtdl(r[0], b, M)

    dy = r[1]*np.sin(dphi)
    r1_minus_r0 = np.sqrt(r[0]*r[0] + r[1]*r[1] - \
            2*r[0]*r[1]*np.cos(dphi))
    sindelta = dy/r1_minus_r0

    # now do the rest
    sgn=-1
    for i in range(1,npts-1):
        rNew   = r[i]   + dl * drdl(r[i], bm2, sgn, M)
        phiNew = phi[i] + dl * dphidl(r[i])
        tNew   = t[i]   + dl * dtdl(r[i], b, M)
        if rNew < rMin:
            break
        r[i+1], phi[i+1], t[i+1] = rNew, phiNew, tNew

    return 0 


def integrateNoTP (r, phi, t, r0, sin_delta0, sgn, M=1., phi0=0., t0=0., \
        rMin=0., rMax=100., dl=1e-2, npts=0):
    '''
    Integrate orbits without any turning points, i.e., where the 
    sign of dr/dlambda doesn't change. 
    '''
    if DEBUGSW:
        print('using integrateNoTP function')

    b = B(r0, M) * sin_delta0

    bm2 = 1.0/b/b

    i=0
    for i in range(npts-1):
        rNew   = r[i]   + dl * drdl(r[i], bm2, sgn, M)
        phiNew = phi[i] + dl * dphidl(r[i])
        tNew   = t[i]   + dl * dtdl(r[i], b, M)
        if rNew < rMin or rNew > rMax:
            break
        r[i+1], phi[i+1], t[i+1] = rNew, phiNew, tNew
        if ((phiNew == (np.pi / 2)) and (5 <= rNew <= 100)):
            return rNew, phiNew, tNew
            

    if r[-1] < rMax:
        print('ran out of points before getting to rMax.')

    return 0 

def integrateSchGeodesic (r0, delta0, M=1., phi0=0., t0=0., \
        rMin=.5 + 1e-6, rMax=140., npts=300000):
    '''
    Integrate geodesics in Schwarzschild metric. 
    Some of these have a turning point, i.e., where the sign of dr/dlambda 
    changes at peri/apastron. The peri/apastron points are given by the 
    positive real roots of R^3 - b^2 R + 2 M b^2 = 0
    '''

    # handle some special cases first

    # Special case: radially outward till rMax
    if np.isclose(delta0, 0.):
        r, phi, t = integrateRadialOutwards (r0, rMax, npts, M, phi0, t0)
        return r, phi, t

    # Special case: radially inward till max(rMin, 2M+0) 
    if np.isclose(delta0, np.pi):
        r, phi, t = integrateRadialInwards  (r0, rMin, npts, M, phi0, t0)
        return r, phi, t
 
    if np.isclose(delta0, np.pi/2):
        sindelta0 = 1.
        sgn=0.
    else:
        sindelta0 = np.sin(delta0)
        sgn = np.sign(np.pi/2 - delta0)

    dl = 1e-2
    r   = np.full(npts, np.nan)
    phi = np.full(npts, np.nan)
    t   = np.full(npts, np.nan)

    r[0]   = r0
    phi[0] = phi0
    t[0]   = t0

    # Special case: starting at periastron and going out till rMax
    if np.isclose(delta0, np.pi/2) and r0>3*M:
        ret = integrateFromPeriastron (r, phi, t, r0, rMax, M, phi0, t0, \
                npts=npts)
        return r, phi, t

    # Special case: starting at apastron and going in till rMin
    if np.isclose(delta0, np.pi/2) and r0<3*M:
        ret = integrateFromApastron (r, phi, t, r0, M=1., phi0=0., \
                t0=0., rMin=rMin, npts=npts)
        return r, phi, t


    # calculate impact parameter for non-radial geodesics
    b = B(r0, M) * sindelta0
    bm2 = 1.0/b/b

    b_bcrit = b / (np.sqrt(27)*M)
    if np.isclose(b_bcrit, 1.):
        b_bcrit = 1.

    if b_bcrit <= 1.0:
        ret = integrateNoTP (r, phi, t, r0, sindelta0, sgn, M, phi0, t0, \
                rMin, rMax, npts=npts)
        return r, phi, t

    apa, per = rTP(b*b, M)            # get apas/periastron for this b
    if DEBUGSW:
        print('apastron = ', apa, 'periastron = ', per)

    if np.isclose(r0, per) and np.isclose(delta0, np.pi/2):
        if DEBUGSW:
            print('at periastron and can only go away')

        sgn = 1.0
        ret = integrateNoTP (r, phi, t, r0, sindelta0, sgn, M, phi0, t0, \
                rMin, rMax, npts=npts)

    elif np.isclose(r0, apa) and np.isclose(delta0, np.pi/2):
        if DEBUGSW:
            print('at apastron and can only go in')

        sgn = -1.0
        ret = integrateNoTP (r, phi, t, r0, sindelta0, sgn, M, phi0, t0, \
                rMin, rMax, npts=npts)

    elif (r0 > per and sgn > 0) or (r0 < apa and sgn < 0):
        sgn = np.sign(np.cos(delta0))
        if DEBUGSW:
            print('noTP ... dr forever:', sgn)

        ret = integrateNoTP (r, phi, t, r0, sindelta0, sgn, M, phi0, t0, \
                rMin, rMax, npts=npts)

    elif r0 < apa and sgn > 0:
        if DEBUGSW:
            print('will get to apastron and then keep going in forever')

        i=0
        sign = 1
        for i in range(npts-1):
            rNew   = r[i]   + dl * drdl(r[i], bm2, sign, M)
            phiNew = phi[i] + dl * dphidl(r[i])
            tNew   = t[i]   + dl * dtdl(r[i], b, M)

            if np.isclose(rNew, apa):
                sign = -1

            if rNew < rMin or rNew > rMax:
                break
            r[i+1], phi[i+1], t[i+1] = rNew, phiNew, tNew

    elif r0 > per and sgn < 0:
        if DEBUGSW:
            print('will get to periastron and then keep going out forever')

        i=0
        sign = -1
        for i in range(npts-1):
            rNew   = r[i]   + dl * drdl(r[i], bm2, sign, M)
            phiNew = phi[i] + dl * dphidl(r[i])
            tNew   = t[i]   + dl * dtdl(r[i], b, M)

            if np.isclose(rNew, per):
                sign = 1

            if rNew < 0 or rNew > rMax:
                break
            r[i+1], phi[i+1], t[i+1] = rNew, phiNew, tNew


    else:
        print('this should never happen!')

    return r, phi, t 



# Chandra's escape cone formula (Chandra, pg.127 eq. (244) )
def avoidAngle (r):      
    nn = np.sqrt(0.5*r - 1)
    d1 = r/3 - 1
    d2 = np.sqrt(r/6 + 1)
    dd = d1*d2
    return np.pi - np.arctan2(nn, dd) # measured from radial direction



def fit_polynomial_and_find_intercept(r, phi):
    phinew = phi[np.isfinite(phi)]
    rnew = r[np.isfinite(r)]
    
    if len(phinew) < 4 or len(rnew) < 4:
        return np.nan  # Safe fallback to avoid `None`
    
    phi_diff = np.abs(phinew - (np.pi / 2))
    sorted_indices = np.argsort(phi_diff)
    sorted_indices = [i for i in sorted_indices if phinew[i] != (np.pi / 2)]
    

    selected_indices = sorted_indices[:4]
    r_selected = rnew[selected_indices]
    phi_selected = phinew[selected_indices]
    
    x_selected = r_selected * np.cos(phi_selected)
    y_selected = r_selected * np.sin(phi_selected)
    
    coefs = poly.Polynomial.fit(x_selected, y_selected, 3).convert().coef
    
    def poly_func(x):
        return coefs[0] + coefs[1]*x + coefs[2]*x**2 + coefs[3]*x**3
    
    y_intercept = poly_func(0)
    
    if y_intercept <= 2:
        print("Ray will go into BH")
        return 0  # Indicate trajectory falls into black hole
    
    return y_intercept




def Deltax_for_y_desired(r0, inner_edge=5, outer_edge=100, deltaMax=None, deltaMin=None, rMin=.2, rMax=110, tolerance=1e-3, MaxIterations=500):
        # Function to compute the angle for a given edge (inner or outer)
    def find_required_angle(y_desired, deltaMin, deltaMax):
        for i in range(MaxIterations):
            # Try the midpoint angle in radians
            deltax = (deltaMin + deltaMax) / 2
                
            # Compute the photon trajectory
            myr, myphi, myt = integrateSchGeodesic(r0, deltax, rMin=rMin, rMax=rMax)
                
            # Find the y-axis intercept for this trajectory
            y_axis_intercept = fit_polynomial_and_find_intercept(myr, myphi)
            time_at_axis = fit_polynomial_and_find_intercept(myt, myphi)
                
            # Print the current iteration and delta if checks are needed
            #print(f"Iteration {i+1}: deltax = {np.rad2deg(deltax)} degrees, y_intercept = {y_axis_intercept}")
                
            # Check if this intercept is close enough to the target
            if np.abs(y_axis_intercept - y_desired) < tolerance:
                print(f"Converged after {i+1} iterations.")
                plotRay(myr, myphi, myt)
                print(f"Hits the disk at time = {time_at_axis}")
                return np.rad2deg(deltax)  # Return the angle in degrees
                
            # Adjust the range for the next iteration
            if y_axis_intercept > y_desired:
                deltaMin = deltax  # Try larger angles
            elif y_axis_intercept < y_desired:
                deltaMax = deltax  # Try smaller angles
            
        print(f"Failed to converge to the desired y-axis intercept for {y_desired} within tolerance.")
        return None
        
    # Find angle for the inner edge
    print(f"Finding angle for inner edge at y = {inner_edge}")
    angle_inner = find_required_angle(inner_edge, deltaMin, deltaMax)

        
    # Find angle for the outer edge
    print(f"Finding angle for outer edge at y = {outer_edge}")
    angle_outer = find_required_angle(outer_edge, deltaMin, deltaMax)
    
    print(f"Angle for inner edge (y = 5): {angle_inner} degrees")
    print(f"Angle for outer edge (y = 100): {angle_outer} degrees")
    
        
    return angle_inner, angle_outer
    


def plotRay(r, phi, t):
    x = r * np.cos(phi)
    y = r * np.sin(phi)
    rr = 1.05*np.nanmax(r)    # x/y range for plotting

    ehT = np.deg2rad(range(361))
    ehX = 2*np.cos(ehT)
    ehY = 2*np.sin(ehT)

    fig = plt.figure(figsize=(16,7))

    fig.suptitle('Ray tracing in Schwarzschild spacetime', \
            fontsize=16, fontweight='bold')
    gs1 = GridSpec(4, 16, left=-0.05, right=0.99, wspace=0.8, hspace=0.25)
    ax1 = fig.add_subplot(gs1[0:4, 0:8])
    ax2 = fig.add_subplot(gs1[0:2, 9:16])
    ax3 = fig.add_subplot(gs1[2:4, 9:16])

    ax1.plot(ehX, ehY, 'k-.', lw=1)     # Plot event horizon
    im = ax1.scatter(x, y, c=t, s=2, cmap=cm.turbo)
    cbar = fig.colorbar(im, fraction=0.046, pad=0.04)
    cbar.set_label(r'time [R$_{\rm g}$/c]', rotation=90, fontsize = 14)
    ax1.set_xlim(-rr, rr)
    ax1.set_ylim(-rr, rr)
    ax1.set_aspect('equal')
    ax1.set_xlabel(r'X [R$_{\rm g}$]', fontsize = 14)
    ax1.set_ylabel(r'Y [R$_{\rm g}$]', fontsize = 14)
    ax1.vlines(x=0, ymin=5, ymax=100, color='k', linestyle='solid', label='Accretion Disk')
    ax1.legend()    

    ax2.scatter(t, r, c=t, s=2, cmap=cm.turbo)
    ax2.set_ylabel(r'r [R$_{\rm g}$]', fontsize = 14)
    ax2.grid()

    ax3.scatter(t, phi, c=t, s=2, cmap=cm.turbo)
    ax3.set_xlabel(r'time [R$_{\rm g}$/c]', fontsize = 14)
    ax3.set_ylabel(r'$\phi$ [radians]', fontsize = 14)
    ax3.grid()

    plt.show()



def SchwarzschildLampPost (z, nrays):   
    # nrays from a lamppost at height z above BH
    delta0s_deg = np.linspace(0, 180, nrays)
    delta0s = np.deg2rad(delta0s_deg)
    #delta0s = np.pi*np.random.rand(nrays)

    rr = 3*z    # x/y range for plotting

    ehT = np.deg2rad(range(361))
    ehX, ehY = 2*np.cos(ehT), 2*np.sin(ehT)

    fig = plt.figure(figsize=(11,11))
    ax = fig.add_subplot()

    ax.set_xlim(-rr, rr)
    ax.set_ylim(-rr, rr)
    ax.set_aspect('equal')
    ax.set_xlabel(r'X [R$_{\rm g}$]', fontsize = 14)
    ax.set_ylabel(r'Y [R$_{\rm g}$]', fontsize = 14)

    ax.plot(ehX, ehY, 'k-.', lw=1)     # Plot event horizon

    for delta0 in delta0s:
        print('delta0 (deg) = %.3f' % np.rad2deg(delta0))
        r, phi, tt = integrateSchGeodesic (z, delta0, \
                rMin=2, rMax=5*z)
        y, x = r * np.cos(phi), r * np.sin(phi)
        ax.plot(+x, y, 'k-', lw=0.5)
        ax.plot(-x, y, 'k-', lw=0.5)

    plt.show()

    return 0



##############################################################

#
# Various test cases with [M, phi0, t0] = [1, 0, 0]
# uncomment each block and run code to see results
#

# radially outward ray
#r0, delta0, rmax = 2.5, 0, 12           
#myr, myphi, myt = integrateSchGeodesic (r0, delta0, rMax=rmax)


# radially inward ray
#r0, delta0, rmin = 6, np.pi, 2.01  
#myr, myphi, myt = integrateSchGeodesic (r0, delta0, rMin=rmin)


# ray starting at periastron
#r0, delta0, rmax, nPts = 6.0, np.pi/2, 30, 100000
#myr, myphi, myt = integrateSchGeodesic (r0, delta0, rMax=rmax, npts=nPts)


# ray starting at apastron
#r0, delta0, rmin = 2.5, np.pi/2, 2.01
#myr, myphi, myt = integrateSchGeodesic (r0, delta0, rMin=rmin)


# ray near bcrit@3M
#r0, delta0, nPts = 3., np.pi/2, 10000
#r0, delta0, nPts = 3., np.pi/2 + 0.0001, 10000
#r0, delta0, nPts = 3., np.pi/2 - 0.0001, 100000
#myr, myphi, myt = integrateSchGeodesic (r0, delta0, rMin=2, rMax=12, npts=nPts)


# ray approaching bcrit from inside 3M 
#r0     = 2.5
#delta0 = np.arcsin(np.sqrt(27-54/r0)/r0) 
#delta0 = np.arcsin(np.sqrt(27-54/r0)/r0) + 0.01 # reaches apastron, then fall onto BH
#delta0 = np.arcsin(np.sqrt(27-54/r0)/r0) - 0.01 # manages to escape to infinity
#myr, myphi, myt = integrateSchGeodesic (r0, delta0, rMin=2)


# ray approaching bcrit from outside 3M 
#r0, delta0 = 6, np.deg2rad(180-45)
#r0, delta0 = 6, np.deg2rad(180-45.1)
#r0, delta0 = 6, np.deg2rad(180-44.9)
#myr, myphi, myt = integrateSchGeodesic (r0, delta0, rMin=2, rMax=9)


# testing Chandra's escape cone formula
# a ray originating inside 3M but can't get out
#r0 = 2.5
#delta0 = avoidAngle(r0)
#myr, myphi, myt = integrateSchGeodesic (r0, delta0, rMin=2, rMax=9)
#myr, myphi, myt = integrateSchGeodesic (r0, delta0-0.0001, rMin=2, rMax=9)
#myr, myphi, myt = integrateSchGeodesic (r0, delta0+0.0001, rMin=2, rMax=9)


#r0, delta0 = 5, np.deg2rad(111.24511718749999)
#myr, myphi, myt = integrateSchGeodesic (r0, delta0, rMin=.2, rMax=101)

#plotRay(myr, myphi, myt)


# a bunch of rays from a lamppost at height z above the black hole
#ret = SchwarzschildLampPost (3.3, 31)

#ret = SchwarzschildLampPost (7, 200)

#r0, delta0 = 5, np.deg2rad(83.05)
#myr, myphi, myt = integrateSchGeodesic (r0, delta0, rMin=.2, rMax=120)
#plotRay(myr, myphi, myt)
'''
r0, delta0 = 3, np.deg2rad(72.25268555)
myr, myphi, myt = integrateSchGeodesic(r0, delta0, rMin=.2, rMax=50)


plotRay(myr, myphi, myt)


time_intercept = fit_polynomial_and_find_intercept(myt, myphi)
    
print(f"The photon's trajectory intercepts the y-axis at t = {time_intercept}")


r0, delta0 = 3, np.deg2rad(49.45268555)
myr, myphi, myt = integrateSchGeodesic(r0, delta0, rMin=.2, rMax=120)


plotRay(myr, myphi, myt)


time_intercept = fit_polynomial_and_find_intercept(myt, myphi)
    
print(f"The photon's trajectory intercepts the y-axis at t = {time_intercept}")
'''

#y_desired = 5

#deltaMax = delta0 + np.deg2rad(10)
#deltaMin = delta0 - np.deg2rad(10)

#Deltax_for_y_desired(r0, y_desired, deltaMax, deltaMin)



def find_delta_bounds(r0, rMin=.2, rMax=100):
    deltaMin = 0
    deltaMax = 0
    
    
    for i in np.arange(180, 45, -1):
        delta0max = np.deg2rad(i)
        
        myr, myphi, myt = [], [], []
        
        myr, myphi, myt = integrateSchGeodesic(r0, delta0max, rMin=.2, rMax=rMax)
        
        
        #print(delta0max)
        #print(np.nanmin(myr))
        if (np.nanmin(myr) <= (2 + .001 or 2 - .001)):
            #print("Ray goes into BH")
            continue
            
            
        elif (np.nanmin(myr) > 2):
            deltaMax = delta0max
            plotRay(myr, myphi, myt)

            break
            
            
    for i in np.arange(np.rad2deg(deltaMax), 1, -1):
        delta0min = np.deg2rad(i)
        
        myr, myphi, myt = [], [], []
                
        y_axis_intercept = 0
        
        myr, myphi, myt = integrateSchGeodesic(r0, delta0min, rMin=.2, rMax=rMax)
        
        y_axis_intercept = fit_polynomial_and_find_intercept(myr, myphi)


        if (y_axis_intercept <=100):
            continue

        elif (((y_axis_intercept > 100))):
            deltaMin = delta0min
            plotRay(myr, myphi, myt)
           
            break
    
    print(f"DeltaMax: {np.rad2deg(deltaMax)} degrees")
    print(f"DeltaMin: {np.rad2deg(deltaMin)} degrees") 
    
    
    return deltaMax, deltaMin


def Find_Time_at_Disk(r0, rMin=.2, rMax=110, angle_inner=None, angle_outer=None):
    time_at_disk, angles = [], []
    allrs, allphis, allts = [], [], []
    
    for i in np.linspace(angle_outer, angle_inner, num=50, endpoint=False):
        delta0 = np.deg2rad(i)
        
        myr, myphi, myt = integrateSchGeodesic(r0, delta0, rMin=rMin, rMax=rMax)
        
        # Intercept values with validation
        y_axis_intercept = fit_polynomial_and_find_intercept(myr, myphi)
        if y_axis_intercept is None:
            continue  # Skip if intercept not available
        
        time_at_intercept = fit_polynomial_and_find_intercept(myt, myphi)
        if time_at_intercept is None:
            continue  # Skip if intercept not available

        # Collect data points safely
        time_at_disk.append(time_at_intercept)
        angles.append(np.rad2deg(delta0))
        
        # Sub-arrays for plotting up to the y-axis intercept
        myrs = myr[:np.argmax(myphi > np.pi / 2)]
        myphis = myphi[:np.argmax(myphi > np.pi / 2)]
        myts = myt[:np.argmax(myphi > np.pi / 2)]
        
        myrs = np.append(myrs, y_axis_intercept)
        myphis = np.append(myphis, np.pi / 2)
        myts = np.append(myts, time_at_intercept)
        
        allrs.append(myrs)
        allphis.append(myphis)
        allts.append(myts)
    
    # Concatenate final arrays for plotting
    allrs = np.concatenate(allrs)
    allphis = np.concatenate(allphis)
    allts = np.concatenate(allts)
    
    return time_at_disk, angles, allrs, allphis, allts

    
    


# Call the function and assign the returned values
deltaMax, deltaMin = find_delta_bounds(5, rMin=.2, rMax=120)

# Call the function to find both angles for the inner and outer edge
angle_inner, angle_outer = Deltax_for_y_desired(5, inner_edge=5, outer_edge=100, deltaMax=deltaMax, deltaMin=deltaMin)

# Using those angles as bounds, launch n rays between those two angles onto the accretion disk and get the times to intercept
time_at_disk, angles, allrs, allphis, allts = Find_Time_at_Disk(5, rMin=.2, rMax=120, angle_inner=angle_inner, angle_outer=angle_outer)

plotRay(allrs, allphis, allts)


#def t_versus_disk_impact(r0min=3, r0max=20,):
    








   
'''
Need to plot each one of the rays in the tables, stopping once they hit the disk

Could test out making a heatmap at the smaller values?

Use np.linspace to get fixed number of points 
'''
   
    
'''

- Once r0 gets larger than 8, the graph doesn't go long enough to cross y-axis, regardless of whether I make npts = 100000, 200000, etc.

- Code won't be able to find deltaMin when r0 = 2.5, but it can find deltaMax.

- Need to know angle he wants me to increment by to plot all possible angles between inner edge delta and outer edge delta.
rather than have the increment be linear, try to do more closer to deltamax than deltamin

- The problem originally was nan values.

'''


