#!/usr/bin/env python
# purpose: to find distinct real roots of a cubic
# 
# based on: https://en.wikipedia.org/wiki/Cubic_equation
#
# 2024-05-20: First working version (DM)
#

import numpy as np

DEBUGSW = 0

def depressedCubicDistinctRealRoots(P, Q):
    '''
    Input: P, Q (both real) are coefficients of the depressed cubic
    t^3 + Pt + Q

    Output: distinct real roots of the depressed cubic, in ascending
    order.
    '''

    p, q = float(P), float(Q)           # make sure to work with floats

    delta = -(4.*p*p*p + 27.*q*q)       # discriminant

    if DEBUGSW:
        print('Discriminant =', delta)
        print( (1.5*q/p) * np.sqrt(-3./p) )

    if np.isclose(delta, 0.):           # delta = 0
        if np.isclose(p, 0.):           # p = q = 0 => triple root
            root = np.array([0.])
        else:                           # one double and one simple root
            smpRoot = 3.*q/p
            dblRoot = -0.5*smpRoot
            root = np.sort([dblRoot, smpRoot])

    elif delta < 0.0:                   # only one real root
        dd = np.sqrt(-delta/3.)/6.
        u1 = -q/2. + dd
        u2 = -q/2. - dd
        root_depressed = np.cbrt(u1) + np.cbrt(u2)
        root = np.array([root_depressed])

    else:                               # three distinct real roots
        p3 = np.sqrt(-p/3.)
        t1 = 1.5*q/p/p3
        theta = np.arccos(t1)/3.
        r1 = 2.*p3*np.cos(theta)
        r2 = 2.*p3*np.cos(theta - 2.*np.pi/3.)
        r3 = 2.*p3*np.cos(theta - 4.*np.pi/3.)
        root = np.sort([r1,r2,r3])

    return root


def cubicDistinctRealRoots(a,b,c,d):
    '''
    Input: a,b,c,d are real numbers.

    Output: distinct real roots of ax^3 + bx^2 + cx + d = 0
    in ascending order.
    '''

    # get coefficients p, q of the depressed cubic t^3 + pt + q
    p = (3.*a*c - b*b)/3./a/a
    q = (2.*b*b*b - 9.*a*b*c + 27.*a*a*d)/27./a/a/a

    root = depressedCubicDistinctRealRoots(p, q)

    # offset between real roots and roots of the depressed cubic
    offset = b/3./a

    return root - offset

'''
# test cases
realRoots = cubicDistinctRealRoots(a=1, b=-3, c=3, d=-1)    # x=1,1,1
print(realRoots)

realRoots = cubicDistinctRealRoots(a=3.0, b=9.0, c=0.0, d=-12.0)  # x=1,-2,-2
print(realRoots)

realRoots = cubicDistinctRealRoots(a=2, b=-6, c=-2, d=6)    # x=-1,1,3
print(realRoots)

realRoots = cubicDistinctRealRoots(a=1, b=0, c=0, d=-1)    # x=1 only real
print(realRoots)

realRoots = cubicDistinctRealRoots(a=4, b=-2, c=92, d=156)    # x=-1.5 only real
print(realRoots)

realRoots = cubicDistinctRealRoots(a=1, b=1, c=-5, d=-1)
print(realRoots)

realRoots = cubicDistinctRealRoots(a=.5, b=4.32, c=-2.78, d=-7)
print(realRoots)
'''

realRoots = depressedCubicDistinctRealRoots(7, -20)
print(realRoots)