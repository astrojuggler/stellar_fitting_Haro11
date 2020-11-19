#!/usr/bin/env python 

import scipy as S 
from numpy import linalg as LA 

fwhm2sigma = 1./(2.*S.sqrt(2.*S.log(2.)))




def trapezoid(x,y): 
	"""
		integrates y=f(x) using trapezoid rule (scipy) 
	"""
	dx=x[1:]-x[:-1]
	dy=y[1:]-y[:-1]
	area=(dx*dy).sum()/2. + (dx*y[:-1]).sum()
	return area




def pi_func(x):
	"""
		returns x[0] * x[1] * x[2] ... x[N-1] 
	"""
	pi=1
	for val in x: pi *= val
	return pi




def argmin_true(x):
	"""
		returns tuple of the true indices for the minimum value
	"""
	shape    = S.shape(x)
	iFlatMin = x.argmin()
	iMin     = S.zeros_like(shape)

	for i in range(len(shape)):
		remdim = pi_func(shape[i+1:])
		div, rem  = iFlatMin / remdim, iFlatMin % remdim
		iMin[i] = div
		iFlatMin = rem
	return tuple(iMin)




def argmax_true(x):
	"""
		returns tuple of the true indices for the maximum value
	"""
	shape    = S.shape(x)
	iFlatMax = x.argmax()
	iMax = S.zeros_like(shape)

	for i in range(len(shape)):
		remdim = pi_func(shape[i+1:])
		div, rem  = iFlatMax / remdim, iFlatMax % remdim
		iMax[i] = div
		iFlatMax = rem
		
	return tuple(iMax)




def chebyshev_poly(a, x): 
	"""
		returns Sum_0^N ( a_n * T_n(x) )
		where T[0] = 1.
		      T[1] = x
					T[i>1] = 2*x*T[i-1] - T[i-2]
	"""
	if not S.iterable(a):
		errors.die(mess='a not iterable')
	else: 
		T=S.ones(len(a), dtype=S.float64)

	if len(T)==1: pass
	else:
		T[1]=x
		if len(T)==2: pass
		else: 
			for i in range(2,len(T)):
				T[i]=2*x*T[i-1] - T[i-2]

	return (T*a).sum()




def pca(vals, Npc=1): 
	means = vals.mean(axis=0)
	da    = S.matrix(vals-means)
	cm    = S.cov(da, rowvar=0)
	eigs  = LA.eig(cm)
	ifv   = eigs[0].argsort()[-Npc:]
	fv    = S.matrix(eigs[1][:,ifv] )
	fd    = ( fv.transpose() * da.transpose() )#[::-1].transpose()

	print (fv.I.shape, fd.shape)
	print (type(fv), type(fd))
	rec   = (fv.transpose().I * fd).transpose()

	print ("MEANS", means.shape)
	print (means)
	print ("COVARIANCE MATRIX", cm.shape)
	print (cm)
	print ("EIGS", eigs[0].shape, eigs[1].shape)
	print (eigs)
	print ("i (field vector):", ifv)
	print ("PRICIPLE COMPNENT(s)")
	print (fv)

#print "FINAL DATA"
#print fd
#print "RECOVERED" 
#print rec

	return means, cm, eigs, fd[::-1].transpose(), S.array(rec+means)



def straddle(vec, x): 
	i_strad=S.array([], dtype=S.int32)
	dvec = vec-x
	for i in range(len(vec)-1):
		if   (dvec[i]>=0. and dvec[i+1]<0.) or (dvec[i]<=0. and dvec[i+1]>0.) : 
			i_strad=S.append(i_strad, i)
	return i_strad
	


def Schechter(L, lstar, phistar, alpha):
	s = phistar * (L/lstar)**alpha * S.exp(-L/lstar) 
	return s



def ind_nearest(arr, val):
	"""
		searches arr for point nearest to val and returns the index
	"""
	return S.argmin(S.fabs(val-arr))



def nearest(arr, val):
	"""
		searches arr for point nearest to val and returns the index
	"""
	return arr[S.argmin(S.fabs(val-arr))]



def err_div(num, dnum, den, dden):
	val  = num/den
	dval = val * S.sqrt( (dnum/num)**2 + (dden/den)**2 )
	return val, dval
