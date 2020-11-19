""" 
	specfunc.py

	Set of functions for spectral anaysis
	Mostly rebinning and fitting
		
	Matthew Hayes
	Apr 16 2007, Stockholm
	May 20 2008, Geneva
	Nov 25 2008, Geneva
"""

import sys
import scipy as sp
from mathfunc import ind_nearest
from scipy.interpolate import interp1d
from astropy.constants import c
from astropy.convolution import convolve, convolve_fft, Gaussian1DKernel

from astroconv import ab2flam, flam2ab
from spectres import spectres





def gauss(x, sigma, mu):     
	"""
		returns simple 1D gaussian for vector x
	"""
	# normalised to 1
	g = 1./sigma/sp.sqrt(2*sp.pi) * sp.e**(-(x-mu)**2./2./sigma**2)
	return g




def lognorm(x, sigma, mu): 
	return 1./x/sigma/sp.sqrt(2.*sp.pi) * sp.e**((sp.log(x)-mu)**2./-2./sigma**2.)




def w(z):
	""" 
		the nuts and bolts for voigt()
	"""
	t = sp.array([ 0.314240376, 0.947788391, 1.59768264, \
					2.27950708, 3.02063703, 3.8897249])
	c = sp.array([ 1.01172805, -0.75197147, 1.2557727e-2, \
					1.00220082e-2,-2.42068135e-4, 5.00848061e-7 ])
	s = sp.array([ 1.393237, 0.231152406, -0.155351466, \
					6.21836624e-3,9.19082986e-5, -6.27525958e-7 ])

	x=z.real
	y=sp.absolute(z.imag)
	wr=0.
	wi=0.
	y1 = y+1.5
	y2 = y1**2.

  # Region II
	if (y < 0.85) and (sp.absolute(x) > 18.1*y+1.65):
		if (sp.absolute(x)<12): wr = sp.exp(-x*x)

		y3 = y+3
		r = x-t
		r2=r**2.
		d=1./(r2+y2)
		d1=y1*d
		d2=r*d
		wr += (y*(c * (r*d2 - 1.5*d1) + s*y3*d2) / (r2+2.25) ).sum()
		r = x + t

		r2 = r**2.
		d = 1. / (r2 + y2)
		d3 = y1*d
		d4 = r*d
		wr += (y * (c*(r*d4 - 1.5*d3) - s*y3*d4) / (r2+2.25)).sum()
		wi += (c * (d2+d4) + s*(d1-d3)).sum()

  # Region I
	else:
		r = x - t
		d = 1 / (r * r + y2)
		d1 = y1 * d
		d2 = r * d
		r = x + t
		d = 1 / (r * r + y2)
		d3 = y1 * d
		d4 = r * d
		wr += (c * (d1 + d3) - s * (d2 - d4)).sum()
		wi += (c * (d2 + d4) + s * (d1 - d3)).sum()

	w = wr+wi*1.j
	return w


def voigt(x, sigma, gamma): 
	""" 
		the voigt profile
	"""
	z=(x+ gamma*1.j) / (sigma*sp.sqrt(2.))

	if sp.iterable(z): 
		V=sp.zeros_like(z)
		for i in range(len(z)):
			V[i]=w(z[i]).real / (sigma*sp.sqrt(2.*sp.pi))
	else:
		V=w(z).real / (sigma*sp.sqrt(2.*sp.pi))
	return V


		

def convolve(y, h):
	"""
		performs simple 1-D convolution with arbitrary kernel
	"""
	yh=sp.zeros_like(y)
	for i in range(len(y)):
		yh[i] = (sp.r_[y[i-len(h)+1:],y[:i-len(h)+1]][::-1] * h).sum()
	return yh



## convolves the array of fluxes with a single velocity, where the ll vector may
## be arbitrarily spaced.  
## 
## - the function makekerns generates the vector of kernels 
## - the function convolve_kernset does the convolution
def makekerns(ll, fwhm, retpx=False):

	ckms  = c.value/1.e3
	sig   = fwhm / ( 2.*sp.sqrt(2.*sp.log(2) )  ) 
	dllaa = sig/ckms*ll

	samplingaa = ll[1:]-ll[:-1]
	samplingaa = sp.append(samplingaa[0], samplingaa)

	sigpx = dllaa/samplingaa 

	kernlist = []
	for ss in sigpx: kernlist.append( Gaussian1DKernel( stddev=ss ) ) 

	if retpx==True:
		return kernlist, sigpx
	else:
		return kernlist


def convolve_kernset(ff, ll, kernlist):

	ffc = sp.zeros_like(ff)	
	for ilam in range(len(ll)):
		t = convolve_fft(ff, kernlist[ilam])
		ffc[ilam] = t[ilam]
	return ffc



def gaussconv(x, y, sigma):
	"""
		performs gaussian convolution using the above gauss() function 
		to generate the kernel and convolve() to do the convolution
	"""
	tol=1.e-5 #tolerance for floats to be called "the same"

	# check for x linear and equally spaced
	for i in range(len(x)-2):
		dx1=x[i+1]-x[i]
		dx2=x[i+2]-x[i+1]
		if sp.fabs((dx1-dx2)/dx1) > tol : 
			print ("error! gaussconv x not evenly spaced")
			print (dx1, dx2)
			print ("exiting here!")
			sys.exit(1)

	#build the convolution kenrnel h
	dx=x[1]-x[0]
	sigmax = sigma / dx  #sigma in index units
	h=gauss(sp.arange(len(x)), sigmax, 0.)
	halfh=len(h)/2
	h[len(h)-halfh:]=h[1:halfh+1][::-1]

	#convolve
	yh=convolve(y,h)
	return yh




def Interpolate1D(xlo,ylo,xhi,yhi,xneed):
	m=(yhi-ylo)/(xhi-xlo)
	yneed=ylo+m*(xneed-xlo)
	return yneed




def boxcar(x, N): 
	if N%2 == 0: 
		print ("!!!! error in boxcar (specfunc)")
		print ("    accepts only odd numbers")
		sys.exit()
		
	filtlim = int(N/2)
	xo = sp.zeros_like(x)

#	print ("filtlim", filtlim)
		
	for i in range(filtlim): 
#		print i, x[:i+filtlim+1]
		xo[i] = x[:i+filtlim+1].sum()

	for i in range(filtlim, len(x)-filtlim):  
#		print i, x[i-filtlim:i+filtlim+1]
		xo[i] = x[i-filtlim:i+filtlim+1].sum()

	for i in range(len(x)-filtlim, len(x)):  
#		print i, x[i-filtlim:]
		xo[i] = x[i-filtlim:].sum()

	return xo/N




def Rebin1D(xold, yold, xnew, outran=0.): 
	"""
	should elements of xnew lie outside the range of xold, 
	they are now set to outran
	"""
	if len(xold)!=len(yold): 
		print ("!!!!Error in Rebin1D")
		print ("xold and yold must be the same length")
		sys.exit(1)

	if sp.iterable(xnew): 
		ynew=sp.ones_like(xnew) * outran
		for ii_new in range(len(xnew)):
			if xnew[ii_new]<xold[0] or xnew[ii_new]>=xold[-1]:
				ynew[ii_new]=outran
			else: 
				ii_old=ind_nearest(xold, xnew[ii_new])
				if xold[ii_old]>xnew[ii_new]:
					ii_old=ii_old-1
				ynew[ii_new]=Interpolate1D(xold[ii_old],yold[ii_old],\
						xold[ii_old+1],yold[ii_old+1],xnew[ii_new])
	else: 
		if xnew<xold[0] or xnew>=xold[-1]:
			ynew=outran
		else: 
			ii_old=ind_nearest(xold, xnew)
			if xold[ii_old]>xnew:
				ii_old=ii_old-1
			ynew=Interpolate1D(xold[ii_old],yold[ii_old],\
					xold[ii_old+1],yold[ii_old+1],xnew)

	return ynew



def rebin_ind(arr,N,err=None):
	"""
	rebin 1D in a way that each pixel contributes once and only once.  Ignores the
	last remainder of the array.  
	"""
	Nbin   = int(len(arr)/N)
	#print (Nbin)
	Npe    = Nbin*N
	twod   = arr[:Npe].reshape(Nbin,N)
	binned = twod.mean(axis=1)

	if err==None: 
		return binned
	else:
		twod = err[:Npe].reshape(Nbin,N)
		ebinned = sp.sqrt((twod**2).mean(axis=1))
		return binned, ebinned




def rebin_ind_err(arr,N):
	"""
	rebin 1D in a way that each pixel contributes once and only once.  Ignores the
	last remainder of the array.  
	"""
	Nbin   = int(len(arr)/N)
	Npe    = Nbin*N
	twod   = arr[:Npe].reshape(Nbin,N)
	twods  = twod**2. 	
	binned = twods.sum(axis=1)
	noise  = sp.sqrt(binned)/N
	return noise




def resample_bin(xold, yold, xnew):
  ynew = spectres(xold, yold, xnew)
  return ynew


def resample_spline(xold, yold, xnew, kind="cubic"):
  ff = interp1d(xold, yold, kind=kind)
  ynew = ff(xnew)
  return ynew


def resample_linear(xold, yold, xnew):
  ff = interp1d(xold, yold, kind="linear")
  ynew = ff(xnew)
  return ynew




def Lsquares(x1, x2): 
	q=sp.sum((x1-x2)**2/x2)
	return q




def get_wavelength(fn): 
	"""
		Reads head fits file fn and returns a wavelength solution as an array
		Is astonishingly ugly but almost conforms to IRAF standars. 
	"""
	hdulist = pyfits.open(fn)
	header  = hdulist[0].header
	data    = hdulist[0].data

	keys    = header.ascardlist().keys()
	ctype = header['CTYPE2']	

	print (ctype)
	wats = [  key for key in keys if 'WAT'+str(2) in key ]
	mspec_sol=" "
	for wat in wats: 
		mspec_sol+=header[wat]
		if len(header[wat]) != 68: mspec_sol+=" "

	print (mspec_sol)
	mspec_sol = mspec_sol.replace("wtype=multispec spec1 = \"", "")
	mspec_sol = mspec_sol.replace("\"", "")
	mspec_sol = mspec_sol.split()
	ap     = int(mspec_sol[0])
	beam   = int(mspec_sol[1])
	dtype  = int(mspec_sol[2])
	w1     = float(mspec_sol[3])
	dw     = float(mspec_sol[4])
	nw     = int(mspec_sol[5])
	z      = float(mspec_sol[6])
	
	print ("ap :", ap, "    beam :", beam, "    dtype :", dtype  )
	print ("w1 :", w1, "    dw :", dw,     "    nw :", nw,     "    z :", z      )

	ltv1 = 0. # not present, assumed to be 0
	ltm1_1 = header['LTM1_1']

	wt= float(mspec_sol[9]) 
	w0= float(mspec_sol[10])
	typ= int(mspec_sol[11]) 

	print ("wt    :", wt)
	print ("w0    :", w0)
	print ("type  :", typ)
	if typ != 1: 
		errors.die(mess="non chebychev polynomial function not yet ready")
	else : 
		print ("chebychev polynomial")
	order = int(mspec_sol[12])
	print ("order :", order)

	ltv1=0.
	ltm1_1=1.
	l=sp.arange(w1,nw+1)
	print (l)
	p=(l-ltv1)/ltm1_1
	print ("p     :", p)
	pmin = float(mspec_sol[13]) 
	pmax = float(mspec_sol[14])
	n=(p - (pmax+pmin)/2.) / ((pmax-pmin)/2.)
	print ("n     :", n)
	x=sp.zeros(order, dtype=sp.float64)
	c=sp.zeros_like(x)
	c[0]= float(mspec_sol[15]) 
	c[1]= float(mspec_sol[16]) 
	c[2]= float(mspec_sol[17]) 
	c[3]= float(mspec_sol[18]) 
	print ("pmin  :", pmin)
	print ("pmax  :", pmax)
	print ("pmin,max :", pmin, pmax)
	print ("c     :", c)
	
	W=sp.zeros_like(n)
	for i in range(len(W)): 
		W[i] = chebyshev_poly(c, n[i])

	print ("W     :", W	)

	return	W, data




def mkline(lamvec, lamline, fline):
	"""
		returns a vector like lamvec, with flux of fline units / dlam in the 
		bin where lamline falls
	"""
	# find bin
	if (lamline<lamvec.min()) or (lamvec.max()<lamline): 
			print ("!!!!error:")
			print ("    lamline no in range spanned by lamvec")
			print ("    requested @", lamline, "... range=", lamvec.min(), "to", lamvec.max())
			sys.exit(1)

	inear = ind_nearest(lamvec, lamline)
	dlam  = lamvec[inear+1]-lamvec[inear]
	flux  = sp.zeros_like(lamvec, dtype=sp.float32)
	flux[inear] = fline/dlam

	return flux
		


# code to start loop over many dimensions 
#	Ndim    = header['WCSDIM']
#	print "there are", Ndim, "WCS headers to read"
#	types = []
#	for i in range(1,Ndim+1): types.append('CTYPE'+str(i))
#	print "    ",types
#
#	for i in range(1,Ndim+1): 
#		ctype = header[types[i-1]]
#		print "dimension:", i, ", type =", ctype
#		if ctype == 'LINEAR': 
#			errors.warn(mess='file '+' fn: WCS: '+ctype+' not implemented yet')
#		elif ctype == 'MULTISPE': 
#			pass	
#		else: 
#			errors.warn(mess='file '+' fn: WCS: '+ctype+' unknown')


def beta_interp_ab(lam1, ab1, lam2, ab2, lamt):
	beta = sp.log10(ab2flam(ab1,lam1) / ab2flam(ab2,lam2)) / sp.log10(lam1/lam2)
	return beta
