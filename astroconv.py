#!/usr/bin/env python 
""" 
	astroconv.py

	Set of unit conversion functions. 
		
		Fluxes   <--> magnitudes
		Redshift   -> luminosity distance 

		Sexadecimal -> decimal ra and dec representation

	Matthew Hayes
	Apr 14 2007, Stockholm
"""


import scipy as S
import sys
from scipy.integrate import quadrature

pc       = 3.08568025e18   # parsec in cgs
c        = 29979245800.    # speed of light in cgs
h        = 6.62606957e-27  # Planck constant in erg.s
AB_ZPT   = 48.6            # AB mag zeropoint 
ST_ZPT   = 21.10           # ST mag zeropoint
Lsun     = 3.827e33        # Lsun in cgs
SFR2LHa  = 1.265e+41       # 1Mo/yr in L(Ha) erg/s 
                           #     Kennicutt 1998 Salphter IMF 0.1-100
SFR2Q    = 9.259e+52       # 1Mo/yr in Q(HI) /s 
                           #     Kennicutt 1998 Salphter IMF 0.1-100
#SFR2LFUV = 1.21052e+40     # 1Mo/yr in L(FUV) erg/s/AA 
#                           #     Leitherer 1999, Sal IMF 0.1-100
SFR2LFUV = 1./1.4e-28     # 1Mo/yr in L(FUV) erg/s/Hz
                           #     Kennicutt 1998 Salphter IMF 0.1-100
SFR2LFIR = 2.22222e+43     # 1Mo/yr in L(FIR) erg/s
                           #     Kennicutt 1998 Salphter IMF 0.1-100 
Sr2Arcsec2 = 1./2.3504e-11 # number of arcsec2 in a Steradian

kro2sal = 1.5    # mass conversion of Kroupa IMF to Salpeter. 

def lam2kmsec(lam, lam0):
	"""
		converts velocity, relative to rest lambda, to wavelength
	"""
	return (lam-lam0)/lam0*c/1.e5




def kmsec2lam(kmsec, lam0):
	"""
		converts velocity, relative to rest lambda, to wavelength
	"""
	return (kmsec/c*1.e5*lam0)+lam0




def ab2flam(ab,lam,dab=None): 
	"""
		converts AB mags to erg/s/cm2/AA
	"""
	if dab==None:
		fnu  = ab2fnu(ab)
		flam = fnu2flam(fnu,lam)
		return flam
	else: 
		fnu,dfnu   = ab2fnu(ab, dab=dab)
		flam,dflam = fnu2flam(fnu,lam, dfnu=dfnu)
		return flam,dflam 




def ab2fnu(ab,dab=None):
	"""
		converts AB mags to erg/s/cm2/Hz
	"""
	fnu = 10.**((ab+AB_ZPT)/-2.5)
	if dab == None:
		return fnu
	else:
		dfnu = 2.303 * (dab/2.5) * fnu
		return fnu, dfnu




def flam2ab(flam,lam,dflam=None):
	"""
		converts erg/s/cm2/AA to AB mags
	"""
	if dflam == None:
		fnu = flam2fnu(flam, lam)
		ab  = fnu2ab(fnu)
		return ab
	else: 
		fnu,dfnu = flam2fnu(flam, lam, dflam=dflam)
		ab,dab   = fnu2ab(fnu,dfnu)
		return ab,dab 



def flam2fnu (flam,lam,dflam=None):
	"""
		converts /AA to /Hz 
	"""
	fnu=flam/(c*1.e8)*lam**2
	if dflam==None:
		return fnu
	else:
		dfnu = dflam/(c*1.e8)*lam**2
		return fnu,dfnu



def flam2jy (flam,lam,dflam=None):
	"""
		converts flam/AA to Jy
	"""
	jy = flam/(c*1.e8)*lam**2*1.e23
	if dflam==None:
		return jy
	else: 
		djy = dflam/(c*1.e8)*lam**2*1.e23
		return jy,djy

		




def fnu2ab(fnu, dfnu=None):
	"""
		converts erg/s/cm2/Hz to AB mags
	"""
	ab = -2.5*S.log10(fnu)-AB_ZPT
	if dfnu == None:
		return ab
	else: 
		dab = 0.434 * 2.5 * dfnu/fnu
		return ab, dab




def fnu2flam (fnu,lam,dfnu=None):
	"""
		converts /Hz to /AA 
	"""
	flam=fnu*(c*1.e8)/lam**2
	if dfnu==None: 
		return flam
	else: 
		dflam = dfnu*(c*1.e8)/lam**2
		return flam, dflam




def flam2st(flam): 
	st = -2.5 * S.log10(flam) - ST_ZPT
	return st




def st2flam(st): 
	flam = 10. ** ((st+ST_ZPT)/-2.5)
	return flam




def jy2ab(jy, djy=None):
	"""
		converts erg/s/cm2/Hz to AB mags
	"""
	fnu = jy*1.e-23
	ab = -2.5*S.log10(fnu)-AB_ZPT
	if djy == None:
		return ab
	else: 
		dab = 0.434 * 2.5 * djy/jy
		return ab, dab




def ab2jy(ab,dab=None):
	"""
		converts AB mags to erg/s/cm2/Hz
	"""
	fnu = 10.**((ab+AB_ZPT)/-2.5)
	if dab == None:
		return fnu*1.e23
	else:
		dfnu = 2.303 * (dab/2.5) * fnu
		return fnu*1.e23, dfnu*1.e23




def de2sex(ra_de, dec_de): 
	"""
		converts decimal ra and dec to sexadecimal
	"""
	#avoid dividebyzero
	if ra_de == 0.: 
		ra_de = 1.e-10
	if dec_de == 0.: 
		dec_de = 1.e-10

	ra_h = int ( S.floor(ra_de/15.) )
	ra_m = int ( S.floor(60.*(ra_de/15. - ra_h)))
	ra_s = 60.*((ra_de/15 - ra_h)*60 - ra_m)

	dec_d = int ((abs(dec_de)/dec_de)) *int (S.floor(abs(dec_de)))
	dec_m = int (S.floor((abs(dec_de)-S.floor(abs(dec_d)))*60))
	dec_s = 60*((abs(dec_de)-S.floor(abs(dec_d)))*60-dec_m) 

	return [ra_h,ra_m,ra_s,dec_d,dec_m,dec_s]




def sex2dec(rasex, decsex): 
	"""
		converts ra and dec STRINGS, 
			hh:mm:ss.ss , dd:mm:ss.ss to float decimal
	"""
	rasex_lst  = rasex .replace(" ", "").split(":")	
	decsex_lst = decsex.replace(" ", "").split(":")	

	if len(rasex_lst) != 3: 
		print ("rasex format hh:mm:ss.sss")
		sys.exit(1)
	if len(decsex_lst) != 3: 
		print ("decsex format dd:mm:ss.sss")
		sys.exit(1)

	if decsex[0] == "-": hem = -1.
	else               : hem =  1.

	ra_dec  = float(rasex_lst[0]) / 24. * 360. 
	ra_dec += float(rasex_lst[1]) / 24. * 360. / 60.
	ra_dec += float(rasex_lst[2]) / 24. * 360. / 60. / 60.

	dec_dec =  abs(float(decsex_lst[0]))
	dec_dec += float(decsex_lst[1]) / 60. 
	dec_dec += float(decsex_lst[2]) / 60. / 60. 
	dec_dec *= hem

	return ra_dec, dec_dec




def sec2rad(sec): 
	"""
		converts arcseconds to radians
	"""
	rad = sec / 3600. / 180. * S.pi
	return rad




def rad2sec(rad): 
	"""
		converts radians to arcseconds
	"""
	sec = rad * 3600. * 180. / S.pi
	return sec




def LineElem( zPrime, WM, WV ) :
	"""
		computes the line element to z'
	"""
	return 1. / S.sqrt(WM * S.power((1.+zPrime), 3.) + WV)




def z2dL ( z, H0=70., WM=0.3, WV=0.7 ) :
	""" 
		converts z to luminosity distance
	"""
	ckms=c/1.e5
	if S.size(z) == 1: 
		I = quadrature(LineElem, 0., z, args=(WM, WV))[0]
	else : 
		I = S.empty_like(z)
		for i in range(len(z)): 
			I[i] = quadrature(LineElem, 0., z[i], args=(WM, WV))[0] 

	return ckms / H0 * 1.e6 * (1.+z) * I


def z2area(z, H0=70., WM=0.3, WV=0.7 ) :
	""" 
		converts z area of circle with correspnding  dL
	"""
	dl = z2dL(z, H0, WM, WV)
	area = 4*S.pi*(dl*pc)**2
	return area


def z2dL_arb ( z1, z2, H0=70., WM=0.3, WV=0.7 ) :
	""" 
		converts z to luminosity distance
	"""
	ckms=c/1.e5
	if S.size(z1) == 1: 
		I = quadrature(LineElem, z1, z2, args=(WM, WV))[0]
	else : 
		I = S.empty_like(z1)
		for i in range(len(z1)): 
			I[i] = quadrature(LineElem, z1[i], z2[i], args=(WM, WV))[0] 

	return ckms / H0 * 1.e6 * (1.+z2) * I




def z2scale ( z, H0=70., WM=0.3, WV=0.7 ) :

  WR = 0.        # Omega(radiation)
  WK = 0.        # Omega curvaturve = 1-Omega(total)
  c = 299792.458 # velocity of light in km/sec
  Tyr = 977.8    # coefficent for converting 1/H into Gyr
  DTT = 0.5      # time from z to now in units of 1/H0
  DTT_Gyr = 0.0  # value of DTT in Gyr
  age = 0.5      # age of Universe in units of 1/H0
  age_Gyr = 0.0  # value of age in Gyr
  zage = 0.1     # age of Universe at redshift z in units of 1/H0
  zage_Gyr = 0.0 # value of zage in Gyr
  DCMR = 0.0     # comoving radial distance in units of c/H0
  DCMR_Mpc = 0.0 
  DCMR_Gyr = 0.0
  DA = 0.0       # angular size distance
  DA_Mpc = 0.0
  DA_Gyr = 0.0
  kpc_DA = 0.0
  DL = 0.0       # luminosity distance
  DL_Mpc = 0.0
  DL_Gyr = 0.0   # DL in units of billions of light years
  V_Gpc = 0.0
  a = 1.0        # 1/(1+z), the scale factor of the Universe
  az = 0.5       # 1/(1+z(object))

  h = H0/100.
  WR = 4.165E-5/(h*h)   # includes 3 massless neutrino species, T0 = 2.72528
  WK = 1-WM-WR-WV
  az = 1.0/(1+1.0*z)
  age = 0.
  n=1000         # number of points in integrals
  for i in range(n):
    a = az*(i+0.5)/n
    adot = S.sqrt(WK+(WM/a)+(WR/(a*a))+(WV*a*a))
    age = age + 1./adot

  zage = az*age/n
  zage_Gyr = (Tyr/H0)*zage
  DTT = 0.0
  DCMR = 0.0

	# do integral over a=1/(1+z) from az to 1 in n steps, midpoint rule
  for i in range(n):
    a = az+(1-az)*(i+0.5)/n
    adot = S.sqrt(WK+(WM/a)+(WR/(a*a))+(WV*a*a))
    DTT = DTT + 1./adot
    DCMR = DCMR + 1./(a*adot)

  DTT = (1.-az)*DTT/n
  DCMR = (1.-az)*DCMR/n
  age = DTT+zage
  age_Gyr = age*(Tyr/H0)
  DTT_Gyr = (Tyr/H0)*DTT
  DCMR_Gyr = (Tyr/H0)*DCMR
  DCMR_Mpc = (c/H0)*DCMR

	# tangential comoving distance
  ratio = 1.00
  x = S.sqrt(abs(WK))*DCMR
  if x > 0.1:
    if WK > 0:
      ratio =  0.5*(S.exp(x)-S.exp(-x))/x 
    else:
      ratio = S.sin(x)/x
  else:
    y = x*x
    if WK < 0: y = -y
    ratio = 1. + y/6. + y*y/120.
  DCMT = ratio*DCMR
  DA = az*DCMT
  DA_Mpc = (c/H0)*DA
  kpc_DA = DA_Mpc/206.264806 

  return kpc_DA


def meurer_beta2a1500(beta):
	return a1500
