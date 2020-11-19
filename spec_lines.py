import numpy as np
import scipy as sp
from astropy.constants import c
from filehandle import ReadStripTok as rst

ckms = c.value/1.e3


class LoadSpecFeatures:

	def __init__(self, fn, nowid=False):

		dd = rst(fn)
		self.species = [ d[0]+" "+d[1] for d in dd ]
		self.lamr    = sp.array([ float(d[2]) for d in dd ])

		#if pure line list is entered without velocity window, then dummy it
		if nowid == True:
			self.dv = sp.ones_like(self.lamr)*-1.
		else:
			self.dv = sp.array([ float(d[3]) for d in dd ])

		self.Nline   = len(self.species)



	def comp_limits(self):

		self.dlamr  = self.dv/ckms * self.lamr
		self.lamrlo = self.lamr - self.dlamr
		self.lamrhi = self.lamr + self.dlamr



	def shift_to_z(self, z):

		self.redshift = z
		self.lamo     = self.lamr   * (1.+z)
		self.dlamo    = self.dlamr  * (1.+z)
		self.lamolo   = self.lamrlo * (1.+z)
		self.lamohi   = self.lamrhi * (1.+z)




def get_windows(p):
	windowdata = {}

	for key,val in p.windows.items():
		print (key, "=>", val)

		if "leitherer11" in val:  windowdata[key] = LoadSpecFeatures(val, nowid=True)
		else                   :  windowdata[key] = LoadSpecFeatures(val)

		windowdata[key].comp_limits()
		windowdata[key].shift_to_z(p.redshift)

	return windowdata




def mask_lam_line(lam, lines, frame="obs", vout=0., maskout=True):

	# spectrum is always in the observer frame.
	# lines can either be redshifted or not.
	if frame=="rest":
		llo  = lines.lamolo * (1. - vout/ckms)
		lhi  = lines.lamohi * (1. - vout/ckms)
	elif frame=="obs":
		llo  = lines.lamrlo * (1. - vout/ckms)
		lhi  = lines.lamrhi * (1. - vout/ckms)
	else:
		print ("frame=", frame, "is not obs or rest")
		sys.exit()

	m    = sp.ones_like(lam, dtype=np.int32)
	for iline in range(lines.Nline):
		#print (lines.species[iline], llo[iline], lhi[iline])
		im    = (llo[iline]<lam) & (lam<lhi[iline])
		m[im] = 0

	if maskout == True : return m
	else               : return m^1



#  masks a bunch of standard features, including ISM absorption liens, nebular
#  lines, geocronal lines
#
def mask_standard(waveobs, waverest, dataqual, lines):

	maskMwLis = mask_lam_line(waveobs, lines['ismLis'], frame="obs")
	maskMwHis = mask_lam_line(waveobs, lines['ismHis'], frame="obs")
	maskGaLis = mask_lam_line(waveobs, lines['ismLis'], frame="rest", vout=200.)
	maskGaNeb = mask_lam_line(waveobs, lines['opNeb'], frame="rest") & \
							mask_lam_line(waveobs, lines['uvNeb' ], frame="rest")	 # optical and UV together
	maskGeo   = mask_lam_line(waveobs, lines['atmGeocor'], frame="obs")
	dataqual.astype(int)
	maskDq    = sp.where(dataqual == 0, 1, 0)
	
	maskEnd = sp.where(waveobs > 1750., 0, 1)
	gapG130M = (1266.0 < waveobs) & (waveobs < 1293.0)
	gapG160M = (1572.0 < waveobs) & (waveobs < 1591.5)
	maskGapG130M = sp.where(gapG130M, 0, 1)
	maskGapG160M = sp.where(gapG160M, 0, 1)

	iBalm     = (3000. < waverest) & (waverest < 3650)
	maskBalBreak =  sp.where(iBalm, 0, 1)

	mask = maskMwLis & maskMwHis & maskGaLis & maskGaNeb & maskBalBreak & maskGeo & maskEnd & maskGapG130M & maskGapG160M & maskDq 
	return mask

#same but for Haro11 spectra... gaps are masked differently
def mask_standard_h(waveobs, waverest, dataqual, lines):

	maskMwLis = mask_lam_line(waveobs, lines['ismLis'], frame="obs")
	maskMwHis = mask_lam_line(waveobs, lines['ismHis'], frame="obs")
	maskGaLis = mask_lam_line(waveobs, lines['ismLis'], frame="rest", vout=200.)
	maskGaNeb = mask_lam_line(waveobs, lines['opNeb'], frame="rest") & \
							mask_lam_line(waveobs, lines['uvNeb' ], frame="rest")	 # optical and UV together
	maskGeo   = mask_lam_line(waveobs, lines['atmGeocor'], frame="obs")
	dataqual.astype(int)
	maskDq    = sp.where(dataqual == 0, 1, 0)
	
	maskEndUv = sp.where((1900. > waveobs) & (waveobs > 1750.), 0, 1)
	maskEndOp = sp.where(waveobs > 8500., 0, 1)
	#gapG130M = (1266.0 < waveobs) & (waveobs < 1293.0)
	#gapG160M = (1572.0 < waveobs) & (waveobs < 1591.5)
	#maskGapG130M = sp.where(gapG130M, 0, 1)
	#maskGapG160M = sp.where(gapG160M, 0, 1)

	iBalm     = (3000. < waverest) & (waverest < 3650.)
	maskBalBreak =  sp.where(iBalm, 0, 1)

	mask = maskMwLis & maskMwHis & maskGaLis & maskGaNeb & maskBalBreak & maskGeo & maskEndUv & maskEndOp & maskDq 
	return mask


#  masks a bunch of non-standard features, specified from the textfile
#  lines, geocronal lines
#
def mask_nonstandard(waveobs, fn):

	mask = sp.ones_like(waveobs)

	if (fn):
		print ("Reading manual mask windows (to mask AWAY!), from:", fn)
		d    = rst(fn)
		for l in d:
			lamlo = float(l[0])
			lamhi = float(l[1])
			ilam  = (lamlo <= waveobs) & (waveobs <= lamhi) #MS changed from < > to <= >=
			mask[ilam] = 0.

		return mask

	else:
		print ("no mask file to read")
		return mask
