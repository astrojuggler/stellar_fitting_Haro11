import scipy as sp 
from astropy.constants import c
from astropy.convolution import convolve, convolve_fft, Gaussian1DKernel


ckms     = c.value/1.e3
sig2fwhm = 2.*sp.sqrt(2.*sp.log(2.))




def convolve_res(lam, flx, respow): 
	
	samp    = sp.median ( sp.diff( lam ) )
	lamc    = sp.mean ( lam )
	fwhm_aa = lamc / respow
	fwhm_px = fwhm_aa / samp
	sig_px  = fwhm_px / sig2fwhm
	print (fwhm_aa, fwhm_px, sig_px)

	kern = Gaussian1DKernel( stddev=sig_px )
	flxc = convolve(flx, kern, boundary='extend')

	return flxc




# function that does normal convolution when spectra come from spectrographs
# with different resolutions. 
def convolve_concat(flx, Nconcat, iconcat, sampingConcat, meanlamConcat, fwhm):
	
	sigma_kms = fwhm / sig2fwhm
	flxc  = sp.zeros_like(flx)  # the output vector
	
	#print ("sigma/kms", sigma_kms)

	for icat in range(Nconcat):
		ipix = iconcat == icat
		samp = sampingConcat[icat]
		lamc = meanlamConcat[icat]
		
		sigma_aa  = sigma_kms / ckms * lamc
		sigma_pix = sigma_aa / samp
		#print ("icat, Npx, sampling, clam", icat, len(flx[ipix]), samp, lamc)
		#print ("    sigma_aa, sigma_pix", sigma_aa, sigma_pix)

		kern = Gaussian1DKernel( stddev=sigma_pix )  # stddev could be width instead.  seems to depend upon version. 
		#print ("flux pixels:", flx[ipix])
		flxc[ipix] = convolve(flx[ipix], kern, boundary='extend')

	return flxc



