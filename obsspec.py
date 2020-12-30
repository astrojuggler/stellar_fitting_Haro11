import sys
import scipy as sp
import numpy as np
from matplotlib import pyplot as plt
from specfunc import rebin_ind, rebin_ind_err
from extin import CCM 
#MS I define another function to resample the spectra and I use these modules:
from spectres_MS import call_spectres 
from spectres import spectres

sanscale = 1.e15

#MS for saving mock data
import os

######################################################################
# Read the observed spectrum from ASCII files 
#
# Assumes data format in columns: 
#    1 - wavelength 
#    2 - flux density in erg/s/cm2/AA
#    3 - standard error on 2
#    4 - data quality such that 0 = good and >0 = bad
#
# Input:
#    1 - Filename of observed spectrum 
#    2 - shortest wavelength of synthetic spectra
#    3 - longest wavelength of synthetic spectra
#
# Output:
#    1 - lamr   1-dim NP array : raw observed wavelengths
#    2 - flam   1-dim NP array : flux densities 
#    3 - dflam  1-dim NP array : standard error on 2
#    4 - dq     1-dim NP array : Data quality
#    5 - Nlam_raw   int : Number of wavelengths
######################################################################
def read_obsspec(fn, lmin, lmax):

    d = sp.loadtxt(fn)
    
    l1  = d[:,0]
    il  = (lmin < l1) & (l1 < lmax)
    
    lam   = d[il,0][:-5]
    flam  = d[il,1][:-5] * sanscale
    dflam = d[il,2][:-5] * sanscale
    dq    = d[il,3][:-5]
    
    mindiff = sp.diff(lam).min()
    if (mindiff < 0.): 
        print ('!!!!error')
        print ('wavelength vector in', fn, 'is not monatonic')
        sys.exit(1)
    
    dflam[dflam<=0.] = 1.e30
    
    N = len(lam)
    
    return lam, flam, dflam, dq, N




class ObsSpec:

    def __init__(self, pars, lammin=900., lammax=9000.):
    
        self.pars   = pars
        self.lammin = lammin
        self.lammax = lammax
        
        
        #########################################################
        ####  Load the UV spectrum (probably from COS) 
        #########################################################
        #MS G130M
        self.lamo_uv130_raw, self.flam_uv130_raw, self.dflam_uv130_raw, self.dq_uv130_raw, self.Nlam_uv130_raw = \
            read_obsspec( self.pars.fnObsUv130, self.lammin, self.lammax ) 

        #MS G160M  
        #self.lamo_uv_raw, self.flam_uv_raw, self.dflam_uv_raw, self.dq_uv_raw, self.Nlam_uv_raw = \
         #   read_obsspec( self.pars.fnObsUv, 1410.0, 1412.0 ) 

#MS for specific use with CLUES gratings
	#G130M extends up to 1430Å but I stop at 1412 to include onyly half of the overlap with G160M	
        
        print ('Read observed ultraviolet spectrum - grating G130M', self.pars.fnObsUv130)
        print ('There are', self.Nlam_uv130_raw, 'raw wavelength elements')

        #MS G160M
        self.lamo_uv160_raw, self.flam_uv160_raw, self.dflam_uv160_raw, self.dq_uv160_raw, self.Nlam_uv160_raw = \
            read_obsspec( self.pars.fnObsUv160, self.lammin, self.lammax )

        print ('Read observed optical spectrum - grating G160M', self.pars.fnObsUv160)
        print ('There are', self.Nlam_uv160_raw, 'raw wavelength elements')
        
        
        #########################################################
        ####  Load the optical spectrum (probably from SDSS)
        #########################################################
	#MUSE optical spectrum extracted from an aperture of same size as COS
        self.lamo_op_raw, self.flam_op_raw, self.dflam_op_raw, self.dq_op_raw, self.Nlam_op_raw = \
            read_obsspec( self.pars.fnObsOp, self.lammin, self.lammax )  

	#MUSE optical spectra have units of 10^-20 erg/s/cm^2/Angstrom
        self.flam_op_raw  *= 1e-20
        self.dflam_op_raw *= 1e-20

#MS for specific use with CLUES gratings
	#G160M extends down to 1395Å but I stop at 1412 to include onyly half of the overlap with G130M	
        
        # this hack removes the last stripN elements of the SDSS spectrum as it
        # seems to screw up the spline interpolation.
        # MS: I don't use SDSS spectra, I comment out the strip thing  
        stripN = 5
        self.lamo_op_raw  =	self.lamo_op_raw [:-stripN]
        self.flam_op_raw  =	self.flam_op_raw [:-stripN]
        self.dflam_op_raw =	self.dflam_op_raw[:-stripN]
        self.dq_op_raw    =	self.dq_op_raw   [:-stripN]
        self.Nlam_op_raw  =	self.Nlam_op_raw -stripN
        
        print ('Read observed optical spectrum', self.pars.fnObsOp)
        print ('There are', self.Nlam_op_raw, 'raw wavelength elements')
        
        
        #################################################
        ####  Correct for MW extinction
        #################################################
        self.klam_uv130_raw = CCM(self.lamo_uv130_raw)
        self.klam_uv160_raw = CCM(self.lamo_uv160_raw)
        self.klam_op_raw = CCM(self.lamo_op_raw)
        self.fesc_uv130_raw = 10.** ( -0.4 * self.klam_uv130_raw * self.pars.ebvMw )
        self.fesc_uv160_raw = 10.** ( -0.4 * self.klam_uv160_raw * self.pars.ebvMw )
        self.fesc_op_raw = 10.** ( -0.4 * self.klam_op_raw * self.pars.ebvMw )
        
        self.flam_uv130_raw  /= self.fesc_uv130_raw
        self.dflam_uv130_raw /= self.fesc_uv130_raw
        self.flam_uv160_raw  /= self.fesc_uv160_raw
        self.dflam_uv160_raw /= self.fesc_uv160_raw
        self.flam_op_raw  /= self.fesc_op_raw
        self.dflam_op_raw /= self.fesc_op_raw
        
        print ('UV and optical spectra are corrected for MW extinction') 
        print ('  Adopted E(B-V) :', self.pars.ebvMw)
        
        
        #MS this is not MW correction, is just inflating artificially the error
        #################################################
        ####  Correct for MW extinction
        #################################################
        self.dflam_uv130_raw *= self.pars.scaleErrUv130
        self.dflam_uv160_raw *= self.pars.scaleErrUv160
        self.dflam_op_raw *= self.pars.scaleErrOp
        
        print ('UV and optical errors are corrected. Adopted values:') 
        print ('  UV G130M: {:1.3f}   UV G160M: {:1.3f}   Optical: {:1.3f}'.format(
               self.pars.scaleErrUv130, self.pars.scaleErrUv160, self.pars.scaleErrOp))
    
    
   
 
    def resample(self):
    
        N = self.pars.NoversampCos	
        #################################################
        ####  Resample the UV spectrum 
        #################################################
        self.lamo_uv_reb  = rebin_ind    (self.lamo_uv_raw , N)
        self.flam_uv_reb  = rebin_ind    (self.flam_uv_raw , N)
        self.dflam_uv_reb = rebin_ind_err(self.dflam_uv_raw, N)
        self.dq_uv_reb    = rebin_ind    (self.dq_uv_raw   , N)

        self.Nlam_uv_reb  = len(self.lamo_uv_reb)

        #the below line is commented out because we don't know exactlu what to do
        #with the noise correlation. 
        #
        #self.dflam_uv_reb = rebin_ind(self.dflam_uv_raw, N) * sp.sqrt(2) / sp.sqrt(N)
        #self.dflam_uv_reb = rebin_ind(self.dflam_uv_raw, N)  / sp.sqrt(N)
        
        print ('Observed UV spectrum rebinned with', self.Nlam_uv_reb, 'wavelength elements')
        
        
        #################################################
        ####  Do not resample the optical spectrum, just fake it -> MS: I upload a different UV grating spectrum instead and actually resample it by the same factor
        #################################################
        self.lamo_op_reb  = rebin_ind    (self.lamo_op_raw, N) 
        self.flam_op_reb  = rebin_ind    (self.flam_op_raw, N)
        self.dflam_op_reb = rebin_ind_err(self.dflam_op_raw, N)
        self.dq_op_reb    = rebin_ind    (self.dq_op_raw, N)
        
        self.Nlam_op_reb  = len(self.lamo_op_reb)
        #print ('Observed Optical spectrum not actually rebinned.', self.Nlam_op_reb, 'wavelength elements')
        print ('Observed Optical spectrum rebinned with', self.Nlam_op_reb, 'wavelength elements')

        
        #################################################
        ####  Concatenate the UV and Optical spectra
        #################################################
        self.lamo_reb     = sp.append( self.lamo_uv_reb , self.lamo_op_reb )
        self.flam_reb     = sp.append( self.flam_uv_reb , self.flam_op_reb )
        self.dflam_reb    = sp.append( self.dflam_uv_reb, self.dflam_op_reb)
        self.dq_reb       = sp.append( self.dq_uv_reb   , self.dq_op_reb   )
        
        self.lamo_reb_log = sp.log10(self.lamo_reb)   # to save many recomputations later
        self.Nlam_reb     = self.Nlam_uv_reb + self.Nlam_op_reb
        print ('Observed UV and Optical spectra are concatenated.', self.Nlam_reb, 'wavelength elements')


        #################################################
        ####  Calculate and store the indices, sizes, and samplings
        ####  of the individual spectra.  
        #################################################
        self.lamc_uv     = self.lamo_uv_reb.mean()
        self.lamc_op     = self.lamo_op_reb.mean()
        print ('      UV central lambda: {:4.2f} AA'.format(self.lamc_uv))
        print (' Optical central lambda: {:4.2f} AA'.format(self.lamc_op))




    def impose_maxsnr(self): 
    
        snr                  = self.flam_reb / self.dflam_reb
        iSnr                 = snr > self.pars.maxSnr 
        self.dflam_reb[iSnr] = self.flam_reb[iSnr] / self.pars.maxSnr
        print ('maximum SNR of {:3.2f} imposed to the observed spectra'.format(self.pars.maxSnr))
    	
    
    
    
    def shift_to_restframe(self):
        #MS I comment out these two lines as I don't define rebinned gratings (wavelengths) separately
        #self.lamr_uv_reb  = self.lamo_uv_reb / (1. + self.pars.redshift)
        #self.lamr_op_reb  = self.lamo_op_reb / (1. + self.pars.redshift)
        self.lamr_reb     = self.lamo_reb    / (1. + self.pars.redshift)
        self.lamr_reb_log = sp.log10(self.lamr_reb)   # to save many (possible?) recomputations later
        print ('observed spectra shifted to the restframe using z={:1.5f}'.format(self.pars.redshift))




    def plot(self, wave='both', frame='obs'):
    
        if   (frame == 'obs' ):
            luv  = self.lam_uv_reb
            lop  = self.lam_op_reb
            ltt  = self.lam_reb
        elif (frame == 'rest'):
            luv  = self.lam_uv_rest
            lop  = self.lam_op_rest
            ltt  = self.lam_rest
        else: 
            print ('!!!!error.')
            print ('    frame =', frame, 'is not either \'obs\' or \'rest\'')
            sys.exit(1)
        
        f = plt.figure(figsize=(10,4))
        a = f.add_subplot(111, ylim=[-3,50])
        
        if   (wave == 'both') :  
            a.plot(luv , self.flam_uv_reb  , ls='steps-', c='indigo'   , label='uv flx' )
            a.plot(luv , self.dflam_uv_reb , ls='steps-', c='gray'     , label='uv dflx')
            
            a.plot(lop , self.flam_op_reb  , ls='steps-', c='firebrick', label='op flx' )
            a.plot(lop , self.dflam_op_reb , ls='steps-', c='gray'     , label='op dflx')
            
            a.plot(ltt , self.flam_reb *1.1, ls='steps-', c='black'    , label='cat flx' )
            a.plot(ltt , self.dflam_reb*1.1, ls='steps-', c='gray'     , label='cat dflx')
        
        elif (wave == 'uv'):  
            a.plot(luv , self.flam_uv_reb  , ls='steps-', c='indigo'   , label='uv flx' )
            a.plot(luv , self.dflam_uv_reb , ls='steps-', c='gray'     , label='uv dflx')
        
        elif (wave == 'op'):  
            a.plot(lop , self.flam_op_reb  , ls='steps-', c='firebrick', label='op flx' )
            a.plot(lop , self.dflam_op_reb , ls='steps-', c='gray'     , label='op dflx')
        
        else: 
            print ('!!!!error.')
            print ('    wave =', wave, 'is not \'uv\', \'op\' or \'both\'')
            sys.exit(1)
        
        a.legend()
        a.set_xlabel('Wavelength [ AA ]')
        a.set_ylabel('flam [ cgs/AA ]')
        a.set_title ('frame ='+frame)


    #MS classic montecarlo: generate a mock observed spectrum by varying the original one within 1 sigma
    def vary(self):
        devUv130 = sp.random.normal(0, self.dflam_uv130_raw)
        devUv160 = sp.random.normal(0, self.dflam_uv160_raw) 
        devOp = sp.random.normal(0, self.dflam_op_raw)
        self.flam_uv130_raw += devUv130
        self.flam_uv160_raw += devUv160
        self.flam_op_raw += devOp
        print("Mock spectrum generated.")

#MS for 
    def save_mock(self, i):
        ID = str(i)
        directory = 'mock_data/'
        if not os.path.exists(directory):
            os.makedirs(directory)
        fOut = open(directory + "mock_spectrum_#" + ID, 'w')
        s = "# Mock spectrum created by adding a gaussian noise to the original observed spectrum\n"
        s += "# sigma of the noise = flux error (for each spectral pixels)\n"
        s += "# Wavelength Flux  Error DataQuality\n"
        fOut.write(s)

        for ilam in range(self.Nlam_reb):
                s = '{:>20.6f}  {:>20.8e}  {:>20.8e}  {:>20.8e}\n'.format(
                    self.lamo_reb[ilam], self.flam_reb[ilam], self.dflam_reb[ilam], self.dq_reb[ilam])
                fOut.write(s)
        fOut.close()

#MS resample gratings G130M and G160M to a same grid of resolution newBin 
#MS resample optical spectrum and join it with the UV spectrum
    def resample_and_combine(self):
	#reference wavelength array        
        self.lamo_uv_reb = np.arange(int(self.lamo_uv130_raw[200]), int(self.lamo_uv160_raw[-200]), self.pars.newBin)

        print(self.lamo_uv130_raw[200], self.lamo_uv160_raw[-200])
        
	#resampling of the two gratings (I call them uv and op because of how the original code was designed)
        self.flam_uv130_reb, self.dflam_uv130_reb = call_spectres(self.lamo_uv130_raw, self.flam_uv130_raw, self.lamo_uv_reb, self.dflam_uv130_raw)
        self.flam_uv160_reb, self.dflam_uv160_reb = call_spectres(self.lamo_uv160_raw, self.flam_uv160_raw, self.lamo_uv_reb, self.dflam_uv160_raw)
        self.dq_uv130_reb = call_spectres(self.lamo_uv130_raw, self.dq_uv130_raw, self.lamo_uv_reb)
        self.dq_uv160_reb = call_spectres(self.lamo_uv160_raw, self.dq_uv160_raw, self.lamo_uv_reb)		

        print ('Observed UV G130M spectrum rebinned to', self.pars.newBin, 'wavelength bin')

        print ('Observed UV G160M spectrum rebinned to', self.pars.newBin, 'wavelength bin')

        #mask for bad dq
        dq_uv130_num = np.nan_to_num(self.dq_uv130_reb, nan=8, copy=False)
        dq_uv160_num = np.nan_to_num(self.dq_uv160_reb, nan=8, copy=False)
        bad_dq_uv130 = dq_uv130_num > 0
        bad_dq_uv160 = dq_uv160_num > 0
        
        #I set flux and errors of bad dq pixels to nan so that in the average only the good ones are considered 
        self.flam_uv130_reb[bad_dq_uv130]  = np.nan
        self.flam_uv160_reb[bad_dq_uv160]  = np.nan
        self.dflam_uv130_reb[bad_dq_uv130] = np.nan
        self.dflam_uv160_reb[bad_dq_uv160] = np.nan
        # PROBLEM: when both gratings have bad quality, weights are both zeros and sum up to zero: can't normalize!
        # SOLUTION: set both such weigths to 1
        both_bad = np.isnan(self.dflam_uv130_reb) & np.isnan(self.dflam_uv160_reb)
        self.dflam_uv130_reb[both_bad] = 1.   
        self.dflam_uv160_reb[both_bad] = 1.         
 
	#combining the two UV gratings with a weighted average
	#flux
        twof = np.c_[self.flam_uv130_reb, self.flam_uv160_reb]
        twof = np.nan_to_num(twof, copy=False)
        twoe = np.c_[self.dflam_uv130_reb, self.dflam_uv160_reb]
        twoe = np.nan_to_num(twoe, copy=False)
        twow = 1/twoe**2
        twow = np.nan_to_num(twow, copy=False, posinf=0.0)

        self.flam_uv_reb = np.average(twof, axis=1, weights = twow)
	
	#error
        self.dflam_uv_reb = np.empty_like(self.flam_uv_reb)
        overlap = np.empty_like(self.lamo_uv_reb)
        #l130 = np.empty_like(self.lamo_uv_reb)
        #l160 = np.empty_like(self.lamo_uv_reb)
        for i in range(len(self.lamo_uv_reb)):
            if np.isnan(self.flam_uv130_reb[i]):
                self.dflam_uv_reb[i] = self.dflam_uv160_reb[i]
            elif np.isnan(self.flam_uv160_reb[i]):
                self.dflam_uv_reb[i] = self.dflam_uv130_reb[i]
            else:
                self.dflam_uv_reb[i] = np.sqrt(self.dflam_uv130_reb[i]**2 + self.dflam_uv160_reb[i]**2)/2

	#dq
        twodq = np.c_[self.dq_uv130_reb, self.dq_uv160_reb]
        #if one pixle of the two gratings has good data quality I keep only that and set dq to 0
        self.dq_uv_reb = np.nanmin(twodq, axis=1)

        #MS no need to resample optical spectrum: resolution is 1.25A 

        #self.lamo_op_reb = np.arange(int(self.lamo_op_raw[0]), int(self.lamo_op_raw[-2]), self.pars.newBin)
        #self.flam_op_reb, self.dflam_op_reb = call_spectres(self.lamo_op_raw, self.flam_op_raw, self.lamo_op_reb, self.dflam_op_raw)
        #self.dq_op_reb = call_spectres(self.lamo_op_raw, self.dq_op_raw, self.lamo_op_reb)
        self.lamo_op_reb = self.lamo_op_raw
        self.flam_op_reb = self.flam_op_raw
        self.dflam_op_reb = self.dflam_op_raw
        self.dq_op_reb = self.dq_op_raw

        #print ('Observed Optical spectrum rebinned with', self.pars.newBin, 'wavelength bin')

        #MS Joining the optical spectrum to the UV

        self.lamo_reb     = sp.append( self.lamo_uv_reb , self.lamo_op_reb )
        self.flam_reb     = sp.append( self.flam_uv_reb , self.flam_op_reb )
        self.dflam_reb    = sp.append( self.dflam_uv_reb, self.dflam_op_reb)
        self.dq_reb       = sp.append( self.dq_uv_reb   , self.dq_op_reb   )

        
        self.lamo_reb_log = sp.log10(self.lamo_reb)   # to save many recomputations later
        
        self.Nlam_reb     = len(self.lamo_reb)
        print ('Observed UV and Optical spectra are concatenated.', self.Nlam_reb, 'wavelength elements')

        
        #in resample() they use self.lamo_uv_reb.mean() but it should be the same!
        self.lamc_uv130     = self.lamo_uv130_raw.mean()
        self.lamc_uv160     = self.lamo_uv160_raw.mean()
        self.lamc_op     = self.lamo_op_raw.mean()
        print ('      UV-G130M central lambda: {:4.2f} AA'.format(self.lamc_uv130))
        print ('      UV-G160M central lambda: {:4.2f} AA'.format(self.lamc_uv160))
        print (' Optical central lambda: {:4.2f} AA'.format(self.lamc_op))


#MS I want to write out a file with the final resampled spectrum and the masked applied
#MS this is four columns
#MS 1. lambda   2. flux   3. error   4. mask

    def print_masked_spectrum(self, mask, fnOut=None):    
        if fnOut : fhOut = open(fnOut, 'w')
        fhOut.write("#Final resampeld spectrum with mask used for the fit\n#\n")
        fhOut.write("#Four columns\n")
        fhOut.write("#1. lambda   2. flux   3. error   4. mask\n#\n")
    
        for ipix in range(len(mask)):
            s = '   {:4.4f}  {:1.6e}  {:1.6e}  {:>20d}\n'.format(self.lamo_reb[ipix], self.flam_reb[ipix], self.dflam_reb[ipix], int(mask[ipix]))
            fhOut.write(s)

        fhOut.close() 



















