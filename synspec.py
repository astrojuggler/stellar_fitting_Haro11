import os 
import numpy as np
import scipy as sp
from scipy.interpolate import interp1d, interp2d
import matplotlib.pyplot as plt
from pandas import read_csv

from astropy import constants 
from astropy.convolution import convolve, convolve_fft, Gaussian1DKernel
from convol import convolve_concat

from specfunc import resample_bin, resample_spline, resample_linear
from mathfunc import ind_nearest
from extin import Cal

sanscale = 1.e15
secyr    = 60.*60.*24.*365.25
sig2fwhm = 2.*sp.sqrt(2.*sp.log(2.))




######################################################################
# Read the model spectrum 
#
# Input:
#    1 - Filename of Starburst99 high resolution model. 
#            can be UV (ifaspec1) or optical (hires1)
#
# Output:
#    1 - age_raw    1-dim NP array : raw ages
#    2 - lamr_raw   1-dim NP array : raw rest wavelengths
#    3 - llam_raw   2-dim NP array : raw luminsoities (Nage x Nlam)
#    4 - Nage_raw   int : Number of ages
#    5 - Nlam_raw   int : Number of wavelengths
#
#          *** note that nflx_raw has been removed 
######################################################################
def read_sb99_hires(fn):

    d = read_csv(fn, delim_whitespace=True, comment='#', header=None, names=['age','lam','llum'], usecols=[0,1,2])#,'nflx'])
    
    davs = d.lam.values
    dups = []
    for i in range(len(davs)-1):
        if (davs[i] == davs[i+1]): 
            dups.append(i)
    
    age_vec  = sp.delete( d.age.values , dups )
    lamr_vec = sp.delete( d.lam.values , dups )
    llam_vec = sp.delete( d.llum.values, dups )
    #nflx_vec = sp.delete( d.nflx.values, dups )
    
    itake_age = age_vec > 0.
    age_raw   = sp.log10(sp.unique(age_vec[itake_age]))  # strip off young ages if necessary
    lamr_raw  = sp.log10(sp.unique(lamr_vec))            # note: both age and lamr are logged
    
    Nage_raw  = len(age_raw)
    Nlam_raw  = len(lamr_raw)
    
    llam_raw  = llam_vec[itake_age].reshape((Nage_raw, Nlam_raw))
    #nflx_raw  = nflx_vec[itake_age].reshape((Nage_raw, Nlam_raw))
    
    print ('Read synthetic spectrum', fn)
    print ('  There are', Nage_raw, 'ages and', Nlam_raw, 'wavelength elements.')

    #return age_raw, lamr_raw, llam_raw, nflx_raw, Nage_raw, Nlam_raw
    return age_raw, lamr_raw, llam_raw, Nage_raw, Nlam_raw




######################################################################
# Read the ionizing photon rate
#
# Input:
#    1 - Filename of Starburst99 quanta model
#
# Output:
#    1 - age_raw    1-dim NP array : raw ages
#    2 - QHI_raw    1-dim NP array : raw Q of HI in photons/sec
#    3 - QHeI_raw   1-dim NP array : raw Q of HeI in photons/sec
#    4 - QHeII_raw  1-dim NP array : raw Q of HeII in photons/sec
#    5 - Nage_raw   int : Number of ages
######################################################################
def read_sb99_quanta(fn):

    d = read_csv(fn, delim_whitespace=True, comment='#', header=None, names=['age','qhi','d1','qhei','d2','qheii','d3','d4'])
    
    davs = d.age.values
    dups = []
    for i in range(len(davs)-1):
        if (davs[i] == davs[i+1]): 
            dups.append(i)
    
    age_raw   = sp.log10(sp.delete( d.age  .values, dups ))
    qhi_raw   =          sp.delete( d.qhi  .values, dups )
    qhei_raw  =          sp.delete( d.qhei .values, dups )
    qheii_raw =          sp.delete( d.qheii.values, dups )
    
    Nage_raw  = len(age_raw)
    
    print ('Read quanta data', fn)
    print ('  There are', Nage_raw, 'ages for the quanta.')
 
    return age_raw, qhi_raw, qhei_raw, qheii_raw, Nage_raw




######################################################################
# Read the mechanical energy rates
#
# Input:
#    1 - Filename of Starburst99 snr model
#
# Output:
#    1 - age_raw      1-dim NP array : raw ages
#    2 - powTot_raw   1-dim NP array : power in stars+SNe [ erg/sec ]
#    3 - engTot_raw   1-dim NP array : energy in stars+SNe [ erg/sec ]
#    4 - Nage_raw     int : Number of ages
#
#    *** OLD RETURNS ***
#    2 - powSn_raw    1-dim NP array : power in SNe [ erg/sec ]
#    3 - powStel_raw  1-dim NP array : power in stars [ erg/sec ]
#    4 - powTot_raw   1-dim NP array : power in stars+SNe [ erg/sec ]
#    5 - Nage_raw     int : Number of ages
######################################################################
def read_sb99_emech(fn):

    d = read_csv(fn, delim_whitespace=True, comment='#', header=None, \
        names=['age','d1','powsn','d2','d3','pow1b','d5','d6','d7','powtot','engtot'])
    #                     ALL SUPERNOVAE               TYPE IB SUPERNOVAE               ALL SUPERNOVAE           STARS + SUPERNOVAE
    #    TIME       TOTAL RATE  POWER   ENERGY    TOTAL RATE  POWER   ENERGY   TYPICAL MASS   LOWEST PROG. MASS    POWER   ENERGY
    
    davs = d.age.values
    dups = []
    for i in range(len(davs)-1):
        if (davs[i] == davs[i+1]): 
            dups.append(i)
    
    age_raw    = sp.log10(sp.delete( d.age   .values, dups ))
    powSn_raw  =          sp.delete( d.powsn .values, dups )
    powTot_raw =          sp.delete( d.powtot.values, dups )
    engTot_raw =          sp.delete( d.engtot.values, dups )
    
    powStel_raw = powTot_raw - powSn_raw 
    Nage_raw  = len(age_raw)
    
    print ('Read mechanical energy data', fn)
    print ('  There are', Nage_raw, 'ages for the O-stars and SNe.')

    #return age_raw, powSn_raw, powStel_raw, powTot_raw, Nage_raw
    return age_raw, powTot_raw, engTot_raw, Nage_raw




class SynSpec:

    def __init__(self, pars):

        self.pars    = pars
        
        self.metalvals = self.pars.metalvals 
        self.metalstrs = self.pars.metalstrs 
        self.Nmetal    = len(self.metalvals)

        self.pars.fnSynUv    = []
        self.pars.fnSynOp    = []
        self.pars.fnSynIon   = []
        self.pars.fnSynEmech = []

        #mlr = ["_std", "_high"]	#mass-loss rates in the atm. of massive stars, I want to run both kinds of libraries
        mlr = ["_high"]	#mass-loss rates in the atm. of massive stars, I want to run only high rates models

        for m in self.pars.metalstrs:
            for r in mlr: 
                smodel  = 'salp_{:s}_{:s}_z{:s}'.format(
                           self.pars.sfh, self.pars.tracks + r, m)
                smodel2 = smodel+'/'+smodel 
           
                # files ending 2 are stipped versions of the very large files ending
                # 1, which contain spectra.  other files (Q and E) remain with the 1
            	# extension as they are realatively small. 
                self.pars.fnSynUv   .append(os.path.join(self.pars.path2models, smodel2+'.ifaspec2'))
                self.pars.fnSynOp   .append(os.path.join(self.pars.path2models, smodel2+'.hires2' ))   
                self.pars.fnSynIon  .append(os.path.join(self.pars.path2models, smodel2+'.quanta1'))
                self.pars.fnSynEmech.append(os.path.join(self.pars.path2models, smodel2+'.snr1'   ))

        #for i in range(self.Nmetal):
        #    print (self.pars.fnSynUv   [i], os.path.exists(self.pars.fnSynUv   [i]))
        #    print (self.pars.fnSynOp   [i], os.path.exists(self.pars.fnSynOp   [i]))  
        #    print (self.pars.fnSynIon  [i], os.path.exists(self.pars.fnSynIon  [i]))  
        #    print (self.pars.fnSynEmech[i], os.path.exists(self.pars.fnSynEmech[i]))  
        #print ('Set up the synthetic spectral data, using the files')
        #print ('       Ultraviolet spectra :')
        #print ( [ m.split('/')[-1] for m in self.pars.fnSynUv   ])
        #print ('           Optical spectra :')
        #print ( [ m.split('/')[-1] for m in self.pars.fnSynOp   ])
        #print ('           Ionizing quanta :')
        #print ( [ m.split('/')[-1] for m in self.pars.fnSynIon  ])
        #print ('   Mechanical luminosities :')
        #print ( [ m.split('/')[-1] for m in self.pars.fnSynEmech])




    def read_syn(self):

        self.age_uv_raw   = []
        self.lamr_uv_raw  = []
        self.llam_uv_raw  = []
        #self.nflx_uv_raw  = []
        self.Nage_uv_raw  = []
        self.Nlam_uv_raw  = []

        self.age_op_raw   = []
        self.lamr_op_raw  = []
        self.llam_op_raw  = []
        #self.nflx_op_raw  = []
        self.Nage_op_raw  = []
        self.Nlam_op_raw  = []

        self.age_qu_raw   = []
        self.qhi_raw      = []
        self.qhei_raw     = []
        self.qheii_raw    = []
        self.Nage_qu_raw  = []

        self.age_em_raw   = []
        self.lmechTot_raw = []
        self.emechTot_raw = []
        self.Nage_em_raw  = []

        for imet in range(self.Nmetal): 

            #  Read the UV model spectra
            d = read_sb99_hires(self.pars.fnSynUv[imet])
            self.age_uv_raw  .append(d[0])
            self.lamr_uv_raw .append(d[1])
            self.llam_uv_raw .append(d[2])
            #self.nflx_uv_raw .append(d[3])
            self.Nage_uv_raw .append(d[3])
            self.Nlam_uv_raw .append(d[4])

            #  Read the optical model spectra
            d = read_sb99_hires(self.pars.fnSynOp[imet])
            self.age_op_raw  .append(d[0])  
            self.lamr_op_raw .append(d[1])  
            self.llam_op_raw .append(d[2])  
            #self.nflx_op_raw .append(d[3])  
            self.Nage_op_raw .append(d[3])  
            self.Nlam_op_raw .append(d[4])  

            #  Load the files of ionizing quanta
            d = read_sb99_quanta(self.pars.fnSynIon[imet])
            self.age_qu_raw  .append(d[0])  
            self.qhi_raw     .append(d[1])  
            self.qhei_raw    .append(d[2])  
            self.qheii_raw   .append(d[3])  
            self.Nage_qu_raw .append(d[4])  
            
            #  Load the files of mechanical energy
            d = read_sb99_emech(self.pars.fnSynEmech[imet])
            self.age_em_raw  .append(d[0])  
            self.lmechTot_raw.append(d[1])  
            self.emechTot_raw.append(d[2])  
            self.Nage_em_raw .append(d[3])  

        print ('Read all the input data')
 



    def redshift_syn(self):
 
        self.lamo_uv_raw  = []
        self.lamo_op_raw  = []
        self.flam_uv_raw  = []
        self.flam_op_raw  = []
       
        for imet in range(self.Nmetal): 
            self.lamo_uv_raw.append( self.lamr_uv_raw[imet] + sp.log10(1.+self.pars.redshift) )
            self.lamo_op_raw.append( self.lamr_op_raw[imet] + sp.log10(1.+self.pars.redshift) )
            self.flam_uv_raw.append( self.llam_uv_raw[imet] - sp.log10(self.pars.LoverF) + sp.log10(sanscale) )
            self.flam_op_raw.append( self.llam_op_raw[imet] - sp.log10(self.pars.LoverF) + sp.log10(sanscale) )

        self.lamo_min = self.lamo_uv_raw[0].min()
        self.lamo_max = self.lamo_op_raw[0].max()

        print ('Redshifted synthetic data to z={:1.5f}'.format(self.pars.redshift))
        print ('  corresponding lumionsity area {:1.5g} cm2 (log)'.format(sp.log10(self.pars.LoverF)))
 



    def conv2inst(self, lamcUv=1500., lamcOp=5600.): 

        # take resolving power and sampling of COS and SDSS from the observed data
#        self.pars.resPowUv

        fwhm_inst_aaUv = self.pars.fwhmUv130 #this is always assumed to be equal to fwhmUv160
        fwhm_inst_aaOp = self.pars.fwhmOp
        fwhm_vdis_aaUv = lamcUv * self.pars.fwhmLo / (constants.c.value / 1.e3)
        fwhm_vdis_aaOp = lamcOp * self.pars.fwhmLo / (constants.c.value / 1.e3)

        fwhm_aaUv = sp.sqrt(fwhm_inst_aaUv**2. + fwhm_vdis_aaUv**2.)
        fwhm_aaOp = sp.sqrt(fwhm_inst_aaOp**2. + fwhm_vdis_aaOp**2.)

        sampUv    = sp.median(sp.diff(10.**self.lamo_uv_raw[0]))
        sampOp    = sp.median(sp.diff(10.**self.lamo_op_raw[0]))

        sig_pxUv  = fwhm_aaUv / sampUv / sig2fwhm
        sig_pxOp  = fwhm_aaOp / sampOp / sig2fwhm

        kernUv    = Gaussian1DKernel( stddev=sig_pxUv )
        kernOp    = Gaussian1DKernel( stddev=sig_pxOp )
       
        for imet in range(self.Nmetal):
     
            for iage in range(self.Nage_uv_raw[imet]):  
                self.flam_uv_raw[imet][iage] = convolve(self.flam_uv_raw[imet][iage], kernUv, boundary='extend')
                #self.nflx_uv_raw[imet][iage] = convolve(self.nflx_uv_raw[imet][iage], kernUv, boundary='extend')
            
            for iage in range(self.Nage_op_raw[imet]):  
                self.flam_op_raw[imet][iage] = convolve(self.flam_op_raw[imet][iage], kernOp, boundary='extend')
                #self.nflx_op_raw[imet][iage] = convolve(self.nflx_op_raw[imet][iage], kernOp, boundary='extend')
        
        print ('Convolved the UV and optical spectra to instrumental resolution')
        print ('  UV R={:6.0f}, FWHM(ins)={:2.3f} AA, (vdis)={:2.3f} AA, Smp={:2.3f} AA/px, sig(tot)={:2.3f} px'.format(self.pars.fwhmUv130, fwhm_inst_aaUv, fwhm_vdis_aaUv, sampUv, sig_pxUv))
        print ('  Op R={:6.0f}, FWHM(ins)={:2.3f} AA, (vdis)={:2.3f} AA, Smp={:2.3f} AA/px, sig(tot)={:2.3f} px'.format(self.pars.fwhmOp, fwhm_inst_aaOp, fwhm_vdis_aaOp, sampOp, sig_pxOp))




    def merge(self): 

        self.age_raw  = []
        self.lamr_raw = []
        self.lamo_raw = []
        self.llam_raw = []
        self.flam_raw = []
        #self.nflx_raw = []
        self.Nage_raw = []
        self.Nlam_raw = []

        for imet in range(self.Nmetal): 
            self.age_raw .append(self.age_uv_raw[imet])
            self.lamr_raw.append(sp.r_[self.lamr_uv_raw[imet], self.lamr_op_raw[imet]])
            self.lamo_raw.append(sp.r_[self.lamo_uv_raw[imet], self.lamo_op_raw[imet]])
            self.llam_raw.append(sp.c_[self.llam_uv_raw[imet], self.llam_op_raw[imet]])
            self.flam_raw.append(sp.c_[self.flam_uv_raw[imet], self.flam_op_raw[imet]])
            #self.nflx_raw.append(sp.c_[self.nflx_uv_raw[imet], self.nflx_op_raw[imet]])
            self.Nage_raw.append(self.Nage_uv_raw[imet])
            self.Nlam_raw.append(self.Nlam_uv_raw[imet] + self.Nlam_op_raw[imet])

        print ('UV and optical are merged')




    def fit_splines(self):

        k  = 'cubic'
        be = True

        self.llam_spl     = []
        self.flam_spl     = []
        #self.nflx_spl     = []
        self.qhi_spl      = [] 
        self.qhei_spl     = [] 
        self.qheii_spl    = [] 
        self.lmechTot_spl = [] 
        self.emechTot_spl = [] 

        for imet in range(self.Nmetal): 
             
            # 2-D splines for the flux surfaces
            self.llam_spl.append(interp2d(self.lamo_raw[imet], self.age_raw[imet], self.llam_raw[imet], kind=k, bounds_error=be))
            self.flam_spl.append(interp2d(self.lamo_raw[imet], self.age_raw[imet], self.flam_raw[imet], kind=k, bounds_error=be))
            #self.nflx_spl.append(interp2d(self.lamo_raw[imet], self.age_raw[imet], self.nflx_raw[imet], kind=k, bounds_error=be))

            # 1-D splines for the flux surfaces Q and E vectors
            self.qhi_spl     .append(interp1d(self.age_qu_raw[imet], self.qhi_raw     [imet], kind=k, bounds_error=be))  
            self.qhei_spl    .append(interp1d(self.age_qu_raw[imet], self.qhei_raw    [imet], kind=k, bounds_error=be))  
            self.qheii_spl   .append(interp1d(self.age_qu_raw[imet], self.qheii_raw   [imet], kind=k, bounds_error=be))  
            self.lmechTot_spl.append(interp1d(self.age_em_raw[imet], self.lmechTot_raw[imet], kind=k, bounds_error=be))  
            self.emechTot_spl.append(interp1d(self.age_em_raw[imet], self.emechTot_raw[imet], kind=k, bounds_error=be))  

        print ('2-D Splines fitted to the merged UV and optical spectra')
        print ('1-D Splines fitted to the Q and E vectors')




    def gen_flam_from_splines(self, zwant, logagewant, loglamwant):
        
        if (zwant < self.metalvals[0]) or (self.metalvals[-1] < zwant):
            print ('requested Z=',zwant,'outside the range', self.metalvals[0], 'to', self.metalvals[-1])

        if zwant in self.metalvals:
            #print ('exact metallicity, no interpolation')
            iwant = sp.argwhere(self.metalvals == zwant)[0][0]
            fin = self.flam_spl[iwant](loglamwant,logagewant)

        else:
            #print ('interpolating metallicity')
            ilo = sp.argwhere((self.metalvals - zwant)<0.)[-1][0]
            ihi = ilo+1
            whi = zwant-self.metalvals[ilo]
            wlo = self.metalvals[ihi]-zwant
            flo = self.flam_spl[ilo](loglamwant,logagewant)
            fhi = self.flam_spl[ihi](loglamwant,logagewant)
            fin = ((flo*wlo) + (fhi*whi)) / (wlo+whi)

        return fin




    def gen_sec_from_splines(self, zwant, logagewant):
        
        if (zwant < self.metalvals[0]) or (self.metalvals[-1] < zwant):
            print ('requested Z=',zwant,'outside the range', self.metalvals[0], 'to', self.metalvals[-1])

        if zwant in self.metalvals:
            iwant    = sp.argwhere(self.metalvals == zwant)[0][0]
            qhi      = self.qhi_spl     [iwant](logagewant)
            qhei     = self.qhei_spl    [iwant](logagewant)
            qheii    = self.qheii_spl   [iwant](logagewant)
            lmechTot = self.lmechTot_spl[iwant](logagewant)
            emechTot = self.emechTot_spl[iwant](logagewant)
            
        else:
            ilo = sp.argwhere((self.metalvals - zwant)<0.)[-1][0]
            ihi = ilo+1
            whi = zwant-self.metalvals[ilo]
            wlo = self.metalvals[ihi]-zwant

            vallo    = self.qhi_spl[ilo](logagewant)
            valhi    = self.qhi_spl[ihi](logagewant)
            qhi      = ((vallo*wlo) + (valhi*whi)) / (wlo+whi)

            vallo    = self.qhei_spl[ilo](logagewant)
            valhi    = self.qhei_spl[ihi](logagewant)
            qhei     = ((vallo*wlo) + (valhi*whi)) / (wlo+whi)

            vallo    = self.qheii_spl[ilo](logagewant)
            valhi    = self.qheii_spl[ihi](logagewant)
            qheii    = ((vallo*wlo) + (valhi*whi)) / (wlo+whi)

            vallo    = self.lmechTot_spl[ilo](logagewant)
            valhi    = self.lmechTot_spl[ihi](logagewant)
            lmechTot = ((vallo*wlo) + (valhi*whi)) / (wlo+whi)

            vallo    = self.emechTot_spl[ilo](logagewant)
            valhi    = self.emechTot_spl[ihi](logagewant)
            emechTot = ((vallo*wlo) + (valhi*whi)) / (wlo+whi)

        return qhi, qhei, qheii, lmechTot, emechTot




    def fit_quant_mech(self):

        iageQ          = self.age_qu_raw < self.pars.logAgeHi 
        iposHi         = sp.log10(self.qhi_raw  ) > 0.
        iposHei        = sp.log10(self.qhei_raw ) > 0.
        iposHeii       = sp.log10(self.qheii_raw) > 0.
        phi            = sp.polyfit(self.age_qu_raw[iageQ&iposHi  ], sp.log10(self.qhi_raw  [iageQ&iposHi  ]), 7)
        phei           = sp.polyfit(self.age_qu_raw[iageQ&iposHei ], sp.log10(self.qhei_raw [iageQ&iposHei ]), 9)
        pheii          = sp.polyfit(self.age_qu_raw[iageQ&iposHeii], sp.log10(self.qheii_raw[iageQ&iposHeii]), 9)
        self.polyHi    = sp.poly1d(phi  )
        self.polyHei   = sp.poly1d(phei )
        self.polyHeii  = sp.poly1d(pheii)
        self.dpolyHi   = sp.polyder(self.polyHi  )
        self.dpolyHei  = sp.polyder(self.polyHei )
        self.dpolyHeii = sp.polyder(self.polyHeii)

        print ('fitted polynomials to Q(HI) and Q(HeI) vs t.  NOT Q(HeII)')
        print ('    note that these equations are are in log space')


        iageE           = self.age_em_raw < self.pars.logAgeHi 
        plmech          = sp.polyfit(self.age_em_raw[iageE], sp.log10(self.lmechTot_raw[iageE]), 9)
        pemech          = sp.polyfit(self.age_em_raw[iageE], sp.log10(self.emechTot_raw[iageE]), 9)
        self.polyLmech  = sp.poly1d(plmech)
        self.polyEmech  = sp.poly1d(pemech)
        self.dpolyLmech = sp.polyder(self.polyLmech )
        self.dpolyEmech = sp.polyder(self.polyEmech )


        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(self.age_em_raw[iageE], sp.log10(self.lmechTot_raw[iageE]), label='Lmech')
        ax.plot(self.age_em_raw[iageE], self.polyLmech(self.age_em_raw[iageE]), label='fL')
        ax.legend()
        ax.set_xlabel('time')
        ax.set_ylabel('Lmech')
        fig.savefig('fit_lmech.pdf')
        
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(self.age_em_raw[iageE], sp.log10(self.emechTot_raw[iageE]), label='Emech')
        ax.plot(self.age_em_raw[iageE], self.polyEmech(self.age_em_raw[iageE]), label='fE')
        ax.legend()
        ax.set_xlabel('time')
        ax.set_ylabel('Emech')
        fig.savefig('fit_emech.pdf')

        print ('fitted polynomials to Lmech and Emech vs t.')
        print ('    note that these equations are are in log space')

        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(self.age_qu_raw[iageQ], sp.log10(self.qhi_raw  [iageQ]), label='HI'  )
        ax.plot(self.age_qu_raw[iageQ], sp.log10(self.qhei_raw [iageQ]), label='HeI' )
        ax.plot(self.age_qu_raw[iageQ], sp.log10(self.qheii_raw[iageQ]), label='HeII')
        ax.plot(self.age_qu_raw[iageQ], self.polyHi(self.age_qu_raw  [iageQ]), label='fHI'  )
        ax.plot(self.age_qu_raw[iageQ], self.polyHei(self.age_qu_raw [iageQ]), label='fHeI' )
        ax.plot(self.age_qu_raw[iageQ], self.polyHeii(self.age_qu_raw[iageQ]), label='fHeII')
        ax.legend()
        ax.set_xlabel('time')
        ax.set_ylabel('Q')
        fig.savefig('fit_quant.pdf')




    def conv2inst_old(self): 

        # currently take the resoluving power of COS and SDSS as defaults. 
        # leave the central wavelengths hard-coded for now  
        # TODO.  change this somehow in future to generalize.
        lamcUv   = 1500.     # central wavelength of COS 
        lamcOp   = 5600.     # central wavelength of SDSS
        
        fwhm_aaUv = lamcUv / self.pars.resPowUv
        sampUv    = sp.median ( sp.diff ( self.lamr_uv_raw ) )
        sig_pxUv  = fwhm_aaUv / sampUv / sig2fwhm
        kernUv    = Gaussian1DKernel( stddev=sig_pxUv )
        
        fwhm_aaOp = lamcOp / self.pars.resPowOp
        sampOp    = sp.median ( sp.diff ( self.lamr_op_raw ) )
        sig_pxOp  = fwhm_aaOp / sampOp / sig2fwhm
        kernOp    = Gaussian1DKernel( stddev=sig_pxOp )
        
        for iage in range(self.Nage_uv_raw):  
            self.llam_uv_raw[iage] = convolve(self.llam_uv_raw[iage], kernUv, boundary='extend')
            #self.nflx_uv_raw[iage] = convolve(self.nflx_uv_raw[iage], kernUv, boundary='extend')
        
        for iage in range(self.Nage_op_raw):  
            self.llam_op_raw[iage] = convolve(self.llam_op_raw[iage], kernOp, boundary='extend')
            #self.nflx_op_raw[iage] = convolve(self.nflx_op_raw[iage], kernOp, boundary='extend')
        
        print ('Convolved the UV and optical spectra to instrumental resolution')
        print ('  UV R={:6.0f}, FWHM={:2.3f} A, Sampl={:2.3f} A/px, sigma={:2.3f} px'.format(self.pars.resPowUv, fwhm_aaUv, sampUv, sig_pxUv))
        print (' Opt R={:6.0f}, FWHM={:2.3f} A, Sampl={:2.3f} A/px, sigma={:2.3f} px'.format(self.pars.resPowOp, fwhm_aaOp, sampOp, sig_pxOp))




    def resample_age(self):

        # rage signifies rebinned in age
        self.llam_uv_rage = sp.zeros((self.pars.Nage, self.Nlam_uv_raw), dtype=np.float64)
        #self.nflx_uv_rage = sp.zeros((self.pars.Nage, self.Nlam_uv_raw), dtype=np.float64)
        self.llam_op_rage = sp.zeros((self.pars.Nage, self.Nlam_op_raw), dtype=np.float64)
        #self.nflx_op_rage = sp.zeros((self.pars.Nage, self.Nlam_op_raw), dtype=np.float64)
        
        for ilam in range(self.Nlam_uv_raw):
            #self.llam_uv_rage[:,ilam] = 10.**resample_linear(self.age_uv_raw, sp.log10(self.llam_uv_raw[:,ilam]), self.age_reb)
            #self.nflx_uv_rage[:,ilam] =      resample_linear(self.age_uv_raw,          self.nflx_uv_raw[:,ilam] , self.age_reb)
            self.llam_uv_rage[:,ilam] = 10.**resample_spline(self.age_uv_raw, sp.log10(self.llam_uv_raw[:,ilam]), self.age_reb, kind='quadratic')
            #self.nflx_uv_rage[:,ilam] =      resample_spline(self.age_uv_raw,          self.nflx_uv_raw[:,ilam] , self.age_reb, kind='quadratic')
        
        for ilam in range(self.Nlam_op_raw):
            #self.llam_op_rage[:,ilam] = 10.**resample_linear(self.age_op_raw, sp.log10(self.llam_op_raw[:,ilam]), self.age_reb)
            #self.nflx_op_rage[:,ilam] =      resample_linear(self.age_op_raw,          self.nflx_op_raw[:,ilam] , self.age_reb)
            self.llam_op_rage[:,ilam] = 10.**resample_spline(self.age_op_raw, sp.log10(self.llam_op_raw[:,ilam]), self.age_reb, kind='quadratic')
            #self.nflx_op_rage[:,ilam] =      resample_spline(self.age_op_raw,          self.nflx_op_raw[:,ilam] , self.age_reb, kind='quadratic')
        
        
        # concatenate the UV and optical spectra
        self.lamr_raw  = sp.append(self.lamr_uv_raw , self.lamr_op_raw)
        self.llam_rage = sp.append(self.llam_uv_rage, self.llam_op_rage, axis=1)
        #self.nflx_rage = sp.append(self.nflx_uv_rage, self.nflx_op_rage, axis=1)
        self.Nlam_raw  = self.Nlam_uv_raw + self.Nlam_op_raw
        
        
        self.lamr_min  = self.lamr_raw.min()
        self.lamr_max  = self.lamr_raw.max()
        
        # resample and integrate the ionizing photon rates
        self.qhi_reb   = 10.**resample_spline( self.age_qu_raw, sp.log10(self.qhi_raw  ), self.age_reb )
        self.qhei_reb  = 10.**resample_spline( self.age_qu_raw, sp.log10(self.qhei_raw ), self.age_reb )
        self.qheii_reb = 10.**resample_spline( self.age_qu_raw, sp.log10(self.qheii_raw), self.age_reb )
        
        # resample and integrate the mechanical luminosities 
        #self.lmechSn_reb   = 10.**resample_linear( self.age_em_raw, sp.log10(self.lmechSn_raw  ), self.age_reb )
        #self.lmechStel_reb = 10.**resample_linear( self.age_em_raw, sp.log10(self.lmechStel_raw), self.age_reb )
        self.lmechTot_reb  = 10.**resample_spline( self.age_em_raw, sp.log10(self.lmechTot_raw ), self.age_reb )
        self.emechTot_reb  = 10.**resample_spline( self.age_em_raw, sp.log10(self.emechTot_raw ), self.age_reb )

        #fhdm = open('delme_lmechraw_test.txt', 'w')
        #for idm in range(len(self.age_em_raw)): fhdm.write('{:1.6g}  {:1.6g}\n'.format(self.age_em_raw[idm], self.lmechTot_raw[idm]) )  
        #fhdm.close()
        #fhdm = open('delme_lmechreb_test.txt', 'w')
        #for idm in range(len(self.age_reb)): fhdm.write('{:1.6g}  {:1.6g}\n'.format(self.age_reb[idm], self.lmechTot_reb[idm]) )  
        #fhdm.close()

        #sys.exit()
        #self.emechSn_reb   = sp.zeros_like(self.lmechSn_reb  )
        #self.emechStel_reb = sp.zeros_like(self.lmechStel_reb)
        #self.emechTot_reb  = sp.zeros_like(self.lmechTot_reb )
        #
        #for iage in range(self.pars.Nage-1):
        #    self.emechSn_reb  [iage] = sp.trapz(self.lmechSn_reb  [:iage+1], x=self.age_reb[:iage+1]*secyr)
        #    self.emechStel_reb[iage] = sp.trapz(self.lmechStel_reb[:iage+1], x=self.age_reb[:iage+1]*secyr)
        #    self.emechTot_reb [iage] = sp.trapz(self.lmechTot_reb [:iage+1], x=self.age_reb[:iage+1]*secyr)
        #
        #self.emechSn_reb  [iage-1] = self.emechSn_reb  [iage-2]
        #self.emechStel_reb[iage-1] = self.emechStel_reb[iage-2]
        #self.emechTot_reb [iage-1] = self.emechTot_reb [iage-2]

        print ('Resampled the UV and optical spectra onto the standard age grid, and appended in lambda')
        print ('Resampled the Ionizing photon rates')
        print ('Resampled the Mechanical Luminosities and integrated them from t=0')
        print (' wavelength = self.lamr_raw,  luminosities = self.llam_rage,   nflux = self.nflx_rage, as below:')
        #print (self.lamr_raw, self.llam_rage, self.nflx_rage)




    def shift_to_z(self):

        self.lamo_raw  = self.lamr_raw * (1. + self.pars.redshift)
        #self.flam_raw  = self.llam_raw  / self.LoverF * sanscale
        self.flam_rage = self.llam_rage / self.pars.LoverF * sanscale
        
        self.lamo_min  = self.lamo_raw.min()
        self.lamo_max  = self.lamo_raw.max()




    def resample_wave(self, lnew):
    
        self.lamo_reb = lnew
        self.lamr_reb = lnew / (1.+self.pars.redshift)
        self.Nlam_reb = len(lnew)
        
        self.klam     = Cal(self.lamo_reb)
        
        #self.llam_reb = sp.zeros((self.pars.Nage, self.pars.Nebv, self.Nlam_reb), dtype=np.float64) # for the discrete reddening method
        self.llam_reb = sp.zeros((self.pars.Nage, self.Nlam_reb), dtype=np.float64) # for the continuous EBV method
        #self.nflx_reb = sp.zeros((self.pars.Nage, self.Nlam_reb), dtype=np.float64)
        
        for iage in range(self.pars.Nage):
            #self.nflx_reb[iage] = resample_bin(self.lamo_raw, self.nflx_rage[iage], self.lamo_reb)
            self.llam_reb[iage] = resample_bin(self.lamo_raw, self.llam_rage[iage], self.lamo_reb)

            # for the discrete ebv method
            #for iebv in range(self.pars.Nebv):
            #    self.llam_reb[iage,iebv] = rflux * 10.**(-0.4 * self.klam * self.ebv[iebv])
            
            #self.flam_reb[iage,iebv] = resample_bin(self.lamo_raw, self.flam_rage[iage], self.lamo_reb)
        
        self.flam_reb = self.llam_reb / self.pars.LoverF * sanscale
        
        print ('  Resampled synthetic spectra in wavelength for', self.pars.Nage, 'ages:', self.age_reb[0], 'to', self.age_reb[-1])




    def standard_convolve(self, Nconcat, iconcat, sampingConcat, meanlamConcat, vel=350.):

        print ('**** WARNING ****')
        print ('this standard covolution happens in place and overwrites the previously rebinned matrix')
    
        for iage in range(self.pars.Nage):
            #self.nflx_reb[iage] = convolve_concat(self.nflx_reb[iage], Nconcat, iconcat, sampingConcat, meanlamConcat, vel)
            self.flam_reb[iage] = convolve_concat(self.flam_reb[iage], Nconcat, iconcat, sampingConcat, meanlamConcat, vel)

            # for the discrete EBV method
            #for iebv in range(self.pars.Nebv):
            #    self.flam_reb[iage,iebv] = convolve_concat(self.flam_reb[iage,iebv], Nconcat, iconcat, sampingConcat, meanlamConcat, vel)




    def calc_secondprop(self, age, norm): 
    
        iage = ind_nearest(self.age_reb, age)
        
        qhi   = norm * self.qhi_reb     [iage]
        qhei  = norm * self.qhei_reb    [iage]
        qheii = norm * self.qheii_reb   [iage]
        lmech = norm * self.lmechTot_reb[iage]
        emech = norm * self.emechTot_reb[iage]
        
        return qhi, qhei, qheii, lmech, emech
