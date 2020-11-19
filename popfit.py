import sys
import scipy as sp
import numpy as np
from matplotlib.ticker import MultipleLocator
from matplotlib import pyplot as plt
import pickle
import lmfit
import corner

from extin import Cal
from mathfunc import ind_nearest, argmin_true
from convol import convolve_concat


collist = [ 'C1', 'C2', 'C3' ]


def emcee_chain_get_percentiles(mcobj, parname, percentiles): 

    mcvals = mcobj.flatchain[parname].values




def zoom_and_restore(fig, ax, axt, xlim, z, lampix, flampix, fn):

    #self.ax1t.set_xbound(sp.array (  ) / (1.+self.o.redshift) )
    
    xlimold   = ax.get_xbound()
    ylimold   = ax.get_ybound()
    xlimoldt  = axt.get_xbound()

    xlimnewt  = sp.array(xlim)
    xlimnew   = sp.array(xlim) * (1.+z)

    ishow     = (xlimnew[0] < lampix) & (lampix < xlimnew[1])
    if len(lampix[ishow])>0:
        ylimnew   = [ 0.5*flampix[ishow].min(), 1.2*flampix[ishow].max() ]

        ax.set_xlim( xlimnew )
        ax.set_ylim( ylimnew )
        axt.set_xbound( xlimnewt )
        fig.savefig(fn)

        ax.set_xlim( xlimold )
        ax.set_ylim( ylimold )
        axt.set_xbound( xlimoldt )




def plot_walkers(fitres, fnOut):
      
    Npar  = fitres.chain.shape[-1]
    #Nwalk = fitres.chain.shape[0]
    Nwalk = fitres.chain.shape[1] #MS correcting the walkers plot
    fig   = plt.figure(figsize=(10,16))
    axs   = []
    #print(fitres.chain.shape)

    for ipar in range(Npar):
        axs.append ( fig.add_subplot(Npar,1,ipar+1) )
        axs[-1].set_ylabel(fitres.var_names[ipar])
        
        for iwalker in range(Nwalk): 
            #chain = fitres.chain[iwalker,:,ipar]
            chain = fitres.chain[:,iwalker,ipar] #MS correcting the walkers plot
            plt.plot(chain)
            plt.text(len(chain), chain[-1], '%d'%iwalker)

    axs[-1].set_xlabel('N step (burn removed)')
    fig.savefig(fnOut)



def anal_norm_1pop(mod, dat, sig): 
  
    n = sp.sum( dat*mod/sig**2 ) / sp.sum( (mod/sig)**2. )
    if n<0.: n=0.
    x2 = sp.sum( ((n*mod - dat) / sig )**2 )

    return n, x2    




def anal_norm_2pop(moda, modb, dat, sig):

    s1 = sp.sum( ( moda / sig )**2. )
    s2 = sp.sum( moda * dat / sig**2. )
    s3 = sp.sum( moda * modb / sig**2. )
    s4 = sp.sum( ( modb / sig )**2. )
    s5 = sp.sum( modb * dat / sig**2. )

    na = (s2 - s3*s5/s4) / (1. - s3**2./s4)
    nb = (s5 - na*s3) / s4
    if na<0.: na=0.
    if nb<0.: nb=0.

    x2 = sp.sum( ((na*moda + nb*modb - dat) / sig )**2 )
    
    return na, nb, x2



# a class that holds all the properties of a stellar population
class StelPop:
    #MS changing nmax: 1e5 -> 1e1 so that the mass (M=10**6 * n) does not exceed 10**7 Msun
    def __init__(self, name, aval=7.0  , afit=True, amin=6.0  , amax=9.0  , \
                             nval=0.01  , nfit=True, nmin=0.0  , nmax=1.e1 , \
                             zval=0.008, zfit=True, zmin=0.001, zmax=0.040):
          
        self.name  = name     # the name

        self.aval  = aval     # the age of population
        self.afit  = afit     # if fit == True, the age is the guess, else it is the constraint
        self.amin  = amin     # min and max ages.  Only relevant if fit==True
        self.amax  = amax

        self.nval  = nval     # normalziation of population 
        self.nfit  = nfit     # 
        self.nmin  = nmin     # 
        self.nmax  = nmax     # 

        self.zval  = zval     # metallicity
        self.zfit  = zfit     # 
        self.zmin  = zmin     # 
        self.zmax  = zmax     # 



 
class Fit:
    
    def __init__(self, p, o, s, mask):
          
        self.o     = o
        self.s     = s
        self.p     = p
        self.mask  = mask
        self.imask = self.mask == 1
        self.Ndp   = int(self.mask.sum())
        
        self.klam  = Cal(self.o.lamr_reb)
  
  


    def plot_obs(self, lines2plot, xminUv=1130., xmaxUv=1772., yminUv=0., ymaxUv=20., 
                 xminOp=3500., xmaxOp=8000., yminOp=0., ymaxOp=1.5):
  
        self.xminUv = 1000.
        self.xmaxUv = 1800.
        self.yminUv = yminUv
        self.ymaxUv = ymaxUv
        self.xminOp = 4500. #MS  
        self.xmaxOp = 8000. #MS
        self.yminOp = yminOp
        self.ymaxOp = ymaxOp
  
        xlab = 'Observed Wavelength [ $\mathrm{\AA}$ ]'
        ylab = 'Observed f$_\lambda$ [ $10^{-15}$ c.g.s./$\mathrm{\AA}$ ]'
        
        # UV focused plot 
        self.fig1, self.ax1 = plt.subplots(1,1, figsize=(9,4))#, tight_layout=True)
        self.ax1.axis([self.xminUv, self.xmaxUv, self.yminUv, self.ymaxUv])
        #MS changing style: plotting mask as grey vertical bands instead of different color spectrum
        self.ax1.fill_between(self.o.lamo_reb, self.yminUv, self.ymaxUv, self.imask==0, alpha=0.5, color='lightgrey')
        #self.ax1.plot(self.o.lamo_reb, self.o.flam_reb, drawstyle='steps', ls='-', c='lightblue')
        self.ax1.plot(self.o.lamo_reb, self.o.flam_reb , drawstyle='steps', ls='-', c='black')
        #MS not interested in plotting these lines now
        #self.ax1.vlines(lines2plot.lamr, ymin=yminUv, ymax=ymaxUv, colors='b', linestyle='dotted')
        #self.ax1.vlines(lines2plot.lamo, ymin=yminUv, ymax=ymaxUv, colors='r', linestyle='dotted')
        self.ax1.set_xlabel(xlab)
        self.ax1.set_ylabel(ylab)
        
        # optical focused plot 
        #MS same changes as above
        self.fig2, self.ax2 = plt.subplots(1,1, figsize=(9,4))#, tight_layout=True)
        self.ax2.axis([self.xminOp, self.xmaxOp, self.yminOp, self.ymaxOp])

        self.ax2.fill_between(self.o.lamo_reb, self.yminOp, self.ymaxOp, self.imask==0, alpha=0.5, color='lightgrey')        
        #self.ax2.plot(self.o.lamo_reb, self.o.flam_reb , ls='-', c='lightblue')
        self.ax2.plot(self.o.lamo_reb, self.o.flam_reb , drawstyle='steps', ls='-', c='black')
        #self.ax2.plot(self.o.lamo_reb[self.imask], self.o.flam_reb[self.imask] , ls='-', c='darkslateblue')
        #self.ax2.vlines(lines2plot.lamr, ymin=yminOp, ymax=ymaxOp, colors='b', linestyle='dotted')
        #self.ax2.vlines(lines2plot.lamo, ymin=yminOp, ymax=ymaxOp, colors='r', linestyle='dotted')
        self.ax2.set_xlabel(xlab)
        self.ax2.set_ylabel(ylab)
  
  
  
  
    def plot_fit_spec_1pop(self):
         
        #### !!!!! check that lam_reb in the blow line is in the rest frame!!
        #fesc = 10.** (-0.4 * Cal(self.o.lam_rest) * self.ebv[self.iebv_best])
        flam_plot = self.norm_best * self.s.flam_reb[self.iage_best,self.iebv_best] #* fesc
        self.ax1.plot(self.o.lam_reb, flam_plot , ls='steps-', c='k')
  
  
  
  
    def fit_spec_1pop(self, d, e, mod, k):
          
        norm = sp.zeros((self.Nebv, self.s.Nage_reb), dtype=np.float64)
        chi2 = sp.zeros((self.Nebv, self.s.Nage_reb), dtype=np.float64)
            
        for iebv in range(self.Nebv):
            fesc = 10.**(-0.4*self.ebv[iebv]*k)        
            for iage in range(self.s.Nage_reb):
                m               = mod[iage] * fesc
                norm[iebv,iage] = sp.sum(d*m/e**2) / sp.sum((m/e)**2)
                chi2[iebv,iage] = sp.sum(((d - norm[iebv,iage]*m)/e)**2)
          
        iebvb, iageb = np.unravel_index(chi2.argmin(), chi2.shape)
        chi2b  = chi2[iebvb,iageb]
        normb  = norm[iebvb,iageb]
        
        return iageb, iebvb, normb, chi2b
  
  
  
  
    def run_fit_spec_1pop(self, plotChi2=False, Nmc=0):
          
        lr   = self.s.lamr_reb [self.imask]
        lo   = self.s.lamo_reb [self.imask]
        d    = self.o.flam_reb [self.imask]
        e    = self.o.dflam_reb[self.imask]
        m    = self.s.flam_reb [:,:,self.imask] 
        Ndp  = self.mask.sum()
        klam = Cal(lr)
        
        self.iage_best, self.iebv_best, self.norm_best, self.chi2_best = self.fit_spec_1pop(d, e, m, klam)
        self.age_best = self.s.age_reb[self.iage_best]
        self.ebv_best = self.ebv[self.iebv_best]
        self.chi2n_best = self.chi2_best / (Ndp - 2 - 1)
        
        # if there is Monte Carlo simultion 
        if Nmc:
            print ('Runing a Monte Carlo simulation with', Nmc, 'realizations')
            mc_ebv  = sp.zeros(Nmc, dtype=sp.float64)
            mc_age  = sp.zeros(Nmc, dtype=sp.float64)
            mc_norm = sp.zeros(Nmc, dtype=sp.float64)
            mc_chi2 = sp.zeros(Nmc, dtype=sp.float64)
            
            for imc in range(Nmc):
                mc_d = d + sp.randn(Ndp)*e
                mc_iage_best, mc_iebv_best, mc_norm_best, mc_chi2_best = self.fit_spec_1pop(mc_d, e, m, klam)
                
                mc_age [imc] = self.s.age_reb[mc_iage_best]
                mc_ebv [imc] = self.ebv[mc_iebv_best]
                mc_norm[imc] = mc_norm_best
                mc_chi2[imc] = mc_chi2_best
              
            self.dage_lo, self.dage_hi   = sp.percentile(mc_age , [14,86])
            self.debv_lo, self.debv_hi   = sp.percentile(mc_ebv , [14,86])
            self.dnorm_lo, self.dnorm_hi = sp.percentile(mc_norm, [14,86]) 
              
        else:
            print ('No Monte Carlo -- errors are -1')
            self.dage_lo, self.dage_hi   = -1.,-1.
            self.debv_lo, self.debv_hi   = -1.,-1.
            self.dnorm_lo, self.dnorm_hi = -1.,-1.
          
        print ('       best age : %6.4g | %6.4g - %6.4g' % (self.age_best , self.dage_lo , self.dage_hi ))
        print ('     best E(B-V): %6.4g | %6.4g - %6.4g' % (self.ebv_best , self.debv_lo , self.debv_hi ))
        print ('  normalization : %6.4g | %6.4g - %6.4g' % (self.norm_best, self.dnorm_lo, self.dnorm_hi))
        print ('           chi2 : %6.4g' % self.chi2_best)
        print ('         chi2nu : %6.4g' % self.chi2n_best)
        print ('')
        
        if plotChi2: 
            fig2, ax2 = plt.subplots(1,1, figsize=(7,7))
            cb2 = ax2.imshow(chi2, origin='lower')#, vmin=chi2.min(), vmax=chi2.min()*20)
            ax2.plot(self.iage_best, self.iebv_best, 'kx')
            fig2.colorbar(cb2)
  
  
  
  
    def plot_fit_windows(self):
        
        for norm in self.norm_best:
            flam_plot = norm * self.s.flam_reb[self.iage_best,self.iebv_best]
            self.ax1.plot(self.o.lam_reb, flam_plot , ls='steps-', c='k')
            
        for iwind in range(len(self.norm_best)):
            self.ax1.add_patch ( patches.Rectangle( \
                (stelWin.lamolo[iwind], -1.), (2*stelWin.dlamo[iwind]), 6, \
                  alpha=0.5, fc='chartreuse') )
  
  
  
  
    def plot_norms(self):
        ymin, ymax = 0., 40
        
        self.fig3, self.ax3 = plt.subplots(1,1, figsize=(10,5))#, tight_layout=True)
        self.ax3.axis([1000,1600,ymin,ymax])
        
        self.ax3.plot(stelWin.lamr, self.norm_best, 'rs', ms=20)
        
        self.ax3.set_xlabel('Wavelength [ AA ]')
        self.ax3.set_ylabel('Normalization')
  
  
  
  
    def fit_windows(self, plotChi2=False):
          
        print ('there are', stelWin.Nline, 'windows with centres', stelWin.lamo)
  
        maskFeat = []
        
        for iwind in range(stelWin.Nline):
            #print (stelWin.lamrlo[iwind], stelWin.lamrhi[iwind])
            indwind = (stelWin.lamolo[iwind] < self.o.lam_reb) & (self.o.lam_reb < stelWin.lamohi[iwind])
            maskFeat.append ( sp.where( indwind, 1, 0 ) )
            #print (self.o.lam_reb[indwind].min(), self.o.lam_reb[indwind].max())
        
        Ndp = 0
        for iwind in range(stelWin.Nline):
            Ndp += maskFeat[iwind].sum()
            #print (len(maskFeat[iwind]), maskFeat[iwind].sum())
        
        norm = sp.zeros((synSspZ008.Nage_reb, stelWin.Nline), dtype=np.float64)
        chi2 = sp.zeros(synSspZ008.Nage_reb, dtype=np.float64)
        
        for iage in range(synSspZ008.Nage_reb):
            for iwind in range(stelWin.Nline):
                im = (self.mask * maskFeat[iwind]) == 1
                d  = self.o.flam_reb   [im]
                e  = self.o.dflam_reb  [im]
                m  = self.s.flam_reb[iage,0][im]
                norm[iage][iwind] = sp.sum(d*m/e**2) / sp.sum((m/e)**2)
                chi2[iage] += sp.sum(((d - norm[iage,iwind]*m)/e)**2)
        
        self.iage_best  = chi2.argmin()
        self.chi2_best  = chi2[self.iage_best]
        self.norm_best  = norm[self.iage_best]    
        self.chi2n_best = self.chi2_best / (Ndp - 1 - len(maskFeat) - 1)
        
        print ('       best age : %6.4g' % self.s.age_reb[self.iage_best])
        print ('  normalization :', self.norm_best)
        print ('           chi2 : %6.4g' % self.chi2_best)
        print ('         chi2nu : %6.4g' % self.chi2n_best)
        print ('')
        
        if plotChi2: 
            fig2, ax2 = plt.subplots(1,1, figsize=(7,7))
            cb2 = ax2.imshow(chi2, origin='lower')#, vmin=chi2.min(), vmax=chi2.min()*20)
            ax2.plot(self.iage_best, self.iebv_best, 'kx')
            fig2.colorbar(cb2)
      
      
  
  
    def setup_multipop_red(self, stellages):       
          
        self.Npop       = len(stellages)
        self.istellages = sp.array([ ind_nearest(self.s.age_reb, a) for a in stellages ])
        self.stell_comp = sp.zeros((self.Npop, len(self.s.flam_reb[0,0])), dtype=sp.float64)
        for ipop in range(self.Npop): 
            self.stell_comp[ipop] = self.s.flam_reb[ self.istellages[ipop] ]
  
  
  
  
    def make_multipop_red(self, ll, e, n1, n2, n3):
        fesc = 10.**(-0.4 * Cal(ll) * e)
        stell_tot = (n1*self.stell_comp[0] + 
                     n2*self.stell_comp[1] + 
                     n3*self.stell_comp[2]) * fesc
        return stell_tot
  
  
  
  
    def fit_multipop_red(self):
  
        #assign infinite errors to the masked regions
        weightloc = sp.where(self.mask == 1, self.o.dflam_reb, 1.e30)
        Ndp       = int(self.mask.sum())
        
        popt, pcov = curve_fit(self.make_multipop_red, self.o.lam_reb, self.o.flam_reb, \
                               p0=[0.1, 1., 1.,1.], sigma=weightloc, \
                               bounds=(0, [1., 100000., 100000., 100000.]))
                     
        perr = sp.sqrt(sp.diag(pcov))
        
        self.ebv_best = popt[0]
        self.debv_lo, self.debv_hi = self.ebv_best - perr[0], self.ebv_best + perr[0] 
        
        self.norm_best  = popt[1:]
        self.dnorm_lo   = popt[1:] - perr[1:]
        self.dnorm_hi   = popt[1:] + perr[1:] 
        self.flam_best  = self.make_multipop_red(self.o.lam_reb, *popt)
        self.chi2_best  = sp.sum( (self.o.flam_reb-self.flam_best)**2./weightloc**2. )
        self.chi2n_best = self.chi2_best / (Ndp - 4 - 1)
        
        print ('  E(B-V) : %9.5g | %9.5g - %9.5g' % (self.ebv_best, self.debv_lo, self.debv_hi))
        print ('  normalizations for ages:')
        for ipop in range(self.Npop):
            print ('   %9.5g yr : %9.5g  |  %9.5g +- %9.5g' % \
                   (self.s.age_reb[ self.istellages[ipop]], self.norm_best[ipop], self.dnorm_lo[ipop], self.dnorm_hi[ipop]))
        print ('    chi2 : %9.5g' % self.chi2_best)
        print ('   chi2n : %9.5g for %d datapoints' % (self.chi2n_best, Ndp))
  
  
  
  
    def plot_fit_multipop_red(self):
          
        fesc = 10.**(-0.4 * Cal(self.o.lam_rest) * self.ebv_best)
        for ipop in range(self.Npop):
            flam_plot = self.norm_best[ipop] * self.stell_comp[ipop] * fesc
            label_str = 'pop{:2d}: a={:1.3f} logyr'.format(ipop+1, self.s.age_reb[self.istellages[ipop]] )
            self.ax1.plot(self.o.lam_reb, flam_plot , ls='steps-', c='orange', label=label_str)
        self.ax1.plot(self.o.lam_reb, self.flam_best, ls='steps-', c='k', label='mod.')
        self.ax1.legend(loc=2)
  
  
  
  
    def fit_multipop_minim(self, pars):
  
        ebv  = pars['ebv']
        n1   = pars['n1' ]
        n2   = pars['n2' ]
        n3   = pars['n3' ]
        fesc = 10.**(-0.4 * Cal(self.s.lamr_reb) * ebv)
        st   = self.make_multipop_red(self.s.lamr_reb, ebv, n1, n2, n3)
        mq   = (st[self.imask] - self.o.flam_reb[self.imask]) / self.o.dflam_reb[self.imask]
        return mq
  
  
  
  
    def fit_multipop_red_lmfit(self):
        lmpars = lmfit.Parameters()
        lmpars.add('ebv', vary=True, value=0.2, min=0, max=1   )
        lmpars.add('n1' , vary=True, value=1.e+1, min=0, max=1.e4)
        lmpars.add('n2' , vary=True, value=1.e+1, min=0, max=1.e4)
        lmpars.add('n3' , vary=True, value=1.e+1, min=0, max=1.e4)
        
        
        # set up the object with the method+return quantity and the parameters to minimize. 
        self.minner = lmfit.Minimizer(self.fit_multipop_minim, lmpars)
        
        # now minimize with the standard "leastsq": Levenberg-Marquardt (default)
        
        # first find the minimum with the LM method, then sample the posterior with emcee
        # are the results convergent? 
        self.fitres_mi = self.minner.minimize()
        self.fitres_mc = self.minner.emcee()
  
  
    
  
    def disp_multipop_red_lmfit(self):
          
        lmfit.report_fit(self.fitres_mi)
        lmfit.report_fit(self.fitres_mc)
        
        vd   = self.fit_result.params.valuesdict()
        perr = sp.sqrt(sp.diag(result.covar))
        self.ebv_best  = vd['ebv']
        self.norm_best = [ vd['n1'], vd['n2'], vd['n3'] ]
        
        self.debv_lo , self.debv_hi  = self.ebv_best  - perr[0] , self.ebv_best  + perr[0] 
        self.dnorm_lo, self.dnorm_hi = self.norm_best - perr[1:], self.norm_best + perr[1:] 
         
        self.flam_best  = self.make_multipop_red(self.s.lamr_reb, self.ebv_best, *self.norm_best)
        self.chi2_best  = sp.sum( (self.o.flam_reb[self.imask]-self.flam_best[self.imask])**2./self.o.dflam_reb[self.imask]**2. )
        self.chi2n_best = self.chi2_best / (self.Ndp - 4 - 1)
        
        print ('  E(B-V) : %9.5g | %9.5g - %9.5g' % (self.ebv_best, self.debv_lo, self.debv_hi))
        print ('  normalizations for ages:')
        for ipop in range(self.Npop):
            print ('   %9.5g yr : %9.5g  |  %9.5g +- %9.5g' % \
                   (self.s.age_reb[ self.istellages[ipop]], self.norm_best[ipop], self.dnorm_lo[ipop], self.dnorm_hi[ipop]))
        print ('    chi2 : %9.5g' % self.chi2_best)
        print ('   chi2n : %9.5g for %d datapoints' % (self.chi2n_best, self.Ndp))
        
        #self.emcee = self.minner.emcee()
  
  
  
  
    #############################################################################
    ### make an initial guess for the normalizaions of the population(s)
    ### analytical solutions for the normalization.  Follows standard methods if
    ### Npop = 1 or Npop = 2.  If Npop > 2 it follows the Npop = 2 method and
    ### writes zero normalization to the other populations. 
    #############################################################################
    def guessnorm_npop_red(self, pars):
          
        parvals = pars.valuesdict()
        
        #MS using a different extinction for each population
        #ebvval  = parvals['ebv']
        #ebvmax  = pars['ebv'].max
        #fescval = 10.**(-0.4 * self.klam * ebvval)
        #fescmax = 10.**(-0.4 * self.klam * ebvmax)
  
        popkeys = [ a.replace('_a', '') for a in parvals.keys() if a.endswith('_a') ]
  
        #MS using a different extinction for each population
        ebvval = 0.
        ebvmax = 0.
        for ipop in range(self.Npop):
            ebvval += parvals[popkeys[ipop]+'_ebv']
            ebvmax += pars[popkeys[0]+'_ebv'].max

        #print(ebvval, ebvmax)
        fescval = 10.**(-0.4 * self.klam * ebvval)
        fescmax = 10.**(-0.4 * self.klam * ebvmax)
  
        ageval_init   = parvals[popkeys[0]+'_a']
        agemax_init   = pars[popkeys[0]+'_a'].max
 
        normval_init  = parvals[popkeys[0]+'_n']
        normmax_init  = pars[popkeys[0]+'_n'].max 
 
        metval        = parvals[popkeys[0]+'_z']

        #syn.flam_spl[imet](sp.log10(lam_test), sp.log10(age_test)) 
        modspecval = 10.**self.s.gen_flam_from_splines(metval, ageval_init, self.o.lamo_reb_log) * fescval
        modspecmax = 10.**self.s.gen_flam_from_splines(metval, agemax_init, self.o.lamo_reb_log) * fescmax
        
        #normval_anal, chi2val_anal = anal_norm_1pop(modspecval[self.imask], self.o.flam_reb[self.imask], self.o.dflam_reb[self.imask])
        #normmax_anal, chi2max_anal = anal_norm_1pop(modspecmax[self.imask], self.o.flam_reb[self.imask], self.o.dflam_reb[self.imask])

        normval_anal = anal_norm_1pop(modspecval[self.imask], self.o.flam_reb[self.imask], self.o.dflam_reb[self.imask])[0]
        normmax_anal = anal_norm_1pop(modspecmax[self.imask], self.o.flam_reb[self.imask], self.o.dflam_reb[self.imask])[0] * 10.


        for ipop in range(self.Npop):
        #MS if the guessed values of n exceed nmax (10) I discard them and set them to nmax  
            if normval_anal < pars[popkeys[ipop]+'_n'].max:
                pars[popkeys[ipop]+'_n'].value = normval_anal
            if normmax_anal < pars[popkeys[ipop]+'_n'].max:
                pars[popkeys[ipop]+'_n'].max   = normmax_anal
            print (popkeys[ipop], 'value updated with normalization of', pars[popkeys[ipop]+'_n'].value, 'max value', pars[popkeys[ipop]+'_n'].max)
 
        #MS no need to plot this
        #plt.figure() 
        #plt.plot( self.o.lamo_reb,             self.o.flam_reb             )
        #plt.plot( self.o.lamo_reb[self.imask], self.o.flam_reb[self.imask] )
        #plt.plot( self.o.lamo_reb,             normval_anal*modspecval     )
      
        # the following was expected to work but fails.. find out why! 
        """
        if (self.Npop == 1): 
  
            ageval_init   = parvals[popkeys[0]+'_a']
            agemax_init   = pars[popkeys[0]+'_a'].max
  
            normval_init  = parvals[popkeys[0]+'_n']
            normmax_init  = pars[popkeys[0]+'_n'].max
  
            iageval_init  = ind_nearest(self.s.age_reb, ageval_init)
            iagemax_init  = ind_nearest(self.s.age_reb, agemax_init)
  
            modspecval = self.s.flam_reb[iageval_init] * fescval
            modspecmax = self.s.flam_reb[iagemax_init] * fescmax
            
            normval_anal, chi2val_anal = anal_norm_1pop(modspecval[self.imask], self.o.flam_reb[self.imask], self.o.dflam_reb[self.imask])
            normmax_anal, chi2max_anal = anal_norm_1pop(modspecmax[self.imask], self.o.flam_reb[self.imask], self.o.dflam_reb[self.imask])
  
            pars[popkeys[0]+'_n'].value = normval_anal      
            pars[popkeys[0]+'_n'].max   = normmax_anal      
            print (popkeys[0], 'value updated with normalization of', normval_anal, 'max value', normmax_anal)
  
        elif (self.Npop >= 2): 
  
            age1val_init   = parvals[popkeys[0]+'_a']
            age2val_init   = parvals[popkeys[1]+'_a']
            age1max_init   = pars[popkeys[0]+'_a'].max
            age2max_init   = pars[popkeys[1]+'_a'].max
  
            norm1val_init  = parvals[popkeys[0]+'_n']
            norm2val_init  = parvals[popkeys[1]+'_n']
            norm1max_init  = pars[popkeys[0]+'_n'].max
            norm2max_init  = pars[popkeys[1]+'_n'].max
  
            iage1val_init  = ind_nearest(self.s.age_reb, age1val_init)
            iage2val_init  = ind_nearest(self.s.age_reb, age2val_init)
            iage1max_init  = ind_nearest(self.s.age_reb, age1max_init)
            iage2max_init  = ind_nearest(self.s.age_reb, age2max_init)
  
            modspec1val = self.s.flam_reb[iage1val_init] * fescval
            modspec2val = self.s.flam_reb[iage2val_init] * fescval
            modspec1max = self.s.flam_reb[iage1max_init] * fescmax
            modspec2max = self.s.flam_reb[iage2max_init] * fescmax
  
            norm1val_anal, norm2val_anal, chi2val_anal = anal_norm_2pop(modspec1val[self.imask], modspec2val[self.imask], self.o.flam_reb[self.imask], self.o.dflam_reb[self.imask])
            norm1max_anal, norm2max_anal, chi2max_anal = anal_norm_2pop(modspec1max[self.imask], modspec2max[self.imask], self.o.flam_reb[self.imask], self.o.dflam_reb[self.imask])
  
            pars[popkeys[0]+'_n'].value = norm1val_anal      
            pars[popkeys[1]+'_n'].value = norm2val_anal      
            pars[popkeys[0]+'_n'].max   = norm1max_anal      
            pars[popkeys[1]+'_n'].max   = norm2max_anal      
            print (popkeys[0], 'value updated with normalization of', norm1val_anal, 'max value', norm1max_anal)
            print (popkeys[1], 'value updated with normalization of', norm2val_anal, 'max value', norm2max_anal)
  
        # remove remainder of method    
        plt.figure()
        plt.plot(  self.o.lam_reb, self.o.flam_reb     , 'k-')
        plt.plot(  self.o.lam_reb, self.o.dflam_reb    , 'k--')
        if (self.Npop == 1):
            plt.plot(  self.o.lam_reb, normval_anal*modspecval, 'g-')
        elif (self.Npop >= 2):
            plt.plot(  self.o.lam_reb, norm1val_anal*modspec1val, 'g--')
            plt.plot(  self.o.lam_reb, norm2val_anal*modspec2val, 'g:')
            plt.plot(  self.o.lam_reb, norm1val_anal*modspec1val + norm2val_anal*modspec2val, 'g-')
        plt.axis([1000,8000,0,2.])
        """
  
  
  
  
    #############################################################################
    ### compute a synthetic spectrum from values in pars 
    #############################################################################
    def speccalc_1pop_red(self, pars, obsspec, convspec):
  
        parvals = pars.valuesdict()
  
        ebv     = parvals['ebv']
        fesc    = 10.** (-0.4 * self.klam * ebv)
  
        norm    = parvals['norm']
  
        specpop = norm * convspec * fesc
        
        mq = (specpop[self.imask] - obsspec[self.imask]) / self.o.dflam_reb[self.imask]
        return mq#**2.
 

 
    def setup_1pop_red(self, ebvval=0.2, ebvmax=1.):
  
        self.lmpars = lmfit.Parameters()
        self.lmpars.add('ebv', vary=True, value=ebvval, min=0, max=ebvmax)
        self.lmpars.add('norm', vary=True, value=1000., min=0., max=10000.)
  
    def guessnorm_1pop_red(self, pars):
          
        parvals = pars.valuesdict()
        
        ebvval  = parvals['ebv']
        ebvmax  = pars['ebv'].max
  
        fescval = 10.** (-0.4 * self.klam * ebvval)
        fescmax = 10.** (-0.4 * self.klam * ebvmax)
  
        normval_init  = parvals['norm']
        normmax_init  = pars['norm'].max
  
        iageval_init  = ind_nearest(self.s.age_reb, sp.log10(3.e6) )
        iagemax_init  = self.p.Nage - 1
  
        modspecval = self.s.flam_reb[iageval_init,0] * fescval
        modspecmax = self.s.flam_reb[iagemax_init,0] * fescmax
          
        normval_anal, chi2val_anal = anal_norm_1pop( \
            modspecval[self.imask], self.o.flam_reb[self.imask], self.o.dflam_reb[self.imask])
        normmax_anal, chi2max_anal = anal_norm_1pop( \
            modspecmax[self.imask], self.o.flam_reb[self.imask], self.o.dflam_reb[self.imask])
  
        pars['norm'].value = normval_anal
        pars['norm'].max   = normmax_anal      
        print ('normalization value updated with normalization of', normval_anal, 'max value', normmax_anal)
  
        """
        plt.figure()
        plt.plot(  self.o.lam_reb, self.o.flam_reb     , 'k-')
        plt.plot(  self.o.lam_reb, self.o.dflam_reb    , 'k--')
        if (self.Npop == 1):
            plt.plot(  self.o.lam_reb, normval_anal*modspecval, 'g-')
        elif (self.Npop >= 2):
            plt.plot(  self.o.lam_reb, norm1val_anal*modspec1val, 'g--')
            plt.plot(  self.o.lam_reb, norm2val_anal*modspec2val, 'g:')
            plt.plot(  self.o.lam_reb, norm1val_anal*modspec1val + norm2val_anal*modspec2val, 'g-')
        plt.axis([1000,8000,0,2.])
        """
  


    def fit_1pop_red(self, obsspec):
  
        Ndof = (self.Ndp-4-1)
        
        #mat_fwhm    = sp.zeros_like(( self.p.Nfwhm, self.p.Nage ), dtype=sp.float64)
        #mat_age     = sp.zeros_like(( self.p.Nfwhm, self.p.Nage ), dtype=sp.float64)
        mat_norm    = sp.zeros(( self.p.Nfwhm, self.p.Nage, self.p.Nebv ), dtype=sp.float64)
        mat_chi2    = sp.ones (( self.p.Nfwhm, self.p.Nage, self.p.Nebv ), dtype=sp.float64) * 1.e40
  
        for iFwhm in range(self.p.Nfwhm): 
            for iAge in range(self.p.Nage):
                for iEbv in range(self.p.Nebv):
  
                    modspec = self.s.flam_reb[iAge,iEbv]
  
                    # compute HERE THE direct minimization!
                    #mat_norm[iFwhm,iAge,iEbv] = self.fitres.params['norm'].value
                    #mat_chi2[iFwhm,iAge,iEbv] = self.fitres.chisqr 
                    mat_norm[iFwhm,iAge,iEbv] = sp.sum(modspec[self.imask] * obsspec[self.imask] / self.o.dflam_reb[self.imask]**2) / \
                                                sp.sum(( modspec[self.imask] / self.o.dflam_reb[self.imask] )**2)
                    mat_chi2[iFwhm,iAge,iEbv] = sp.sum( ( ( mat_norm[iFwhm,iAge,iEbv] * modspec[self.imask] - obsspec[self.imask]) / \
                                                       self.o.dflam_reb[self.imask] )**2 )
  
        self.ibest_fwhm, self.ibest_age, self.ibest_ebv = argmin_true(mat_chi2)
        self.best_norm  = mat_norm[self.ibest_fwhm, self.ibest_age, self.ibest_ebv]
        self.best_chi2n = mat_chi2[self.ibest_fwhm, self.ibest_age, self.ibest_ebv]/Ndof
  
  
  
  
    def plot_fit_12pop_red(self, npop=1, fnOut='delme'):
         
        fontsize = 12 
  
        if npop == 1:
   
            spec = self.result_norm * self.s.flam_reb[self.result_iage,self.result_iebv] 
  
            self.ax1.plot(self.o.lam_reb, spec, ls='steps-', c=collist[0], label=r'%6.2f logYr. %6.1f M$_\odot$' % (self.result_age, self.result_norm)) 
            self.ax2.plot(self.o.lam_reb, spec, ls='-'     , c=collist[0], label=r'%6.2f logYr. %6.1f M$_\odot$' % (self.result_age, self.result_norm)) 
  
        elif npop == 2:
   
            spec1 = self.result_norm1 * self.s.flam_reb[self.result_iage1,self.result_iebv] 
            spec2 = self.result_norm2 * self.s.flam_reb[self.result_iage2,self.result_iebv] 
            spect = spec1 + spec2 
  
            self.ax1.plot(self.o.lam_reb, spec1, ls='steps-', c=collist[0], label=r'%6.2f logYr. %6.1f M$_\odot$' % (self.result_age1, self.result_norm1)) 
            self.ax2.plot(self.o.lam_reb, spec1, ls='-'     , c=collist[0], label=r'%6.2f logYr. %6.1f M$_\odot$' % (self.result_age1, self.result_norm1)) 
            self.ax1.plot(self.o.lam_reb, spec2, ls='steps-', c=collist[1], label=r'%6.2f logYr. %6.1f M$_\odot$' % (self.result_age2, self.result_norm2)) 
            self.ax2.plot(self.o.lam_reb, spec2, ls='-'     , c=collist[1], label=r'%6.2f logYr. %6.1f M$_\odot$' % (self.result_age2, self.result_norm2)) 
            self.ax1.plot(self.o.lam_reb, spect, ls='steps-', c='k'       , label=r'total.') 
            self.ax2.plot(self.o.lam_reb, spect, ls='-'     , c='k'       , label=r'total.') 
  
  
        self.ax1.text(0.40, 0.92, r'E$_\mathrm{B-V}=$%5.3f' % (self.result_ebv), verticalalignment='center', transform=self.ax1.transAxes, fontsize=fontsize)
        self.ax1.text(0.40, 0.85, r'$\chi^2_\mathrm{d.o.f.} = $%7.3f' % (self.result_chi2n), verticalalignment='center', transform=self.ax1.transAxes, fontsize=fontsize)
  
        self.ax2.text(0.7, 0.67, r'E$_\mathrm{B-V}=$%5.3f' % (self.result_ebv), verticalalignment='center', transform=self.ax2.transAxes, fontsize=fontsize)
        self.ax2.text(0.7, 0.59, r'$\chi^2_\mathrm{d.o.f.} = $%7.3f' % (self.result_chi2n), verticalalignment='center', transform=self.ax2.transAxes, fontsize=fontsize)
  
        #self.ax1.plot(self.o.lam_reb, self.flam_best, ls='steps-', c='k', label='Total Model')
        #self.ax2.plot(self.o.lam_reb, self.flam_best, ls='-', c='k', label='Total Model')
        self.ax1.legend(loc=2)
        self.ax2.legend(loc=1)
  
        self.ax1t = self.ax1.twiny()
        self.ax2t = self.ax2.twiny()
        self.ax1t.set_xbound(sp.array ( self.ax1.get_xbound() ) / (1.+self.p.redshift) )  
        self.ax2t.set_xbound(sp.array ( self.ax2.get_xbound() ) / (1.+self.p.redshift) )  
  
        self.ax1.xaxis.set_minor_locator(MultipleLocator(100))
        self.ax2.xaxis.set_minor_locator(MultipleLocator(100))
        self.ax1t.xaxis.set_minor_locator(MultipleLocator(100))
        self.ax2t.xaxis.set_minor_locator(MultipleLocator(100))
  
        self.ax1t.set_xlabel('Restframe Wavelength [ $\mathrm{\AA}$ ]')
        self.ax2t.set_xlabel('Restframe Wavelength [ $\mathrm{\AA}$ ]')
  
        self.fig1.savefig(fnOut+'_uv.pdf')
        self.fig2.savefig(fnOut+'_op.pdf')
  
        # plot some zoom-ins around the P Cyg lines in the UV, and round the 4000 AA
        # break and high order Balmer lines in the optical. 
        zoom_and_restore(self.fig1, self.ax1, self.ax1t, \
            [1035-20, 1035+20], self.p.redshift, self.o.lam_reb[self.imask], self.o.flam_reb[self.imask], fnOut+'_ovi.pdf')
        zoom_and_restore(self.fig1, self.ax1, self.ax1t, \
            [1240-20, 1240+20], self.p.redshift, self.o.lam_reb[self.imask], self.o.flam_reb[self.imask], fnOut+'_nv.pdf')
        zoom_and_restore(self.fig1, self.ax1, self.ax1t, \
            [1400-20, 1400+20], self.p.redshift, self.o.lam_reb[self.imask], self.o.flam_reb[self.imask], fnOut+'_siiv.pdf')
        zoom_and_restore(self.fig1, self.ax1, self.ax1t, \
            [1550-20, 1550+20], self.p.redshift, self.o.lam_reb[self.imask], self.o.flam_reb[self.imask], fnOut+'_civ.pdf')
        zoom_and_restore(self.fig2, self.ax2, self.ax2t, \
            [3400.  , 4700   ], self.p.redshift, self.o.lam_reb[self.imask], self.o.flam_reb[self.imask], fnOut+'_4kb.pdf')
  
  
  
  
    def spec_build_resultvals(self): 
        self.result_spec = self.result_norm * self.s.flam_reb[self.result_iage,self.result_iebv] 
  
        self.result_specconv = convolve_concat(self.result_spec, \
             self.o.Nconcat, self.o.iconcat, self.o.sampingConcat, self.o.meanlamConcat, \
             self.result_fwhm) 
  
  
  
  
    def wrap_fit_1pop_red(self, minalg='leastsq'):
  
        self.minalg  = minalg
  
        self.fit_1pop_red(self.o.flam_reb)
  
        self.result_iage  = self.ibest_age
        self.result_age   = self.s.age_reb[self.ibest_age]
        self.result_iebv  = self.ibest_ebv
        self.result_ebv   = self.s.ebv[self.ibest_ebv]
        self.result_norm  = self.best_norm 
        self.result_fwhm  = self.s.fwhm[self.ibest_fwhm]
        self.result_ifwhm = self.ibest_fwhm
        self.result_chi2n = self.best_chi2n 
  
        self.plot_fit_12pop_red(npop=1, fnOut=self.p.fnFigroot1)
  
        self.result_qhi, self.result_qhei, self.result_qheii, self.result_lmech, self.result_emech = \
            self.s.calc_secondprop(self.result_age, self.result_norm)
  
        fhOut = open(self.p.fnProps1, 'w')
  
        s  = '# {:8}  {:10s}  {:8s}  {:8s}  {:8s}'.format('logAge', 'Normal.', 'E(B-V)', 'FWHM', 'chi2n')
        s += '  {:10s}  {:10s}  {:10s}  {:10s}  {:10s}'.format('qhi', 'qhei', 'qheii', 'lmech', 'Emech')
        print (s)
        fhOut.write(s+'\n')
        s  = '# {:8s}  {:10s}  {:8s}  {:8s}  {:8s}'.format('Myr', '', 'Mag', 'km/s', '')
        s += '  {:10s}  {:10s}  {:10s}  {:10s}  {:10s}'.format('Q/sec', 'Q/sec', 'Q/sec', 'erg/sec', 'erg')
        print (s)
        fhOut.write(s+'\n')
        s  = '  {:8.4f}  {:10.7g}  {:8.4f}  {:8.4f}  {:8.4f}'.format(self.result_age, self.result_norm, self.result_ebv, self.result_fwhm, self.result_chi2n)
        s += '  {:10.7g}  {:10.7g}  {:10.7g}  {:10.7g}  {:10.7g}   #main result'.format(self.result_qhi, self.result_qhei, self.result_qheii, self.result_lmech, self.result_emech)
        print (s)
        fhOut.write(s+'\n')
  
        self.mc_iage  = sp.zeros(self.p.Nstep, dtype=sp.int64  )
        self.mc_iebv  = sp.zeros(self.p.Nstep, dtype=sp.int64  )
        self.mc_ifwhm = sp.zeros(self.p.Nstep, dtype=sp.int64  )
        self.mc_age   = sp.zeros(self.p.Nstep, dtype=sp.float64)
        self.mc_ebv   = sp.zeros(self.p.Nstep, dtype=sp.float64)
        self.mc_norm  = sp.zeros(self.p.Nstep, dtype=sp.float64)
        self.mc_fwhm  = sp.zeros(self.p.Nstep, dtype=sp.float64)
        self.mc_chi2n = sp.zeros(self.p.Nstep, dtype=sp.float64)
        self.mc_qhi   = sp.zeros(self.p.Nstep, dtype=sp.float64)
        self.mc_qhei  = sp.zeros(self.p.Nstep, dtype=sp.float64)
        self.mc_qheii = sp.zeros(self.p.Nstep, dtype=sp.float64)
        self.mc_lmech = sp.zeros(self.p.Nstep, dtype=sp.float64)
        self.mc_emech = sp.zeros(self.p.Nstep, dtype=sp.float64)
  
        print ('Beginning MC simulation of', self.p.Nstep, 'realizations')
        for imc in range(self.p.Nstep): 
  
            randspec = self.o.flam_reb + sp.randn(self.o.Nlam_reb) * self.o.dflam_reb # * 100.
            self.fit_1pop_red(randspec)
  
            self.mc_iage [imc] = self.ibest_age
            self.mc_age  [imc] = self.s.age_reb[self.ibest_age]
            self.mc_iebv [imc] = self.ibest_ebv
            self.mc_ebv  [imc] = self.s.ebv[self.ibest_ebv]
            self.mc_norm [imc] = self.best_norm
            self.mc_fwhm [imc] = self.s.fwhm[self.ibest_fwhm]
            self.mc_ifwhm[imc] = self.ibest_fwhm
            self.mc_chi2n[imc] = self.best_chi2n
            
            self.mc_qhi[imc], self.mc_qhei[imc], self.mc_qheii[imc], self.mc_lmech[imc], self.mc_emech[imc] = \
                self.s.calc_secondprop(self.mc_age[imc], self.mc_norm[imc])
  
            s  = '  {:8.4f}  {:10.7g}  {:8.4f}  {:8.4f}  {:8.4f}'.format(self.mc_age[imc], self.mc_norm[imc], self.mc_ebv[imc], self.mc_fwhm[imc], self.mc_chi2n[imc])
            s += '  {:10.7g}  {:10.7g}  {:10.7g}  {:10.7g}  {:10.7g}   # MCinstance {:6d}'.format(self.mc_qhi[imc], self.mc_qhei[imc], self.mc_qheii[imc], self.mc_lmech[imc], self.mc_emech[imc], imc+1)
            fhOut.write(s+'\n')
            print (s)
  
        print ('exiting the MC simulator, and the wrapper')
        fhOut.close()
  
    
  
  
    #################3
    ## now duplicate the above with 2 populations
    #################3
    def fit_2pop_red(self, obsspec):
  
        print ('entered the 2pop fitter')
        Ndof = (self.Ndp-6-1)
        
        mat_norm1   = sp.zeros(( self.p.Nfwhm, self.p.Nage, self.p.Nage, self.p.Nebv ), dtype=sp.float64)
        mat_norm2   = sp.zeros(( self.p.Nfwhm, self.p.Nage, self.p.Nage, self.p.Nebv ), dtype=sp.float64)
        mat_chi2    = sp.ones (( self.p.Nfwhm, self.p.Nage, self.p.Nage, self.p.Nebv ), dtype=sp.float64) * 1.e40
  
        # TODO
        # ind_nearest find the cutoff between log10(4e7) [SN cutoff] and then execute age1 up to this and age2 from this to the end
        for iFwhm in range(self.p.Nfwhm): 
            for iAge1 in range(self.p.Nage):
                for iAge2 in range(iAge1+1, self.p.Nage):
                    for iEbv in range(self.p.Nebv):
                    
                      modspec1 = self.s.flam_reb[iAge1,iEbv]
                      modspec2 = self.s.flam_reb[iAge2,iEbv]
  
                      s1 = sp.sum ( ( modspec1[self.imask] / self.o.dflam_reb[self.imask] ) **2 )
                      s2 = sp.sum ( obsspec[self.imask] * modspec1[self.imask] / self.o.dflam_reb[self.imask]**2 )
                      s3 = sp.sum ( modspec1[self.imask] * modspec2[self.imask] / self.o.dflam_reb[self.imask]**2 )
                      s4 = sp.sum ( ( modspec2[self.imask] / self.o.dflam_reb[self.imask] ) **2 )
                      s5 = sp.sum ( obsspec[self.imask] * modspec2[self.imask] / self.o.dflam_reb[self.imask]**2 )
  
                      mat_norm1[iFwhm,iAge1,iAge2,iEbv] = ( s2 - s3*s5/s4 ) / ( s3**2 / s4 ) 
                      mat_norm2[iFwhm,iAge1,iAge2,iEbv] = ( s5 - mat_norm1[iFwhm,iAge1,iAge2,iEbv]*s3 ) / s4
  
                      if (mat_norm1[iFwhm,iAge1,iAge2,iEbv] < 0.) or (mat_norm2[iFwhm,iAge1,iAge2,iEbv] < 0.): 
                          mat_norm2[iFwhm,iAge1,iAge2,iEbv] = 0.  
                          mat_norm1[iFwhm,iAge1,iAge2,iEbv] = sp.sum(modspec1[self.imask] * obsspec[self.imask] / self.o.dflam_reb[self.imask]**2) / \
                                                              sp.sum(( modspec1[self.imask] / self.o.dflam_reb[self.imask] )**2)
  
                      mat_chi2[iFwhm,iAge1,iAge2,iEbv] = sp.sum( ( ( mat_norm1[iFwhm,iAge1,iAge2,iEbv] * modspec1[self.imask] + mat_norm2[iFwhm,iAge1,iAge2,iEbv] * \
                                                         modspec2[self.imask] - obsspec[self.imask]) / self.o.dflam_reb[self.imask] )**2 )
  
        self.ibest_fwhm, self.ibest_age1, self.ibest_age2, self.ibest_ebv = argmin_true(mat_chi2)
        self.best_norm1 = mat_norm1[self.ibest_fwhm, self.ibest_age1, self.ibest_age2, self.ibest_ebv]
        self.best_norm2 = mat_norm2[self.ibest_fwhm, self.ibest_age1, self.ibest_age2, self.ibest_ebv]
        self.best_chi2n = mat_chi2 [self.ibest_fwhm, self.ibest_age1, self.ibest_age2, self.ibest_ebv]/Ndof
  
  
  
  
    def wrap_fit_2pop_red(self, minalg='leastsq'):
  
        self.minalg  = minalg
  
        self.fit_2pop_red(self.o.flam_reb)
  
        self.result_iage1  = self.ibest_age1
        self.result_age1   = self.s.age_reb[self.ibest_age1]
        self.result_iage2  = self.ibest_age2
        self.result_age2   = self.s.age_reb[self.ibest_age2]
        self.result_iebv   = self.ibest_ebv
        self.result_ebv    = self.s.ebv[self.ibest_ebv]
        self.result_norm1  = self.best_norm1 
        self.result_norm2  = self.best_norm2
        self.result_fwhm   = self.s.fwhm[self.ibest_fwhm]
        self.result_ifwhm  = self.ibest_fwhm
        self.result_chi2n  = self.best_chi2n 
  
        self.plot_fit_12pop_red(npop=2, fnOut=self.p.fnFigroot2)
  
        self.result_qhi1, self.result_qhei1, self.result_qheii1, self.result_lmech1, self.result_emech1 = \
            self.s.calc_secondprop(self.result_age1, self.result_norm1)
        self.result_qhi2, self.result_qhei2, self.result_qheii2, self.result_lmech2, self.result_emech2 = \
            self.s.calc_secondprop(self.result_age2, self.result_norm2)
  
        fhOut = open(self.p.fnProps2, 'w')
  
        s  = '# {:8}  {:10s}  {:8}  {:10s}  {:8s}  {:8s}  {:8s}'.format('logAge1', 'Normal.1', 'logAge2', 'Normal.2', 'E(B-V)', 'FWHM', 'chi2n')
        s += '  {:10s}  {:10s}  {:10s}  {:10s}  {:10s}'.format('qhi1', 'qhei1', 'qheii1', 'lmech1', 'Emech1')
        s += '  {:10s}  {:10s}  {:10s}  {:10s}  {:10s}'.format('qhi2', 'qhei2', 'qheii2', 'lmech2', 'Emech2')
        print (s)
        fhOut.write(s+'\n')
        s  = '# {:8s}  {:10s}  {:8s}  {:10s}  {:8s}  {:8s}  {:8s}'.format('Myr', '', 'Myr', '', 'Mag', 'km/s', '')
        s += '  {:10s}  {:10s}  {:10s}  {:10s}  {:10s}'.format('Q/sec', 'Q/sec', 'Q/sec', 'erg/sec', 'erg')
        s += '  {:10s}  {:10s}  {:10s}  {:10s}  {:10s}'.format('Q/sec', 'Q/sec', 'Q/sec', 'erg/sec', 'erg')
        print (s)
        fhOut.write(s+'\n')
        s  = '  {:8.4f}  {:10.7g}  {:8.4f}  {:10.7g}  {:8.4f}  {:8.4f}  {:8.4f}'.format(self.result_age1, self.result_norm1, self.result_age2, self.result_norm2, self.result_ebv, self.result_fwhm, self.result_chi2n)
        s += '  {:10.7g}  {:10.7g}  {:10.7g}  {:10.7g}  {:10.7g}'.format(self.result_qhi1, self.result_qhei1, self.result_qheii1, self.result_lmech1, self.result_emech1)
        s += '  {:10.7g}  {:10.7g}  {:10.7g}  {:10.7g}  {:10.7g}   #main result'.format(self.result_qhi2, self.result_qhei2, self.result_qheii2, self.result_lmech2, self.result_emech2)
        print (s)
        fhOut.write(s+'\n')
  
        self.mc_iage1  = sp.zeros(self.p.Nstep, dtype=sp.int64  )
        self.mc_iage2  = sp.zeros(self.p.Nstep, dtype=sp.int64  )
        self.mc_iebv   = sp.zeros(self.p.Nstep, dtype=sp.int64  )
        self.mc_ifwhm  = sp.zeros(self.p.Nstep, dtype=sp.int64  )
        self.mc_age1    = sp.zeros(self.p.Nstep, dtype=sp.float64)
        self.mc_age2    = sp.zeros(self.p.Nstep, dtype=sp.float64)
        self.mc_ebv    = sp.zeros(self.p.Nstep, dtype=sp.float64)
        self.mc_norm1  = sp.zeros(self.p.Nstep, dtype=sp.float64)
        self.mc_norm2  = sp.zeros(self.p.Nstep, dtype=sp.float64)
        self.mc_fwhm   = sp.zeros(self.p.Nstep, dtype=sp.float64)
        self.mc_chi2n  = sp.zeros(self.p.Nstep, dtype=sp.float64)
        self.mc_qhi1   = sp.zeros(self.p.Nstep, dtype=sp.float64)
        self.mc_qhei1  = sp.zeros(self.p.Nstep, dtype=sp.float64)
        self.mc_qheii1 = sp.zeros(self.p.Nstep, dtype=sp.float64)
        self.mc_lmech1 = sp.zeros(self.p.Nstep, dtype=sp.float64)
        self.mc_emech1 = sp.zeros(self.p.Nstep, dtype=sp.float64)
        self.mc_qhi2   = sp.zeros(self.p.Nstep, dtype=sp.float64)
        self.mc_qhei2  = sp.zeros(self.p.Nstep, dtype=sp.float64)
        self.mc_qheii2 = sp.zeros(self.p.Nstep, dtype=sp.float64)
        self.mc_lmech2 = sp.zeros(self.p.Nstep, dtype=sp.float64)
        self.mc_emech2 = sp.zeros(self.p.Nstep, dtype=sp.float64)
  
        print ('Beginning MC simulation of', self.p.Nstep, 'realizations')
        for imc in range(self.p.Nstep): 
  
            randspec = self.o.flam_reb + sp.randn(self.o.Nlam_reb) * self.o.dflam_reb 
            self.fit_2pop_red(randspec)
  
            self.mc_iage1 [imc] = self.ibest_age1
            self.mc_age1  [imc] = self.s.age_reb[self.ibest_age1]
            self.mc_iage2 [imc] = self.ibest_age2
            self.mc_age2  [imc] = self.s.age_reb[self.ibest_age2]
            self.mc_iebv  [imc] = self.ibest_ebv
            self.mc_ebv   [imc] = self.s.ebv[self.ibest_ebv]
            self.mc_norm1 [imc] = self.best_norm1
            self.mc_norm2 [imc] = self.best_norm2
            self.mc_fwhm  [imc] = self.s.fwhm[self.ibest_fwhm]
            self.mc_ifwhm [imc] = self.ibest_fwhm
            self.mc_chi2n [imc] = self.best_chi2n
            
            self.mc_qhi1[imc], self.mc_qhei1[imc], self.mc_qheii1[imc], self.mc_lmech1[imc], self.mc_emech1[imc] = \
                self.s.calc_secondprop(self.mc_age1[imc], self.mc_norm1[imc])
            self.mc_qhi2[imc], self.mc_qhei2[imc], self.mc_qheii2[imc], self.mc_lmech2[imc], self.mc_emech2[imc] = \
                self.s.calc_secondprop(self.mc_age2[imc], self.mc_norm2[imc])
  
            s  = '  {:8.4f}  {:10.7g}  {:8.4f}  {:10.7g}  {:8.4f}  {:8.4f}  {:8.4f}'.format(self.mc_age1[imc], self.mc_norm1[imc], self.mc_age2[imc], self.mc_norm2[imc], self.mc_ebv[imc], self.mc_fwhm[imc], self.mc_chi2n[imc])
            s += '  {:10.7g}  {:10.7g}  {:10.7g}  {:10.7g}  {:10.7g}'.format(self.mc_qhi1[imc], self.mc_qhei1[imc], self.mc_qheii1[imc], self.mc_lmech1[imc], self.mc_emech1[imc])
            s += '  {:10.7g}  {:10.7g}  {:10.7g}  {:10.7g}  {:10.7g}   # MCinstance {:6d}'.format(self.mc_qhi2[imc], self.mc_qhei2[imc], self.mc_qheii2[imc], self.mc_lmech2[imc], self.mc_emech2[imc], imc+1)
            fhOut.write(s+'\n')
            print (s)
  
        print ('exiting the MC simulator, and the wrapper')
        fhOut.close()
  
  
  
  
    #############################################################################
    ### compute a synthetic spectrum from values in pars 
    #############################################################################
    def speccalc_npop_red(self, pars):
          
        parvals = pars.valuesdict()
        
        #MS using a different extinction for each population
        #ebv     = parvals['ebv']
        #fesc    = 10.** (-0.4 * self.klam * ebv)
  
        popkeys = [ a.replace('_a', '') for a in parvals.keys() if a.endswith('_a') ]
        
        #MS using a different extinction for each population
        ebv  = sp.zeros(self.Npop)
        fesc = sp.zeros((self.Npop,self.o.Nlam_reb))
        for ipop in range(self.Npop):
            ebv[ipop]  = parvals[popkeys[ipop]+'_ebv']
            fesc[ipop] = 10.**(-0.4 * self.klam * ebv[ipop])
        
        spectot = sp.zeros_like(self.o.lamo_reb)
        
        self.npop_ages = sp.zeros(self.Npop)
        
        for ipop in range(self.Npop):
            
            logage  = parvals[popkeys[ipop]+'_a']
            norm    = parvals[popkeys[ipop]+'_n']
            metal   = parvals[popkeys[ipop]+'_z']
            specpop = norm * 10.**self.s.gen_flam_from_splines(metal, logage, self.o.lamo_reb_log)

            # attenuate 
            specpop *= fesc[ipop]
            
            #self.npop_ages[ipop] = 
            #iage = ind_nearest(self.s.age_reb, self.npop_ages[ipop])
            #specpop = norm * self.s.flam_reb[iage]
            spectot += specpop
  
  
        # feed the convolution this spectrum, then the sampling information, then
        # the fwhm in km/s. 
        # this is no longer needed because the convolution is done in place in synspec standard_convolve()
        #spectot = convolve_concat(spectot, \
        #                          self.o.Nconcat, self.o.iconcat, self.o.sampingConcat, self.o.meanlamConcat, 350)

        #self.Nconcat       # number of concatenated spectra 
        #self.iconcat       # incremented indices of 0,0,0,  ... 1,1,1 ... identifying the relevant spectral pixels
        #self.sampingConcat # average sampling in each of the blocks
        #self.meanlamConcat # average wavelength of each of the blocks
  
        #mq = (spectot[self.imask] - self.o.flam_reb[self.imask]) #/ self.o.dflam_reb[self.imask]
        #return mq**2. / self.o.flam_reb[self.imask]    # for standard, linear residual
        #return  sp.sum( mq**2. + sp.log( 2.*sp.pi*self.o.dflam_reb[self.imask]**2. ) ) / -2.   # for the ln probability
  
        mq = (spectot[self.imask] - self.o.flam_reb[self.imask]) / self.o.dflam_reb[self.imask]
        return mq#**2. 

 
  
  
    def setup_npop_red(self, pops, ebvval=0.2, ebvmin=0., ebvmax=1.):
  
        self.Npop       = len(pops)
            
        self.lmpars = lmfit.Parameters()
        #MS using a different extinction for each population  
        #self.lmpars.add('ebv', vary=True, value=ebvval, min=ebvmin, max=ebvmax)
            
        # setting up the parameters of age and normalization for each population
        for ipop in range(self.Npop):
            print ('adding parameter', pops[ipop].name, 'age=', pops[ipop].aval)
            #MS using a different extinction for each population
            #MS setting ebv of secondary populations equal to ebv of the first one
            #if ipop > 0:
            #    self.lmpars.add(pops[ipop].name+'_ebv', expr=pops[0].name+'_ebv')
            #else:
            #    self.lmpars.add(pops[ipop].name+'_ebv', vary=True, value=ebvval, min=ebvmin, max=ebvmax)
            self.lmpars.add(pops[ipop].name+'_ebv', vary=True, value=ebvval, min=ebvmin, max=ebvmax)
            self.lmpars.add(pops[ipop].name+'_a', vary=pops[ipop].afit, value=pops[ipop].aval, min=pops[ipop].amin, max=pops[ipop].amax)
            self.lmpars.add(pops[ipop].name+'_n', vary=pops[ipop].nfit, value=pops[ipop].nval, min=pops[ipop].nmin, max=pops[ipop].nmax)
            self.lmpars.add(pops[ipop].name+'_z', vary=pops[ipop].zfit, value=pops[ipop].zval, min=pops[ipop].zmin, max=pops[ipop].zmax)



#MS I remove attributes for the MCMC
    def fit_npop_red(self, minalg='leastsq'):#, Nstep=1000, Nwalker=30, Nburn=100):
  
        self.minalg  = minalg
        #self.Nstep   = Nstep  
        #self.Nwalker = Nwalker
        #self.Nburn   = Nburn  
               
        print (self.minalg)
 
        # set up the object with the method+return quantity and the parameters to minimize. 
        self.minner = lmfit.Minimizer(self.speccalc_npop_red, self.lmpars)
                
        # first find the minimum with the LM method, then sample the posterior with emcee
        # are the results convergent? 
        print ('\nMinimizing with', self.minalg, 'algorithm (default=LM)')
        self.fitres_mi = self.minner.minimize(method=self.minalg, kws={'popsize':20, 'mutation':(1.,1.5), 'recombination':0.5})

        #MS setup minimization best fit values as initial guess for mcmc
        #MS commenting out mcmc
        #self.minner = lmfit.Minimizer(self.speccalc_npop_red, self.fitres_mi.params)
   
        #print ('Sampling posterior with emcee')
        #print ('    %d steps in %d walkers burning %d results' % (self.Nstep, self.Nwalker, self.Nburn) )
        #self.fitres_mc = self.minner.emcee(steps=self.Nstep, nwalkers=self.Nwalker, burn=self.Nburn, is_weighted=True)  # if tartget function returns a summation: float_behavior='posterior', 
 
 
  
  
    def calc_bestspec_npop_red(self, fnOut=None):    
          
        # calculate the reddening quantities
        #MS commenting out mcmc  
        #mcmc_ebv       = self.fitres_mc.flatchain['ebv'].values
        #self.ebv_best, self.ebv_lo, self.ebv_hi = sp.percentile(mcmc_ebv, [50., 15.8655, 84.1345])
        #fesc           = 10.** (-0.4 * self.klam * self.ebv_best)

        #MS same from minimization fit results (all variables ending with _mi in this function are part of this addition)
        #self.ebv_best_mi = self.fitres_mi.params['ebv'].value
        #fesc_mi     = 10.** (-0.4 * self.klam * self.ebv_best_mi)
  
        #parvals = self.fitres_mc.params.valuesdict()
        #print(parvals)
        #print(parvals.keys())
        parvals_mi = self.fitres_mi.params.valuesdict()
        #print(parvals_mi)
        self.popkeys   = [ a.replace('_a', '') for a in parvals_mi.keys() if a.endswith('_a') ]
        #print(self.popkeys)        

        #self.age_best   = sp.zeros(self.Npop)
        #self.age_lo     = sp.zeros(self.Npop)
        #self.age_hi     = sp.zeros(self.Npop)
        #self.met_best   = sp.zeros(self.Npop)
        #self.met_lo     = sp.zeros(self.Npop)
        #self.met_hi     = sp.zeros(self.Npop)
        #self.norm_best  = sp.zeros(self.Npop)
        #self.norm_lo    = sp.zeros(self.Npop)
        #self.norm_hi    = sp.zeros(self.Npop)

        #self.flam_ind   = sp.zeros((self.Npop,self.o.Nlam_reb))

        #self.qhi_best   = sp.zeros(self.Npop)
        #self.qhi_lo     = sp.zeros(self.Npop)
        #self.qhi_hi     = sp.zeros(self.Npop)
        #self.qhei_best  = sp.zeros(self.Npop)
        #self.qhei_lo    = sp.zeros(self.Npop)
        #self.qhei_hi    = sp.zeros(self.Npop)
        #self.qheii_best = sp.zeros(self.Npop)
        #self.qheii_lo   = sp.zeros(self.Npop)
        #self.qheii_hi   = sp.zeros(self.Npop)
        #self.lmech_best = sp.zeros(self.Npop)
        #self.lmech_lo   = sp.zeros(self.Npop)
        #self.lmech_hi   = sp.zeros(self.Npop)
        #self.emech_best = sp.zeros(self.Npop)
        #self.emech_lo   = sp.zeros(self.Npop)
        #self.emech_hi   = sp.zeros(self.Npop)

        self.ebv_best_mi  = sp.zeros(self.Npop)        
        self.age_best_mi  = sp.zeros(self.Npop)
        self.met_best_mi  = sp.zeros(self.Npop)
        self.norm_best_mi = sp.zeros(self.Npop)

        self.qhi_best_mi   = sp.zeros(self.Npop)
        self.qhei_best_mi  = sp.zeros(self.Npop)
        self.qheii_best_mi = sp.zeros(self.Npop)
        self.lmech_best_mi = sp.zeros(self.Npop)
        self.emech_best_mi = sp.zeros(self.Npop)

        self.fesc_mi      = sp.zeros((self.Npop,self.o.Nlam_reb))
        self.flam_ind_mi  = sp.zeros((self.Npop,self.o.Nlam_reb))

        self.flux1270     = sp.zeros(self.Npop)
        
        for ipop in range(self.Npop):
            popstring = self.popkeys[ipop]
            #self.age_best  [ipop] = parvals[popstring+'_a']
            #self.met_best  [ipop] = parvals[popstring+'_z']
            #self.norm_best [ipop] = parvals[popstring+'_n']
            #self.dage_best [ipop] = self.fitres_mc.params[popstring+'_a'].stderr
            #self.dmet_best [ipop] = self.fitres_mc.params[popstring+'_z'].stderr
            #self.dnorm_best[ipop] = self.fitres_mc.params[popstring+'_n'].stderr
            #
            #d = self.s.gen_sec_from_splines(self.met_best[ipop], self.age_best[ipop])
            #self.qhi_best  [ipop] = self.norm_best[ipop] * 10.**d[0] 
            #self.qhei_best [ipop] = self.norm_best[ipop] * 10.**d[1] 
            #self.qheii_best[ipop] = self.norm_best[ipop] * 10.**d[2] 
            #self.lmech_best[ipop] = self.norm_best[ipop] * 10.**d[3] 
            #self.emech_best[ipop] = self.norm_best[ipop] * 10.**d[4] 

            #mcmc_a     = self.fitres_mc.flatchain[popstring+'_a'].values
            #mcmc_z     = self.fitres_mc.flatchain[popstring+'_z'].values
            #mcmc_z = sp.ones(len(mcmc_a)) * self.p.metalvals[0]
            #mcmc_n     = self.fitres_mc.flatchain[popstring+'_n'].values
            #mcmc_qhi   = sp.zeros_like(mcmc_a)
            #mcmc_qhei  = sp.zeros_like(mcmc_a)
            #mcmc_qheii = sp.zeros_like(mcmc_a)
            #mcmc_lmech = sp.zeros_like(mcmc_a)
            #mcmc_emech = sp.zeros_like(mcmc_a)

            #for imc in range(len(mcmc_a)):
                #d = self.s.gen_sec_from_splines(mcmc_z[ipop], mcmc_a[ipop])
                #mcmc_qhi  [imc] = mcmc_n[imc] * 10.**d[0] 
                #mcmc_qhei [imc] = mcmc_n[imc] * 10.**d[1] 
                #mcmc_qheii[imc] = mcmc_n[imc] * 10.**d[2] 
                #mcmc_lmech[imc] = mcmc_n[imc] * 10.**d[3] 
                #mcmc_emech[imc] = mcmc_n[imc] * 10.**d[4] 

            #self.age_best  [ipop], self.age_lo  [ipop], self.age_hi  [ipop] = sp.percentile(mcmc_a    , [50., 15.8655, 84.1345])
            #print(mcmc_a.shape)
            #self.met_best  [ipop], self.met_lo  [ipop], self.met_hi  [ipop] = sp.percentile(mcmc_z    , [50., 15.8655, 84.1345])  
            #self.norm_best [ipop], self.norm_lo [ipop], self.norm_hi [ipop] = sp.percentile(mcmc_n    , [50., 15.8655, 84.1345])  
            #self.qhi_best  [ipop], self.qhi_lo  [ipop], self.qhi_hi  [ipop] = sp.percentile(mcmc_qhi  , [50., 15.8655, 84.1345])  
            #self.qhei_best [ipop], self.qhei_lo [ipop], self.qhei_hi [ipop] = sp.percentile(mcmc_qhei , [50., 15.8655, 84.1345])
            #self.qheii_best[ipop], self.qheii_lo[ipop], self.qheii_hi[ipop] = sp.percentile(mcmc_qheii, [50., 15.8655, 84.1345])
            #self.lmech_best[ipop], self.lmech_lo[ipop], self.lmech_hi[ipop] = sp.percentile(mcmc_lmech, [50., 15.8655, 84.1345])
            #self.emech_best[ipop], self.emech_lo[ipop], self.emech_hi[ipop] = sp.percentile(mcmc_emech, [50., 15.8655, 84.1345])

            #self.flam_ind[ipop] = self.norm_best[ipop] * 10.**self.s.gen_flam_from_splines(
                                  #self.met_best[ipop], self.age_best[ipop], self.o.lamo_reb_log) * fesc

            #printing the flux best model of the minimization fit 
            self.norm_best_mi[ipop] = self.fitres_mi.params[popstring+'_n'].value
            self.met_best_mi[ipop]  = self.fitres_mi.params[popstring+'_z'].value
            self.age_best_mi[ipop]  = self.fitres_mi.params[popstring+'_a'].value
            self.ebv_best_mi[ipop]  = self.fitres_mi.params[popstring+'_ebv'].value 

            self.fesc_mi    [ipop] = 10.** (-0.4 * self.klam * self.ebv_best_mi[ipop])
            self.flam_ind_mi[ipop] = self.norm_best_mi[ipop] * 10.**self.s.gen_flam_from_splines(
                                  self.met_best_mi[ipop], self.age_best_mi[ipop], self.o.lamo_reb_log) * self.fesc_mi[ipop]

            
            lam1270 = 0
            while self.o.lamo_reb[lam1270] < 1270: 
                lam1270 += 1

            self.flux1270   [ipop] = self.flam_ind_mi[ipop][lam1270]


            #MS deriving ionizing rates and energies wiht lmfit best values
            d_mi = self.s.gen_sec_from_splines(self.met_best_mi[ipop], self.age_best_mi[ipop])
            self.qhi_best_mi  [ipop] = self.norm_best_mi[ipop] * 10.**d_mi[0] 
            self.qhei_best_mi [ipop] = self.norm_best_mi[ipop] * 10.**d_mi[1] 
            self.qheii_best_mi[ipop] = self.norm_best_mi[ipop] * 10.**d_mi[2] 
            self.lmech_best_mi[ipop] = self.norm_best_mi[ipop] * 10.**d_mi[3] 
            self.emech_best_mi[ipop] = self.norm_best_mi[ipop] * 10.**d_mi[4] 

            # need a solution for the errors on secondary quantities, now fn.s are spline and no longer differentiatable
            # comment for now.. 
#            dqhi_logage   = abs(self.s.dpolyHi   (self.age_best[ipop]))*self.dage_best[ipop]  # get the error from the derivative
#            dqhei_logage  = abs(self.s.dpolyHei  (self.age_best[ipop]))*self.dage_best[ipop]  
#            dqheii_logage = abs(self.s.dpolyHeii (self.age_best[ipop]))*self.dage_best[ipop]  
#            demech_logage = abs(self.s.dpolyEmech(self.age_best[ipop]))*self.dage_best[ipop]  
#            dlmech_logage = abs(self.s.dpolyLmech(self.age_best[ipop]))*self.dage_best[ipop]  
#
#            dqhi_age      = dqhi_logage   / 0.434 * self.s.qhi_reb  [iage]    # calculate in linear.  note we take the UNNORMALIZED VALUE so we can compute the error propagation.
#            dqhei_age     = dqhei_logage  / 0.434 * self.s.qhei_reb [iage]    
#            dqheii_age    = dqheii_logage / 0.434 * self.s.qheii_reb[iage]    
#            demech_age    = demech_logage / 0.434 * self.s.emechTot_reb[iage]    
#            dlmech_age    = dlmech_logage / 0.434 * self.s.lmechTot_reb[iage]    
#
#            self.dqhi_best  [ipop] = self.qhi_best  [ipop] * sp.sqrt( (dqhi_age  /self.s.qhi_reb     [iage])**2. + (self.dnorm_best[ipop]/self.norm_best[ipop])**2. )  # and propagate with normalization error
#            self.dqhei_best [ipop] = self.qhei_best [ipop] * sp.sqrt( (dqhei_age /self.s.qhei_reb    [iage])**2. + (self.dnorm_best[ipop]/self.norm_best[ipop])**2. )  
#            self.dqheii_best[ipop] = self.qheii_best[ipop] * sp.sqrt( (dqheii_age/self.s.qheii_reb   [iage])**2. + (self.dnorm_best[ipop]/self.norm_best[ipop])**2. )  
#            self.demech_best[ipop] = self.emech_best[ipop] * sp.sqrt( (demech_age/self.s.emechTot_reb[iage])**2. + (self.dnorm_best[ipop]/self.norm_best[ipop])**2. )  
#            self.dlmech_best[ipop] = self.lmech_best[ipop] * sp.sqrt( (dlmech_age/self.s.lmechTot_reb[iage])**2. + (self.dnorm_best[ipop]/self.norm_best[ipop])**2. )  


            # calculate the individual spectra, for best normalization, reddening#, and
  
            # then convolve with the best FWHM. 
            #t = self.norm_best[ipop] * self.s.flam_reb[iage] * fesc
            #self.flam_ind[ipop] = convolve_concat(t, \
            #  self.o.Nconcat, self.o.iconcat, self.o.sampingConcat, self.o.meanlamConcat), self.fwhm_best)
  
        #self.flam_best   = self.flam_ind  .sum(axis=0)
        #self.qhi_bestt   = self.qhi_best  .sum()
        #self.qhei_bestt  = self.qhei_best .sum()
        #self.qheii_bestt = self.qheii_best.sum()
        #self.lmech_bestt = self.lmech_best.sum()
        #self.emech_bestt = self.emech_best.sum()

        self.qhi_bestt_mi   = self.qhi_best_mi  .sum()
        self.qhei_bestt_mi  = self.qhei_best_mi .sum()
        self.qheii_bestt_mi = self.qheii_best_mi.sum()
        self.lmech_bestt_mi = self.lmech_best_mi.sum()
        self.emech_bestt_mi = self.emech_best_mi.sum()  

        self.flam_best_mi = self.flam_ind_mi.sum(axis=0)
        
        #self.chi2_best  = sp.sum( (self.o.flam_reb[self.imask]-self.flam_best[self.imask])**2./self.o.dflam_reb[self.imask]**2. )
        #self.chi2n_best = self.chi2_best / (self.Ndp - 1)

        self.chi2_best_mi  = sp.sum( (self.o.flam_reb[self.imask]-self.flam_best_mi[self.imask])**2./self.o.dflam_reb[self.imask]**2. )
        self.chi2n_best_mi = self.chi2_best_mi / (self.Ndp - 1)
        
        #MS I want to call pop1 the population that weights more on the flux and pop2 the other one, and calssify accordingly all the parameters

        inds = sp.flip(self.flux1270.argsort())

        self.ebv_best_mi  = self.ebv_best_mi[inds]        
        self.age_best_mi  = self.age_best_mi[inds]  
        self.met_best_mi  = self.met_best_mi[inds]  
        self.norm_best_mi = self.norm_best_mi[inds]  

        self.qhi_best_mi   = self.qhi_best_mi[inds]  
        self.qhei_best_mi  = self.qhei_best_mi[inds]  
        self.qheii_best_mi = self.qheii_best_mi[inds]  
        self.lmech_best_mi = self.lmech_best_mi[inds]  
        self.emech_best_mi = self.emech_best_mi[inds]  

        self.fesc_mi      = self.fesc_mi[inds]  
        self.flam_ind_mi  = self.flam_ind_mi[inds]  
            

        print ('\n')
  
        s=[]
        #s.append ( '!!! Temporary results !!!  Ndof is not quite right!' )
        #s.append ( 'Derived quantities and confidence ranges.')
        #s.append ( '  Given as BestValue [ LowVal HighVal ]')
        #s.append ( '    and computed assuming symmetric error distribution from covariance matrix' )
        #s.append ( '' )
       
        #s.append('  Dust reddening, given as E(B-V), in magnitudes.  Same for each population')
        #s.append('    {:1.5e} [ {:1.5e} {:1.5e} ]'.format(self.ebv_best, self.ebv_lo, self.ebv_hi))

        #s.append('  Stellar population ages, t, in logYears')
        #for ipop in range(self.Npop):
        #    s.append('{:>20} : {:1.5e} [ {:1.5e} {:1.5e} ]'.format(
        #             self.popkeys  [ipop], self.age_best [ipop], self.age_lo [ipop], self.age_hi [ipop]))

        #s.append('  Stellar population metallicities, Z, in X/H')
        #for ipop in range(self.Npop):
        #    s.append('{:>20} : {:1.5e} [ {:1.5e} {:1.5e} ]'.format(
        #             self.popkeys  [ipop], self.met_best [ipop], self.met_lo [ipop], self.met_hi [ipop]))

        #s.append('  Stellar population normalizations, n, given in units of input spectra')
        #for ipop in range(self.Npop):
        #    s.append('{:>20} : {:1.5e} [ {:1.5e} {:1.5e} ]'.format(
        #             self.popkeys  [ipop], self.norm_best[ipop], self.norm_lo[ipop], self.norm_hi[ipop]))

        #s.append('  H I ionizing photon rate, given in sec^-1')
        #for ipop in range(self.Npop):
        #    s.append('{:>20} : {:1.5e} [ {:1.5e} {:1.5e} ]'.format(
        #             self.popkeys  [ipop], self.qhi_best [ipop], self.qhi_lo [ipop], self.qhi_hi [ipop]))

        #s.append('  He I ionizing photon rate, given in sec^-1')
        #for ipop in range(self.Npop):
        #    s.append('{:>20} : {:1.5e} [ {:1.5e} {:1.5e} ]'.format(
        #             self.popkeys  [ipop], self.qhei_best[ipop], self.qhei_lo[ipop], self.qhei_hi[ipop]))

        #s.append('  He II ionizing photon rate, given in sec^-1')
        #for ipop in range(self.Npop):
        #    s.append('{:>20} : {:1.5e} [ {:1.5e} {:1.5e} ]'.format(
        #             self.popkeys  [ipop], self.qheii_best[ipop], self.qheii_lo[ipop], self.qheii_hi[ipop]))

        #s.append('  Integrated mechanical energy since t=0, given in erg')
        #for ipop in range(self.Npop):
        #    s.append('{:>20} : {:1.5e} [ {:1.5e} {:1.5e} ]'.format(
        #             self.popkeys  [ipop], self.emech_best[ipop], self.emech_lo[ipop], self.emech_hi[ipop]))

        #s.append('  Instantaneous mechanical power, given in erg sec^-1')
        #for ipop in range(self.Npop):
        #    s.append('{:>20} : {:1.5e} [ {:1.5e} {:1.5e} ]'.format(
        #             self.popkeys  [ipop], self.lmech_best[ipop], self.lmech_lo[ipop], self.lmech_hi[ipop]))

        #s.append('    chi2 : {:1.5e}'.format(self.chi2_best))
        #s.append('   chi2n : {:1.5e} for {:d} datapoints (N params is wrong here!)'.format(self.chi2n_best, self.Ndp))
        #s.append('')

        #MS reporting also the values of the lmfit minimization, which are those that I care about...
        s.append('Results from lmfit minimization method, errors are not calculated:')

        s.append('  Light weight (i.e. flux at 1270A):')
        for ipop in range(self.Npop):
            s.append('{:>20} : {:1.5e}'.format(
                     self.popkeys  [ipop], sp.flip(self.flux1270)[ipop]))

        s.append('  Dust reddening, given as E(B-V), in magnitudes. Same for each population')
        for ipop in range(self.Npop):
            s.append('    {:1.5e}'.format(self.ebv_best_mi [ipop]))

        s.append('  Stellar population ages, t, in logYears')
        for ipop in range(self.Npop):
            s.append('{:>20} : {:1.5e}'.format(
                     self.popkeys  [ipop], self.age_best_mi [ipop]))

        s.append('  Stellar population metallicities, Z, in X/H')
        for ipop in range(self.Npop):
            s.append('{:>20} : {:1.5e}'.format(
                     self.popkeys  [ipop], self.met_best_mi [ipop]))

        s.append('  Stellar population normalizations, n, given in units of input spectra')
        for ipop in range(self.Npop):
            s.append('{:>20} : {:1.5e}'.format(
                     self.popkeys  [ipop], self.norm_best_mi[ipop]))

        s.append('  H I ionizing photon rate, given in sec^-1')
        for ipop in range(self.Npop):
            s.append('{:>20} : {:1.5e}'.format(
                     self.popkeys  [ipop], self.qhi_best_mi [ipop]))

        s.append('  He I ionizing photon rate, given in sec^-1')
        for ipop in range(self.Npop):
            s.append('{:>20} : {:1.5e}'.format(
                     self.popkeys  [ipop], self.qhei_best_mi[ipop]))

        s.append('  He II ionizing photon rate, given in sec^-1')
        for ipop in range(self.Npop):
            s.append('{:>20} : {:1.5e}'.format(
                     self.popkeys  [ipop], self.qheii_best_mi[ipop]))

        s.append('  Integrated mechanical energy since t=0, given in erg')
        for ipop in range(self.Npop):
            s.append('{:>20} : {:1.5e}'.format(
                     self.popkeys  [ipop], self.emech_best_mi[ipop]))

        s.append('  Instantaneous mechanical power, given in erg sec^-1')
        for ipop in range(self.Npop):
            s.append('{:>20} : {:1.5e}'.format(
                     self.popkeys  [ipop], self.lmech_best_mi[ipop]))

        s.append('    chi2 : {:1.5e}'.format(self.chi2_best_mi))
        s.append('   chi2n : {:1.5e} for {:d} datapoints (N params is wrong here!)'.format(self.chi2n_best_mi, self.Ndp))
        s.append('')

  
        if fnOut : fhOut = open(fnOut, 'w')
  
        for l in s:
            print (l)
            if fnOut : fhOut.write('#'+l+'\n')
 
        # to write to the output  
        #self.flam_ind[ipop] = self.norm_best[ipop] * self.s.flam_reb[iage] * fesc
        #self.flam_best   = self.flam_ind  .sum(axis=0)
        #for ipop in range(self.Npop):
        #    s.append('{:>20}'.format(  self.popkeys[ipop]  ))
        if fnOut : 
            s = '#{:>19}  {:>20}  {:>20}  {:>20}'.format('lambda', 'flam_obs', 'dflam_obs', 'dq_obs')
            for ipop in range(self.Npop): s += '  {:>20}'.format('flam_'+self.popkeys[ipop])
            s += '  {:>20}  {:>20}  {:>20}\n'.format('flam_mod_tot_mi', 'flam_Normalized', 'mask')
            fhOut.write(s)

            for ilam in range(self.o.Nlam_reb):
                s = '{:>20.6f}  {:>20.8e}  {:>20.8e}  {:>20.8e}'.format(
                    self.o.lamo_reb[ilam], self.o.flam_reb[ilam], self.o.dflam_reb[ilam], self.o.dq_reb[ilam])
                for ipop in range(self.Npop): 
                    s += '  {:>20.8e}'.format( self.flam_ind_mi[ipop,ilam] )
                s += '  {:>20.8e}  {:>20.8e}  {:>20d}\n'.format( self.flam_best_mi[ilam], self.o.flam_reb[ilam] / self.flam_best_mi[ilam], self.imask[ilam])
                fhOut.write(s)
  
        if fnOut : fhOut.close()




  
  
  
  
    def show_fitres_npop_red(self):
 
        print (self.minalg) 
        print ('*** Fit results from the', self.minalg, 'minimization only\n********************') 
        lmfit.report_fit(self.fitres_mi)
  
        #print ('\n*** And errors from Markov Chain Monte Carlo\n********************') 
        #lmfit.report_fit(self.fitres_mc)
  
  
  
  
    def plot_npop_red(self, fnOut='delme'):
         
        fontsize = 12 
        #MS commenting out mcmc         
        for ipop in range(self.Npop):
            #label_str = r'Pop{:2d}: {:1.3f} Myr. Z={:1.5f}. {:.0f} M$_\odot$'.format(
            #              ipop+1, 10**self.age_best[ipop]/10**6, self.met_best[ipop], self.norm_best[ipop]*10**6)
            #self.ax1.plot(self.o.lamo_reb, self.flam_ind[ipop] , drawstyle='steps', ls='-', c=collist[ipop], label=label_str) 
            #self.ax2.plot(self.o.lamo_reb, self.flam_ind[ipop] , ls='-'     , c=collist[ipop], label=label_str)
            
            #MS: print in the legend also the results of lmfit (all variables that end with _mi have been added for this purpose)
            label_str_mi = r'Pop{:2d}: {:1.3f} Myr. Z={:1.5f}. {:.0f} M$_\odot$ E$_\mathrm{{B-V}}=${:1.3f}'.format(
                          ipop+1, 10**self.age_best_mi[ipop]/10**6, self.met_best_mi[ipop], self.norm_best_mi[ipop]*10**6, self.ebv_best_mi[ipop])
            #MS: if I use only one population I want to match the color and the line style with the final model
            if self.Npop == 1:
                line_s = '-'
                line_c = 'C0'
            else:
                line_s = '--'
                line_c = collist[ipop]
            self.ax1.plot(self.o.lamo_reb, self.flam_ind_mi[ipop] , drawstyle='steps', ls=line_s, c=line_c, label=label_str_mi) 
            self.ax2.plot(self.o.lamo_reb, self.flam_ind_mi[ipop] , ls=line_s, c=line_c, label=label_str_mi) 
        
        #self.ax1.text(0.55, 0.92, r'MCMC E$_\mathrm{{B-V}}=${:1.3f}  $\chi^2_\mathrm{{d.o.f.}}=${:2.3f}'.format(self.ebv_best, self.chi2n_best), verticalalignment='center', transform=self.ax1.transAxes, fontsize=fontsize)
        #self.ax1.text(0.55, 0.85, r'$\chi^2_\mathrm{{d.o.f.}}=${:2.3f}'.format(self.chi2n_best), verticalalignment='center', transform=self.ax1.transAxes, fontsize=fontsize)
        #self.ax2.text(0.05, 0.92, r'MCMC E$_\mathrm{{B-V}}=${:1.3f}  $\chi^2_\mathrm{{d.o.f.}}=${:2.3f}'.format(self.ebv_best, self.chi2n_best), verticalalignment='center', transform=self.ax2.transAxes, fontsize=fontsize)
          
        self.ax1.text(0.72, 0.80, r'lmfit $\chi^2_\mathrm{{d.o.f.}}=${:2.3f}'.format(self.chi2n_best_mi), verticalalignment='center', transform=self.ax1.transAxes, fontsize=fontsize)
        self.ax2.text(0.05, 0.80, r'lmfit $\chi^2_\mathrm{{d.o.f.}}=${:2.3f}'.format(self.chi2n_best_mi), verticalalignment='center', transform=self.ax2.transAxes, fontsize=fontsize)
        #self.ax2.text(0.7, 0.59, r'$\chi^2_\mathrm{{d.o.f.}}=${:2.3f}'.format(self.chi2n_best), verticalalignment='center', transform=self.ax2.transAxes, fontsize=fontsize)
  
        #self.ax1.plot(self.o.lamo_reb, self.flam_best, drawstyle='steps', ls='-', c='k')
        #self.ax2.plot(self.o.lamo_reb, self.flam_best, ls='-', c='k')
        self.ax1.plot(self.o.lamo_reb, self.flam_best_mi, drawstyle='steps', ls='-', c='C0')
        self.ax2.plot(self.o.lamo_reb, self.flam_best_mi, ls='-', c='C0')
        self.ax1.legend(loc=2)
        self.ax2.legend(loc=1)

	#MS 
        self.ax1.set_ylim(0, 20.)
        self.ax2.set_ylim(0, 1.5 * self.o.flam_op_reb.mean())
  
        self.ax1t = self.ax1.twiny()
        self.ax2t = self.ax2.twiny()
        self.ax1t.set_xbound(sp.array ( self.ax1.get_xbound() ) / (1.+self.p.redshift) )  
        self.ax2t.set_xbound(sp.array ( self.ax2.get_xbound() ) / (1.+self.p.redshift) )  
  
        self.ax1.xaxis.set_minor_locator(MultipleLocator(100))
        self.ax2.xaxis.set_minor_locator(MultipleLocator(100))
        self.ax1t.xaxis.set_minor_locator(MultipleLocator(100))
        self.ax2t.xaxis.set_minor_locator(MultipleLocator(100))
  
        self.ax1t.set_xlabel('Restframe Wavelength [ $\mathrm{\AA}$ ]')
        self.ax2t.set_xlabel('Restframe Wavelength [ $\mathrm{\AA}$ ]')
  
        self.fig1.savefig(fnOut+'_uv.png')
        self.fig2.savefig(fnOut+'_op.png')
  
        # plot some zoom-ins around the P Cyg lines in the UV, and round the 4000 AA
        # break and high order Balmer lines in the optical. 
        zoom_and_restore(self.fig1, self.ax1, self.ax1t, \
            [1035-20, 1035+20], self.p.redshift, self.o.lamo_reb[self.imask], self.o.flam_reb[self.imask], fnOut+'_ovi.png')
        zoom_and_restore(self.fig1, self.ax1, self.ax1t, \
            [1240-20, 1240+20], self.p.redshift, self.o.lamo_reb[self.imask], self.o.flam_reb[self.imask], fnOut+'_nv.png')
        zoom_and_restore(self.fig1, self.ax1, self.ax1t, \
            [1400-20, 1400+20], self.p.redshift, self.o.lamo_reb[self.imask], self.o.flam_reb[self.imask], fnOut+'_siiv.png')
        zoom_and_restore(self.fig1, self.ax1, self.ax1t, \
            [1550-20, 1550+20], self.p.redshift, self.o.lamo_reb[self.imask], self.o.flam_reb[self.imask], fnOut+'_civ.png')
        zoom_and_restore(self.fig1, self.ax1, self.ax1t, \
            [1640-20, 1640+20], self.p.redshift, self.o.lamo_reb[self.imask], self.o.flam_reb[self.imask], fnOut+'_heii.png')
        zoom_and_restore(self.fig2, self.ax2, self.ax2t, \
            [3400.  , 4700   ], self.p.redshift, self.o.lamo_reb[self.imask], self.o.flam_reb[self.imask], fnOut+'_4kb.png')
  
  
        # plot the corner diagram and the walkers
        #cornfig = corner.corner(self.fitres_mc.flatchain, labels=self.fitres_mc.var_names,  \
        #                        truths=list(self.fitres_mc.params.valuesdict().values()))
        #cornfig.savefig(fnOut+'_corner.pdf')
  
        
        #plot_walkers(self.fitres_mc, fnOut+'_walkers.pdf')
  
  
        #pobuv = open(fnOut+'_uv.pkl'    , 'wb')
        #pobop = open(fnOut+'_op.pkl'    , 'wb')
        #pobco = open(fnOut+'_corner.pkl', 'wb')
        #pickle.dump(self.fig1,  pobuv)
        #pickle.dump(self.fig2,  pobop)
        #pickle.dump(cornfig  ,  pobco)
        #pobuv.close()
        #pobop.close()
        #pobco.close()
  
  
  
  
    #############################################################################
    ### make an initial guess for the normalizaions of the population(s)
    ### analytical solutions for the normalization.  Follows standard methods if
    ### Npop = 1 or Npop = 2.  If Npop > 2 it follows the Npop = 2 method and
    ### writes zero normalization to the other populations. 
    #############################################################################
    def write_energy_history(self, pars, fnOut='delme'):
         
        fh = open(fnOut, 'w') 
        #s  = '# Ionizing photon and mechanical energy rates, from mcmc results\n'
        #fh.write(s)
        #s  = '# Numer of popualations\n#     {:>3d}\n'.format(self.Npop)
        #fh.write(s)
        #s  = '# names and number of entries:\n'
        #fh.write(s)
        #for ipop in range(self.Npop):
        #    age_best   = self.age_best [ipop]
        #    met_best   = self.met_best [ipop]
        #    norm_best  = self.norm_best[ipop]
        #    fh.write('# {:>30s}  {:1.5f}  {:1.5f}  {:1.5e}\n'.format(self.popkeys[ipop], age_best, met_best, norm_best))
        #fh.write('#\n')
  
        #for ipop in range(self.Npop):
  
        #    popstring = self.popkeys  [ipop]
        #    age_best  = self.age_best [ipop]
        #    met_best  = self.met_best [ipop]
        #    norm_best = self.norm_best[ipop]

        #    s  = '# population {:>4d}  , name = {:>30s}\n'.format(ipop+1, popstring)
        #    fh.write(s)
        #    s = '# log age = {:1.5f}  metallicitiy = {:1.5f}  norm = {:1.5e}\n'.format(age_best, met_best, self.norm_best[ipop])
        #    fh.write(s)
          
        #    s = '# {:>8}  {:>11}  {:>11}  {:>11}  {:>11}  {:>11}\n'.format('logage' ,
        #                                                                   'Q(HI)'  ,
        #                                                                   'Q(HeI)' ,
        #                                                                   'Q(HeII)',
        #                                                                   'Lmech'  ,
        #                                                                   'Emech')
        #    fh.write(s)
        #    s = '# {:>8}  {:>11}  {:>11}  {:>11}  {:>11}  {:>11}\n'.format('yr'      ,
        #                                                                   'phot/sec',
        #                                                                   'phot/sec',
        #                                                                   'phot/sec',
        #                                                                   'erg'     ,
        #                                                                   'erg/sec')
        #    fh.write(s)
        #    s = '# {:>8}  {:>11}  {:>11}  {:>11}  {:>11}  {:>11}\n'.format('(1)',
        #                                                                   '(2)',
        #                                                                   '(3)',
        #                                                                   '(4)',
        #                                                                   '(5)',
        #                                                                   '(6)')
        #    fh.write(s)

        #    NageWrite   = 500 
        #    age_write   = sp.linspace(4., age_best, NageWrite)
        #    qhi_write   = sp.zeros_like(age_write)
        #    qhei_write  = sp.zeros_like(age_write)
        #    qheii_write = sp.zeros_like(age_write)
        #    lmech_write = sp.zeros_like(age_write)
        #    emech_write = sp.zeros_like(age_write)

        #    for iage in range(NageWrite): 
        #        d = self.s.gen_sec_from_splines(met_best, age_write[iage])
        #        qhi_write  [iage] = norm_best*10.**d[0]
        #        qhei_write [iage] = norm_best*10.**d[1]
        #        qheii_write[iage] = norm_best*10.**d[2]
        #        lmech_write[iage] = norm_best*10.**d[3]
        #        emech_write[iage] = norm_best*10.**d[4]
  
        #        s = '  {:1.4f}  {:1.5e}  {:1.5e}  {:1.5e}  {:1.5e}  {:1.5e}\n'.format(age_write  [iage],
        #                                                                              qhi_write  [iage],
        #                                                                              qhei_write [iage],
        #                                                                              qheii_write[iage],
        #                                                                              lmech_write[iage],
        #                                                                              emech_write[iage])
        #        fh.write(s)

        s  = '# Ionizing photon and mechanical energy rates, from lmfit results\n'
        fh.write(s)
        s  = '# Numer of popualations\n#     {:>3d}\n'.format(self.Npop)
        fh.write(s)
        s  = '# names and number of entries:\n'
        fh.write(s)
        for ipop in range(self.Npop):
            age_best_mi   = self.age_best_mi [ipop]
            met_best_mi   = self.met_best_mi [ipop]
            norm_best_mi  = self.norm_best_mi[ipop]
            fh.write('# {:>30s}  {:1.5f}  {:1.5f}  {:1.5e}\n'.format(self.popkeys[ipop], age_best_mi, met_best_mi, norm_best_mi))
        fh.write('#\n')
  
        for ipop in range(self.Npop):
  
            popstring = self.popkeys  [ipop]
            age_best_mi  = self.age_best_mi [ipop]
            met_best_mi  = self.met_best_mi [ipop]
            norm_best_mi = self.norm_best_mi[ipop]

            s  = '# population {:>4d}  , name = {:>30s}\n'.format(ipop+1, popstring)
            fh.write(s)
            s = '# log age = {:1.5f}  metallicitiy = {:1.5f}  norm = {:1.5e}\n'.format(age_best_mi, met_best_mi, self.norm_best_mi[ipop])
            fh.write(s)
          
            s = '# {:>8}  {:>11}  {:>11}  {:>11}  {:>11}  {:>11}\n'.format('logage' ,
                                                                           'Q(HI)'  ,
                                                                           'Q(HeI)' ,
                                                                           'Q(HeII)',
                                                                           'Lmech'  ,
                                                                           'Emech')
            fh.write(s)
            s = '# {:>8}  {:>11}  {:>11}  {:>11}  {:>11}  {:>11}\n'.format('yr'      ,
                                                                           'phot/sec',
                                                                           'phot/sec',
                                                                           'phot/sec',
                                                                           'erg'     ,
                                                                           'erg/sec')
            fh.write(s)
            s = '# {:>8}  {:>11}  {:>11}  {:>11}  {:>11}  {:>11}\n'.format('(1)',
                                                                           '(2)',
                                                                           '(3)',
                                                                           '(4)',
                                                                           '(5)',
                                                                           '(6)')
            fh.write(s)

            NageWrite   = 500 
            age_write_mi   = sp.linspace(4., age_best_mi, NageWrite)
            qhi_write_mi   = sp.zeros_like(age_write_mi)
            qhei_write_mi  = sp.zeros_like(age_write_mi)
            qheii_write_mi = sp.zeros_like(age_write_mi)
            lmech_write_mi = sp.zeros_like(age_write_mi)
            emech_write_mi = sp.zeros_like(age_write_mi)

            for iage in range(NageWrite): 
                d_mi = self.s.gen_sec_from_splines(met_best_mi, age_write_mi[iage])
                qhi_write_mi  [iage] = norm_best_mi*10.**d_mi[0]
                qhei_write_mi [iage] = norm_best_mi*10.**d_mi[1]
                qheii_write_mi[iage] = norm_best_mi*10.**d_mi[2]
                lmech_write_mi[iage] = norm_best_mi*10.**d_mi[3]
                emech_write_mi[iage] = norm_best_mi*10.**d_mi[4]
  
                s = '  {:1.4f}  {:1.5e}  {:1.5e}  {:1.5e}  {:1.5e}  {:1.5e}\n'.format(age_write_mi  [iage],
                                                                                      qhi_write_mi  [iage],
                                                                                      qhei_write_mi [iage],
                                                                                      qheii_write_mi[iage],
                                                                                      lmech_write_mi[iage],
                                                                                      emech_write_mi[iage])
                fh.write(s)

  
        fh.close()





    #MS classic monte carlo: storing the best fit values in a text file and dervied quantities in another text file
    def store_values(self, i, fnOut, fnOut2):
        parvals_mi = self.fitres_mi.params.valuesdict()
        self.popkeys   = [ a.replace('_a', '') for a in parvals_mi.keys() if a.endswith('_a') ]
        
        self.ebv_best_mi  = sp.zeros(self.Npop)        
        self.age_best_mi  = sp.zeros(self.Npop)
        self.met_best_mi  = sp.zeros(self.Npop)
        self.norm_best_mi = sp.zeros(self.Npop)

        self.qhi_best_mi   = sp.zeros(self.Npop)
        self.qhei_best_mi  = sp.zeros(self.Npop)
        self.qheii_best_mi = sp.zeros(self.Npop)
        self.lmech_best_mi = sp.zeros(self.Npop)
        self.emech_best_mi = sp.zeros(self.Npop)

        self.fesc_mi      = sp.zeros((self.Npop,self.o.Nlam_reb))
        self.flam_ind_mi  = sp.zeros((self.Npop,self.o.Nlam_reb))

        self.flux1270     = sp.zeros(self.Npop)
        
        for ipop in range(self.Npop):
            popstring = self.popkeys[ipop]    
            self.norm_best_mi[ipop] = self.fitres_mi.params[popstring+'_n'].value
            self.met_best_mi[ipop]  = self.fitres_mi.params[popstring+'_z'].value
            self.age_best_mi[ipop]  = self.fitres_mi.params[popstring+'_a'].value
            self.ebv_best_mi[ipop]  = self.fitres_mi.params[popstring+'_ebv'].value

            #MS deriving ionizing rates and energies wiht lmfit best values
            d_mi = self.s.gen_sec_from_splines(self.met_best_mi[ipop], self.age_best_mi[ipop])
            self.qhi_best_mi  [ipop] = self.norm_best_mi[ipop] * 10.**d_mi[0] 
            self.qhei_best_mi [ipop] = self.norm_best_mi[ipop] * 10.**d_mi[1] 
            self.qheii_best_mi[ipop] = self.norm_best_mi[ipop] * 10.**d_mi[2] 
            self.lmech_best_mi[ipop] = self.norm_best_mi[ipop] * 10.**d_mi[3] 
            self.emech_best_mi[ipop] = self.norm_best_mi[ipop] * 10.**d_mi[4]

            self.fesc_mi    [ipop] = 10.** (-0.4 * self.klam * self.ebv_best_mi[ipop])
            self.flam_ind_mi[ipop] = self.norm_best_mi[ipop] * 10.**self.s.gen_flam_from_splines(
            self.met_best_mi[ipop], self.age_best_mi[ipop], self.o.lamo_reb_log) * self.fesc_mi[ipop]

            lam1270 = 0
            while self.o.lamo_reb[lam1270] < 1270: 
                lam1270 += 1

            self.flux1270   [ipop] = self.flam_ind_mi[ipop][lam1270]

            self.fesc_mi    [ipop] = 10.** (-0.4 * self.klam * self.ebv_best_mi[ipop])
            self.flam_ind_mi[ipop] = self.norm_best_mi[ipop] * 10.**self.s.gen_flam_from_splines(
                                  self.met_best_mi[ipop], self.age_best_mi[ipop], self.o.lamo_reb_log) * self.fesc_mi[ipop]

        self.flam_best_mi = self.flam_ind_mi.sum(axis=0)
        self.chi2_best_mi  = sp.sum( (self.o.flam_reb[self.imask]-self.flam_best_mi[self.imask])**2./self.o.dflam_reb[self.imask]**2. )
        self.chi2n_best_mi = self.chi2_best_mi / (self.Ndp - 1)

        inds = sp.flip(self.flux1270.argsort())

        self.ebv_best_mi  = self.ebv_best_mi[inds]        
        self.age_best_mi  = self.age_best_mi[inds]  
        self.met_best_mi  = self.met_best_mi[inds]  
        self.norm_best_mi = self.norm_best_mi[inds]  

        self.qhi_best_mi   = self.qhi_best_mi[inds]  
        self.qhei_best_mi  = self.qhei_best_mi[inds]  
        self.qheii_best_mi = self.qheii_best_mi[inds]  
        self.lmech_best_mi = self.lmech_best_mi[inds]  
        self.emech_best_mi = self.emech_best_mi[inds]  

        
        if i == 0:
            fh = open(fnOut, 'w') 
            s = "# best fit values of the 100 fits of the classic monte carlo"
            s += "\n#{:>5}  ".format("N") 
            s += "{:>5}   ".format("chi2n") #self.chi2n_best_mi
            for ipop in range(self.Npop):
                s += "{:>8}  {:>8}  {:>11}  {:>8}  ".format(self.popkeys[ipop]+"_age", self.popkeys[ipop]+"_met", self.popkeys[ipop]+"_norm1", self.popkeys[ipop]+"_ebv1")
        else: 
            fh = open(fnOut, "a+")
            s = ""
        s += "\n"
        s += "{:6d}  ".format(i)
        s += "{:1.5e}   ".format(self.chi2n_best_mi)
        for ipop in range(self.Npop):
            s += "{:1.5f}  {:1.5f}  {:1.5e}  {:1.5f}  ".format(self.age_best_mi [ipop], self.met_best_mi [ipop], self.norm_best_mi[ipop], self.ebv_best_mi[ipop])           
        
        fh.write(s)
        fh.close()

        if i == 0:
            fh2 = open(fnOut2, 'w') 
            s = "# Derived-quantities values from the 100 fits of the classic monte carlo"
            s += "\n#{:>5}  ".format("N") 
            for ipop in range(self.Npop):
                s += "{:>8}  {:>8}  {:>11}  {:>8}  {:>8}  ".format(self.popkeys[ipop]+"_qhi", self.popkeys[ipop]+"_qhei", self.popkeys[ipop]+"_qheii", self.popkeys[ipop]+"_lmech", self.popkeys[ipop]+"_emech")
        else: 
            fh2 = open(fnOut2, "a+")
            s = ""
        s += "\n"
        s += "{:6d}  ".format(i)
        for ipop in range(self.Npop):
            s += "{:1.5e}  {:1.5e}  {:1.5e}  {:1.5e}  {:1.5e}  ".format(self.qhi_best_mi [ipop], self.qhei_best_mi [ipop], self.qheii_best_mi[ipop], self.lmech_best_mi[ipop], self.emech_best_mi[ipop])            
        
        fh2.write(s)
        fh2.close()

    #MS classic monte carlo: printing and saving the errors
    def errors(self, n, fnIn, fnIn2, fnOut):       
        self.ageLog_med = sp.zeros(self.Npop)
        self.age_med    = sp.zeros(self.Npop)
        self.met_med    = sp.zeros(self.Npop)
        self.norm_med   = sp.zeros(self.Npop)
        self.ebv_med    = sp.zeros(self.Npop) 

        self.qhi_med    = sp.zeros(self.Npop)
        self.qhei_med   = sp.zeros(self.Npop)
        self.qheii_med  = sp.zeros(self.Npop)
        self.lmech_med  = sp.zeros(self.Npop)
        self.emech_med  = sp.zeros(self.Npop)

        self.ageLog_lo = sp.zeros(self.Npop)
        self.age_lo    = sp.zeros(self.Npop)
        self.met_lo    = sp.zeros(self.Npop)
        self.norm_lo   = sp.zeros(self.Npop)
        self.ebv_lo    = sp.zeros(self.Npop) 

        self.qhi_lo    = sp.zeros(self.Npop)
        self.qhei_lo   = sp.zeros(self.Npop)
        self.qheii_lo  = sp.zeros(self.Npop)
        self.lmech_lo  = sp.zeros(self.Npop)
        self.emech_lo  = sp.zeros(self.Npop)

        self.ageLog_hi = sp.zeros(self.Npop)
        self.age_hi    = sp.zeros(self.Npop)
        self.met_hi    = sp.zeros(self.Npop)
        self.norm_hi   = sp.zeros(self.Npop)
        self.ebv_hi    = sp.zeros(self.Npop) 

        self.qhi_hi    = sp.zeros(self.Npop)
        self.qhei_hi   = sp.zeros(self.Npop)
        self.qheii_hi  = sp.zeros(self.Npop)
        self.lmech_hi  = sp.zeros(self.Npop)
        self.emech_hi  = sp.zeros(self.Npop)
  
        chi2n    = sp.zeros(n)
        agesLog  = sp.zeros((self.Npop,n))
        mets     = sp.zeros((self.Npop,n))
        norms    = sp.zeros((self.Npop,n))
        ebvs     = sp.zeros((self.Npop,n))
        qhis     = sp.zeros((self.Npop,n))
        qheis    = sp.zeros((self.Npop,n))
        qheiis   = sp.zeros((self.Npop,n))
        lmechs   = sp.zeros((self.Npop,n))      
        emechs   = sp.zeros((self.Npop,n))
        
        chi2n = np.loadtxt(fnIn, unpack=True, usecols=[0])
        self.chi2n_best = chi2n.min()
        self.n_best = chi2n.argmin()

        for ipop in range(self.Npop):
            agesLog[ipop], mets[ipop], norms[ipop], ebvs[ipop] = np.loadtxt(fnIn, unpack=True, usecols=list(1+ 5**ipop + np.array([0,1,2,3])))
            qhis[ipop], qheis[ipop], qheiis[ipop], lmechs[ipop], emechs[ipop] = np.loadtxt(fnIn2, unpack=True, usecols=list(6**ipop + np.array([0,1,2,3,4])))
            
#        for i in range(n):
#            inds = ages[:,i].argsort()
#            ages[:,i] = ages[inds,i]
#            mets[:,i] = mets[inds,i]
#            norms[:,i] = norms[inds,i]
#            ebvs[:,i] = ebvs[inds,i]
#            qhis[:,i] = qhis[inds,i]
#            qheis[:,i] = qheis[inds,i]
#            qheiis[:,i] = qheiis[inds,i]
#            lmechs[:,i] = lmechs[inds,i]
#            emechs[:,i] = emechs[inds,i]
        
#MS first attempt of estimating ocnfidence intervals for derivable quantities... not quite the correct way! better to estimate std() as for the fit par.
#        self.qhi_lo     = sp.zeros(self.Npop)
#        self.qhi_hi     = sp.zeros(self.Npop)
#        self.qhei_lo    = sp.zeros(self.Npop)
#        self.qhei_hi    = sp.zeros(self.Npop)
#        self.qheii_lo   = sp.zeros(self.Npop)
#        self.qheii_hi   = sp.zeros(self.Npop)
#        self.lmech_lo   = sp.zeros(self.Npop)
#        self.lmech_hi   = sp.zeros(self.Npop)
#        self.emech_lo   = sp.zeros(self.Npop)
#        self.emech_hi   = sp.zeros(self.Npop)

        for ipop in range(self.Npop):
            self.ageLog_med[ipop], self.ageLog_lo[ipop], self.ageLog_hi[ipop] = np.percentile(agesLog[ipop,:], [50., 15.8655, 84.1345])
            self.age_med[ipop], self.age_lo[ipop], self.age_hi[ipop] = np.percentile(10**agesLog[ipop,:], [50., 15.8655, 84.1345])
            self.met_med[ipop], self.met_lo[ipop], self.met_hi[ipop] = np.percentile(mets[ipop,:], [50., 15.8655, 84.1345])
            self.norm_med[ipop], self.norm_lo[ipop], self.norm_hi[ipop] = np.percentile(norms[ipop,:], [50., 15.8655, 84.1345])
            self.ebv_med[ipop], self.ebv_lo[ipop], self.ebv_hi[ipop] = np.percentile(ebvs[ipop,:], [50., 15.8655, 84.1345])
            self.qhi_med[ipop], self.qhi_lo[ipop], self.qhi_hi[ipop] = np.percentile(qhis[ipop,:], [50., 15.8655, 84.1345])
            self.qhei_med[ipop], self.qhei_lo[ipop], self.qhei_hi[ipop] = np.percentile(qheis[ipop,:], [50., 15.8655, 84.1345])
            self.qheii_med[ipop], self.qheii_lo[ipop], self.qheii_hi[ipop] = np.percentile(qheiis[ipop,:], [50., 15.8655, 84.1345])
            self.lmech_med[ipop], self.lmech_lo[ipop], self.lmech_hi[ipop] = np.percentile(lmechs[ipop,:], [50., 15.8655, 84.1345])
            self.emech_med[ipop], self.emech_lo[ipop], self.emech_hi[ipop] = np.percentile(emechs[ipop,:], [50., 15.8655, 84.1345])

            #MS deriving confidence intervals of ionizing rates and energies wiht lmfit best values, min and max values of Z are bound to 0.001 - 0.04
#            if self.met_best_mi[ipop] - self.met_err[ipop] > 0:
#                d_lo = self.s.gen_sec_from_splines(self.met_best_mi[ipop] - self.met_err[ipop], self.age_best_mi[ipop] - self.age_err[ipop])
#            else:
#                d_lo = self.s.gen_sec_from_splines(0.001, self.age_best_mi[ipop] - self.age_err[ipop])
#            if self.met_best_mi[ipop] + self.met_err[ipop] > 0.04:
#                d_hi = self.s.gen_sec_from_splines(0.04, self.age_best_mi[ipop] + self.age_err[ipop])
#            else:
#               d_hi = self.s.gen_sec_from_splines(self.met_best_mi[ipop] + self.met_err[ipop], self.age_best_mi[ipop] + self.age_err[ipop])
        
#            self.qhi_lo  [ipop] = (self.norm_best_mi[ipop]-self.norm_err[ipop]) * 10.**d_lo[0] 
#            self.qhei_lo [ipop] = (self.norm_best_mi[ipop]-self.norm_err[ipop]) * 10.**d_lo[1] 
#            self.qheii_lo[ipop] = (self.norm_best_mi[ipop]-self.norm_err[ipop]) * 10.**d_lo[2] 
#            self.lmech_lo[ipop] = (self.norm_best_mi[ipop]-self.norm_err[ipop]) * 10.**d_lo[3] 
#            self.emech_lo[ipop] = (self.norm_best_mi[ipop]-self.norm_err[ipop]) * 10.**d_lo[4] 
        
#            self.qhi_hi  [ipop] = (self.norm_best_mi[ipop]+self.norm_err[ipop]) * 10.**d_hi[0] 
#            self.qhei_hi [ipop] = (self.norm_best_mi[ipop]+self.norm_err[ipop]) * 10.**d_hi[1] 
#            self.qheii_hi[ipop] = (self.norm_best_mi[ipop]+self.norm_err[ipop]) * 10.**d_hi[2] 
#            self.lmech_hi[ipop] = (self.norm_best_mi[ipop]+self.norm_err[ipop]) * 10.**d_hi[3] 
#            self.emech_hi[ipop] = (self.norm_best_mi[ipop]+self.norm_err[ipop]) * 10.**d_hi[4]        
            
        fh = open(fnOut, 'w')
        #s = "# errors of the best fit values, estimated with classic monte carlo\n"
        #for ipop in range(self.Npop):
        #    s += "{:>8}  {:>8}  {:>11}  {:>8}  ".format(self.popkeys[ipop]+"_age", self.popkeys[ipop]+"_met", self.popkeys[ipop]+"_norm1", self.popkeys[ipop]+"_ebv1")
        #s += "\n"
        #for ipop in range(self.Npop):
        #    s += "{:1.5f}  {:1.5f}  {:1.5e}  {:1.5f}  ".format(age_err[ipop], met_err[ipop], norm_err[ipop], ebv_err[ipop])
            
        #fh.write(s)

        #MS reporting also the values of the lmfit minimization, with their corresponfing error...
        s = []
        s.append('Results from lmfit minimization method, errors calculated with classic montecarlo:')

        s.append('  Stellar population ages, t, in logYears')
        for ipop in range(self.Npop):
            s.append('{:>20} : {:1.5e} [ {:1.5e} {:1.5e} ]'.format(
                     self.popkeys  [ipop], self.age_best_mi [ipop], self.ageLog_lo[ipop], self.ageLog_hi[ipop]))
        
        s.append('  Stellar population ages, t, in MegaYears, converted from log before std calculation:')
        for ipop in range(self.Npop):
            s.append('{:>20} : {:1.3f} [ {:1.3f} {:1.3f} ]'.format(
                     self.popkeys  [ipop], 10**self.age_best_mi[ipop]/10**6, self.age_lo[ipop]/10**6, self.age_hi[ipop]/10**6))

        s.append('  Stellar population metallicities, Z, in X/H')
        for ipop in range(self.Npop):
            s.append('{:>20} : {:1.5e} [ {:1.5e} {:1.5e} ]'.format(
                     self.popkeys  [ipop], self.met_best_mi [ipop], self.met_lo[ipop], self.met_hi[ipop]))

        s.append('  Stellar population normalizations, n, given in units of input spectra')
        for ipop in range(self.Npop):
            s.append('{:>20} : {:1.5e} [ {:1.5e} {:1.5e} ]'.format(
                     self.popkeys  [ipop], self.norm_best_mi[ipop], self.norm_lo[ipop], self.norm_hi[ipop]))

        s.append('  Dust reddening, given as E(B-V), in magnitudes.')
        for ipop in range(self.Npop):
            s.append('{:>20} : {:1.5e} [ {:1.5e} {:1.5e} ]'.format(
                     self.popkeys  [ipop], self.ebv_best_mi [ipop], self.ebv_lo[ipop], self.ebv_hi[ipop]))

#        s.append('Errors for the following quantities not properly propagated (worst case scenario considered):')
#        s.append('min and max values of Z are bound to 0.001 - 0.04')
#        s.append('\ncheck if confidence intervals make sense with i=100...\n')

#        s.append('  H I ionizing photon rate, given in sec^-1')
#        for ipop in range(self.Npop):
#            s.append('{:>20} : {:1.5e} [ {:1.5e} {:1.5e} ]'.format(
#                     self.popkeys  [ipop], self.qhi_best_mi [ipop], self.qhi_lo [ipop], self.qhi_hi [ipop]))

#        s.append('  He I ionizing photon rate, given in sec^-1')
#        for ipop in range(self.Npop):
#            s.append('{:>20} : {:1.5e} [ {:1.5e} {:1.5e} ]'.format(
#                     self.popkeys  [ipop], self.qhei_best_mi[ipop], self.qhei_lo[ipop], self.qhei_hi[ipop]))

#        s.append('  He II ionizing photon rate, given in sec^-1')
#        for ipop in range(self.Npop):
#            s.append('{:>20} : {:1.5e} [ {:1.5e} {:1.5e} ]'.format(
#                     self.popkeys  [ipop], self.qheii_best_mi[ipop], self.qheii_lo[ipop], self.qheii_hi[ipop]))

#        s.append('  Integrated mechanical energy since t=0, given in erg')
#        for ipop in range(self.Npop):
#            s.append('{:>20} : {:1.5e} [ {:1.5e} {:1.5e} ]'.format(
#                     self.popkeys  [ipop], self.emech_best_mi[ipop], self.emech_lo[ipop], self.emech_hi[ipop]))

#        s.append('  Instantaneous mechanical power, given in erg sec^-1')
#        for ipop in range(self.Npop):
#            s.append('{:>20} : {:1.5e} [ {:1.5e} {:1.5e} ]'.format(
#                     self.popkeys  [ipop], self.lmech_best_mi[ipop], self.lmech_lo[ipop], self.lmech_hi[ipop]))


        s.append('Quantities that can be derived from fit parameters...')
        s.append('  H I ionizing photon rate, given in sec^-1')
        for ipop in range(self.Npop):
            s.append('{:>20} : {:1.5e} [ {:1.5e} {:1.5e} ]'.format(
                     self.popkeys  [ipop], self.qhi_best_mi [ipop], self.qhi_lo [ipop], self.qhi_hi [ipop]))

        s.append('  He I ionizing photon rate, given in sec^-1')
        for ipop in range(self.Npop):
            s.append('{:>20} : {:1.5e} [ {:1.5e} {:1.5e} ]'.format(
                     self.popkeys  [ipop], self.qhei_best_mi[ipop], self.qhei_lo[ipop], self.qhei_hi[ipop]))

        s.append('  He II ionizing photon rate, given in sec^-1')
        for ipop in range(self.Npop):
            s.append('{:>20} : {:1.5e} [ {:1.5e} {:1.5e} ]'.format(
                     self.popkeys  [ipop], self.qheii_best_mi[ipop], self.qheii_lo[ipop], self.qheii_hi[ipop]))

        s.append('  Integrated mechanical energy since t=0, given in erg')
        for ipop in range(self.Npop):
            s.append('{:>20} : {:1.5e} [ {:1.5e} {:1.5e} ]'.format(
                     self.popkeys  [ipop], self.emech_best_mi[ipop], self.emech_lo[ipop], self.emech_hi[ipop]))

        s.append('  Instantaneous mechanical power, given in erg sec^-1')
        for ipop in range(self.Npop):
            s.append('{:>20} : {:1.5e} [ {:1.5e} {:1.5e} ]'.format(
                     self.popkeys  [ipop], self.lmech_best_mi[ipop], self.lmech_lo[ipop], self.lmech_hi[ipop]))


        s.append('    chi2 : {:1.5e}'.format(self.chi2_best_mi))
        s.append('   chi2n : {:1.5e} for {:d} datapoints (N params is wrong here!)'.format(self.chi2n_best_mi, self.Ndp))
        s.append('')

        for l in s:
            print(l)
            fh.write('#' + l + '\n')
        
        fh.close()





















