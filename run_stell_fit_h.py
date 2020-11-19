import os
import sys
import time

import scipy as sp
import matplotlib.pyplot as plt

from parameters import params
from synspec import SynSpec
from obsspec import ObsSpec
from spec_lines import get_windows, mask_standard, mask_standard_h, mask_nonstandard, mask_lam_line
from popfit import StelPop, Fit
from plotpdf import write_mc_pdf #,plot_pops_1and2



#### ---------- run the setup, reading, spline fitting ---------- ####

## get the starting time
starttime = time.time()

## read the configuration file. 
pars = params(sys.argv)

## read and sort the spectral line windows
windows = get_windows(pars)
# accessed as follows:
#print (windows['ismHis'].species)
#print (windows['ismHis'].lamolo)

## read the synthetic spectral data.  
syn = SynSpec(pars)
syn.read_syn()
syn.redshift_syn()


## setup of observed data, now we know limits of synthetic input
#  read, resample the UV (if necessary), apply max SNR, and de-redshift
obs = ObsSpec(pars, lammin=10.**syn.lamo_min, lammax=10.**syn.lamo_max)
#obs.resample()
obs.resample_and_combine()
obs.impose_maxsnr()
obs.shift_to_restframe()
#obs.plot(frame="rest", wave="both")
#obs.plot(frame="obs", wave="both")

## continue setup of synthetic data, now we know limits of observed input
## convolve to instrument resolution (in place)
syn.conv2inst(lamcUv=(obs.lamc_uv130 + obs.lamc_uv160) / 2, lamcOp=obs.lamc_op)        

## merge the IFASPEC (UV) and HIRES (optical) spectra 
syn.merge()

## represent all the synthetic data as splines
syn.fit_splines()


#MS plotting models as a diagnostics
#fig, axs = plt.subplots(syn.Nmetal, sharex=True, figsize=(10,20))
#for imet in range(syn.Nmetal):
#        axs[imet].set_title("Z = %0.4f" %syn.metalvals[imet])
#        intercepts = sp.array([0.1, 0.2, 0.3, 0.4, 0.5])*10
#        ages = sp.log10([1e6,2e6,3e6,5e6,8e6])
#       	for j in range(len(ages)):
#                axs[imet].text(0, intercepts[j]+0.1, "age = %0.3f" %ages[j])
#                x = syn.lamo_uv_raw
#                y = 40000 * 10.**syn.gen_flam_from_splines(syn.metalvals[imet], ages[j], sp.log10(syn.lamo_uv_raw))
#                axs[i].plot(x, y)
#fig.savefig("models_subplots.pdf") 


##### Generate the fitting masks 
mask   = mask_standard_h(obs.lamo_reb, obs.lamr_reb, obs.dq_reb, windows) * \
				 mask_nonstandard(obs.lamo_reb, pars.manmask)



#### ---------- Do the fitting ---------- ####

#### One stellar population
fit1pop = Fit(pars, obs, syn, mask)
fit1pop.plot_obs(windows['ismLis'])

# generate the fitting object 
    #MS changing nmax so that the mass (M=10**6 * n) does not exceed 10**7 Msun
spops1 = [ StelPop ("onepop" , aval=6.000, afit=True , amin=6.0, amax=7.7, \
                     nfit=True, nmax=1000., \
                     zval=0.008, zfit=True, zmin=pars.metalLo, zmax=pars.metalHi) ]

# setup and pring the initial guesses
fit1pop.setup_npop_red(spops1, ebvval=0.1, ebvmin=pars.ebvLo, ebvmax=pars.ebvHi)  
fit1pop.lmpars.pretty_print()
fit1pop.guessnorm_npop_red(fit1pop.lmpars)
fit1pop.lmpars.pretty_print()

# do the fitting
fit1pop.fit_npop_red(minalg="differential_evolution")#, Nstep=pars.Nstep, Nwalker=pars.Nwalker, Nburn=pars.Nburn)  
fit1pop.show_fitres_npop_red()

# calculate the best spectra, write and plot the outputs
fit1pop.calc_bestspec_npop_red(fnOut=pars.fnRes1)
fit1pop.plot_npop_red(fnOut=pars.fnFigroot1)
fit1pop.write_energy_history(pars, fnOut=pars.fnSecHist1)



#### store the fit results from the MCMC calculation
#MS commenting out mcmc
#write_mc_pdf(fit1pop.fitres_mc, fit2pop.fitres_mc, syn, fnroot=pars.fnMcRes) 

#MS classic montecarlo for estimating errors of the best fit values
#This part of the code is run only if you provided the argument "errors"
if (len(sys.argv) == 3) and (sys.argv[2] == "errors"): 
    for i in range(100):
        #generate a mock spectrum
        obs_mock = ObsSpec(pars, lammin=10.**syn.lamo_min, lammax=10.**syn.lamo_max)
        obs_mock.vary()
        #obs_mock.resample()
        obs_mock.resample_and_combine()
        obs_mock.impose_maxsnr()
        obs_mock.shift_to_restframe()
        obs_mock.save_mock(i)
    
        #fit the mock spectrum with the same setup, method, initial conditions
        fit1pop_mock = Fit(pars, obs_mock, syn, mask)
        fit1pop_mock.setup_npop_red(spops1, ebvval=0.1, ebvmin=pars.ebvLo, ebvmax=pars.ebvHi)
        fit1pop_mock.guessnorm_npop_red(fit2pop_mock.lmpars)  
        fit1pop_mock.fit_npop_red(minalg="differential_evolution")#, Nstep=pars.Nstep, Nwalker=pars.Nwalker, Nburn=pars.Nburn) 

        #store best fit values in a text file 
        fit1pop_mock.store_values(i, fnOut=pars.fnClassic, fnOut2=pars.fnClassicDe)


    #MS save and print errors
    fit1pop.errors(n=100, fnIn=pars.fnClassic, fnIn2=pars.fnClassicDe, fnOut=pars.fnClassicErr)
    

    endtime = time.time()
    runtime = endtime-starttime
    print ("Ran job for file {:s} in {:f} seconds ~{:f} hours".format(
	pars.fnPars, runtime, runtime/3600.))


# gen debug tests 
"""
from mathfunc import ind_nearest
lam_test  = sp.arange(1100., 7000., 0.2)

f1 = plt.figure()
a = f1.add_subplot(111)
f2 = plt.figure(figsize=(14,4))
a1 = f2.add_subplot(151)
a2 = f2.add_subplot(152)
a3 = f2.add_subplot(153)
a4 = f2.add_subplot(154)
a5 = f2.add_subplot(155)

imet = 2
age_test  = 4.e6
iage_test = ind_nearest(syn.age_raw[imet], sp.log10(age_test))
print ('testing age',age_test,'at', iage_test, syn.age_raw[imet][iage_test], 'metallicity', syn.metalvals[imet])
a.plot(sp.log10(lam_test), syn.flam_spl[imet](sp.log10(lam_test), sp.log10(age_test)))
a.plot(syn.lamo_raw[imet], syn.flam_raw[imet][iage_test], label="%s - %f"%(syn.metalstrs[imet],syn.age_raw[imet][iage_test]))

age_test = sp.arange (6,8,0.01)
a1.plot(syn.age_qu_raw[imet], syn.qhi_raw     [imet], label="%s - %f"%(syn.metalstrs[imet],syn.age_raw[imet][iage_test]))
a2.plot(syn.age_qu_raw[imet], syn.qhei_raw    [imet])
a3.plot(syn.age_qu_raw[imet], syn.qheii_raw   [imet])
a4.plot(syn.age_em_raw[imet], syn.lmechTot_raw[imet])
a5.plot(syn.age_em_raw[imet], syn.emechTot_raw[imet])
a1.plot(age_test, syn.qhi_spl     [imet](age_test))
a2.plot(age_test, syn.qhei_spl    [imet](age_test))
a3.plot(age_test, syn.qheii_spl   [imet](age_test))
a4.plot(age_test, syn.lmechTot_spl[imet](age_test))
a5.plot(age_test, syn.emechTot_spl[imet](age_test))



imet = 3
age_test  = 2.e6
iage_test = ind_nearest(syn.age_raw[imet], sp.log10(age_test))
print ('testing age',age_test,'at', iage_test, syn.age_raw[imet][iage_test], 'metallicity', syn.metalvals[imet])
a.plot(sp.log10(lam_test), syn.flam_spl[imet](sp.log10(lam_test), sp.log10(age_test)))
a.plot(syn.lamo_raw[imet], syn.flam_raw[imet][iage_test], label="%s - %f"%(syn.metalstrs[imet],syn.age_raw[imet][iage_test]))

age_test = sp.arange (6,8,0.01)
a1.plot(syn.age_qu_raw[imet], syn.qhi_raw     [imet], label="%s - %f"%(syn.metalstrs[imet],syn.age_raw[imet][iage_test]))
a2.plot(syn.age_qu_raw[imet], syn.qhei_raw    [imet])
a3.plot(syn.age_qu_raw[imet], syn.qheii_raw   [imet])
a4.plot(syn.age_em_raw[imet], syn.lmechTot_raw[imet])
a5.plot(syn.age_em_raw[imet], syn.emechTot_raw[imet])
a1.plot(age_test, syn.qhi_spl     [imet](age_test))
a2.plot(age_test, syn.qhei_spl    [imet](age_test))
a3.plot(age_test, syn.qheii_spl   [imet](age_test))
a4.plot(age_test, syn.lmechTot_spl[imet](age_test))
a5.plot(age_test, syn.emechTot_spl[imet](age_test))

imet = 1
age_test  = 7.e6
iage_test = ind_nearest(syn.age_raw[imet], sp.log10(age_test))
print ('testing age',age_test,'at', iage_test, syn.age_raw[imet][iage_test], 'metallicity', syn.metalvals[imet])
a.plot(sp.log10(lam_test), syn.flam_spl[imet](sp.log10(lam_test), sp.log10(age_test)))
a.plot(syn.lamo_raw[imet], syn.flam_raw[imet][iage_test], label="%s - %f"%(syn.metalstrs[imet],syn.age_raw[imet][iage_test]))

age_test = sp.arange (6,8,0.01)
a1.plot(syn.age_qu_raw[imet], syn.qhi_raw     [imet], label="%s - %f"%(syn.metalstrs[imet],syn.age_raw[imet][iage_test]))
a2.plot(syn.age_qu_raw[imet], syn.qhei_raw    [imet])
a3.plot(syn.age_qu_raw[imet], syn.qheii_raw   [imet])
a4.plot(syn.age_em_raw[imet], syn.lmechTot_raw[imet])
a5.plot(syn.age_em_raw[imet], syn.emechTot_raw[imet])
a1.plot(age_test, syn.qhi_spl     [imet](age_test))
a2.plot(age_test, syn.qhei_spl    [imet](age_test))
a3.plot(age_test, syn.qheii_spl   [imet](age_test))
a4.plot(age_test, syn.lmechTot_spl[imet](age_test))
a5.plot(age_test, syn.emechTot_spl[imet](age_test))

a.legend()
a1.legend()

f1 = plt.figure()
a = f1.add_subplot(111)
imet = 2
age_test  = 4.e6
iage_test = ind_nearest(syn.age_raw[imet], sp.log10(age_test))

a.plot(sp.log10(lam_test), syn.flam_spl[imet  ](sp.log10(lam_test), sp.log10(4.4e6)), 'k-', lw=2.)
a.plot(sp.log10(lam_test), syn.flam_spl[imet+1](sp.log10(lam_test), sp.log10(4.4e6)), 'k-', lw=2.)
a.plot(sp.log10(lam_test), syn.gen_flam_from_splines(syn.metalvals[imet]*1. , sp.log10(4.4e6), sp.log10(lam_test)), 'm--')
a.plot(sp.log10(lam_test), syn.gen_flam_from_splines(syn.metalvals[imet]*1.3, sp.log10(4.4e6), sp.log10(lam_test)), 'r-')
a.plot(sp.log10(lam_test), syn.gen_flam_from_splines(syn.metalvals[imet]*1.8, sp.log10(4.4e6), sp.log10(lam_test)), 'g-')
a.plot(sp.log10(lam_test), syn.gen_flam_from_splines(syn.metalvals[imet]*2.4, sp.log10(4.4e6), sp.log10(lam_test)), 'b-')
a.legend()

"""
#plt.show()
