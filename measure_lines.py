import sys
from os.path import join, exists

import numpy as np
import scipy as sp

import matplotlib.pyplot as plt
import lmfit
from astropy import constants
import pyneb as pn
from pyneb.utils.misc import parseAtom

from filehandle import getlines
from extin import CCM


ckms = constants.c.value / 1.e3
fwhm_g = 250. 
fwhmfit = 1500.
minimize_method = 'leastsq'
Nmc = 1000

def med_filter(dd, Nfilt): 
    Npix  = len(dd)
    hfilt = int(Nfilt/2)

    ddf = sp.zeros_like(dd)
    
    for ii in range(len(dd)): 
        ilo = ii-hfilt
        ihi = ii+hfilt
        if ilo<0    : ilo = 0
        if ihi>Npix : ihi = Npix
        ddf [ii] = sp.median(dd[ilo:ihi])
        
    return ddf 

def gauss(x, sigma, mu):
    g = 1./sigma/sp.sqrt(2*sp.pi) * sp.e**(-(x-mu)**2./2./sigma**2)
    return g

def add_line(wavel, lamc_rest, reds, fwhm_kms, flux): 
    
    lamc_obs = lamc_rest*(1+reds)
    sigma_aa = fwhm_kms / ckms * lamc_obs / ( 2.*sp.sqrt(2.*sp.log(2)) )
    ff = gauss(wavel, sigma_aa, lamc_obs) * flux
    return ff
      
def gen_linespec(wavel, pars):
    reds = pars[ "reds" ]
    fwhm = pars[ "fwhm" ]

    linespec = sp.zeros_like(wavel)
    for line in neblines.keys():
        linespec += add_line(wavel, neblines[line][0], reds, fwhm, pars[line])

    return linespec

def residual(pars, wavel, ff, dff, mask10):

    model  = gen_linespec(wavel, pars)
    imtf = mask10 == 1
    if 'WEIGHTED' in sys.argv: 
        return ((ff[imtf]-model[imtf]) / dff[imtf])
    else: 
        return (ff[imtf]-model[imtf]) 



#path2files = "/disk/sopdu_1/matthew/Science/LowzWinds/Sample/11727_Heckman/SDSSJ0055m0021/"
path2files = sys.argv[1]

tox = [ a.replace("\n", "").split() for a in getlines(join(path2files, 'setup_ssp.pars')) if not a.startswith("#") and len(a.strip()) > 5 ]
for toc in tox: 
    if   toc[0] == 'RUNNAME'  : runname = toc[1] 
    elif toc[0] == 'REDSHIFT' : reds_g = float(toc[1])        
      
galname = runname.split('_')[0]
print ('galaxy name:', galname)
   
if 'SSP1' in sys.argv: 
    fns = join(path2files, runname+'_fitres1_cs.txt')
elif 'SSP2' in sys.argv: 
    fns = join(path2files, runname+'_fitres2_cs.txt')
elif 'CSF1' in sys.argv: 
    fns = join(path2files, runname+'_fitres1_cs.txt').replace("ssp","csf")
elif 'CSF2' in sys.argv: 
    fns = join(path2files, runname+'_fitres2_cs.txt').replace("ssp","csf")
else: 
    fns = join(path2files, runname+'_fitres2_cs.txt')
    
print ('csfile:', fns, 'exists? --->', exists(fns))
print ('guess redshift', reds_g)

fnOutLines = join(path2files, runname.split('_')[0]+'_neb_fluxes.txt')
fnOutNebProps = join(path2files, runname.split('_')[0]+'_neb_properties.txt')

print ('writing output to:')
print (fnOutLines)
print (fnOutNebProps)



d = sp.loadtxt(fns)

itake = (3500 < d[:,0]) & (d[:,0]<9500)
lam   = d[itake,0]   # replace these with the format of your file
flam  = d[itake,1]
dflam = d[itake,3]

for iarg, arg in enumerate(sys.argv): 
    if arg == 'MEDIANFILTER': 
        kernel = int(sys.argv[iarg+1])
        print ('median filtering, with kernel of', kernel, '...')
        flam = med_filter(flam, kernel)

    elif arg == 'MINIMIZEMETHOD': 
        minimize_method = sys.argv[iarg+1]

    elif arg == 'FWHMFIT': 
        fwhmfit = float(sys.argv[iarg+1])

print ('minimization method:', minimize_method)
print ('FWHM for fit:', fwhmfit)
#plt.figure()
#plt.plot(lam,flam)
#plt.plot(lam,dflam)
#plt.axhline(0, color='k', linestyle='-')


### line dictionary 
### wavelengths (element 0) must be precise, and all be in the same vacuum/air system as the spectrum
### element 1 is guess value with respect to Ha.  

# hydrogen
neblines = {'HI_6563' : [ 6564.61, 1] }
neblines['HI_4861']    = [ 4862.683 , 0.3  ] 
neblines['HI_4341']    = [ 4341.684 , 0.1  ]
neblines['HI_4102']    = [ 4102.892 , 0.05 ]

# oxygen
neblines['OIII_4363']  = [ 4364.436 , 0.005] 
neblines['OIII_5007']  = [ 5008.240 , 1.   ] 
neblines['OIII_4959']  = [ 4960.295 , 0.3  ] 

neblines['OII_3726']   = [ 3727.092 , 0.5  ] 
neblines['OII_3729']   = [ 3729.875 , 0.5  ] 

neblines['OI_6300']    = [ 6302.046 , 0.05 ] 

# sulfur
neblines['SII_6717']   = [ 6718.29  , 0.2  ] 
neblines['SII_6731']   = [ 6732.67  , 0.2  ] 

# nitrogen
neblines['NII_6548']   = [ 6549.85  , 0.03 ] 
neblines['NII_6584']   = [ 6585.28  , 0.1  ] 
neblines['NII_5755']   = [ 5756.240 , 0.001] 
     
# helium 
neblines['HeI_5876']   = [ 5877.24  , 0.05] 
neblines['HeII_4686']  = [ 4687.02  , 0.01]

# neon
neblines['NeIII_3869'] = [ 3870.16 , 0.02] 


# do a numerical integral to guess the H-alpha flux
iha_g  = ( ((1.+reds_g)*neblines['HI_6563'][0]-12) < lam ) & (lam < ((1.+reds_g)*neblines['HI_6563'][0]+12))
fha_g  = sp.trapz(flam[iha_g], lam[iha_g])
print ("Initially guessed H-alpha flux to be:", fha_g,)



# set up the parameter object for fitting.  Includes redshift, FWHM in kms, and a flux for each line entry
lmpars = lmfit.Parameters()
lmpars.add("reds", vary=True, value=reds_g, min=reds_g*0.97, max=reds_g*1.03) # add redshift; tolerate 3% change
lmpars.add("fwhm", vary=True, value=fwhm_g, min=50, max=500) # add FWHM; toleratre min=spec.res; max = 500 km/s
for nebline in neblines.keys(): 
    print ("adding lines", nebline, "at lam_rest=", neblines[nebline][0], "guessing flux", neblines[nebline][1], "of Ha")
    lmpars.add(nebline, vary=True, value=fha_g*neblines[nebline][1], min=-10, max=fha_g*2.) 
    
lmpars.pretty_print()  

# generate the mask from the *initial guess* so it is constant over all instances of the fit.
mask10 = sp.zeros_like(lam)
guess = gen_linespec(lam, lmpars)
for line in neblines.keys():
    lc    = neblines[line][0] * (1.+reds_g)
    lamlo = lc - fwhmfit/ckms * lc
    lamhi = lc + fwhmfit/ckms * lc
    ilam  = (lamlo < lam) & (lam < lamhi)
    mask10[ilam] = 1.
    
#plt.figure()
#plt.plot(lam,guess)
#plt.plot(lam,mask10)    
#plt.title("guess spectrum and windows")
    
out = lmfit.minimize(residual, lmpars, method=minimize_method, args=(lam, flam, dflam, mask10))
out.params.pretty_print()
fitgas = gen_linespec(lam, out.params)

zbest = out.params['reds'].value

plt.figure()
plt.plot(lam,flam, label="data-cont")
plt.plot(lam,fitgas,label="fit.gas")
plt.plot(lam, mask10,label="mask.", c="pink")
for nebline in neblines.keys(): 
    plt.axvline( neblines[nebline][0]*(1.+zbest), ls=":", c="k")
    plt.text(neblines[nebline][0]*(1.+zbest), plt.ylim()[1]*0.7  , nebline, rotation=90)
plt.legend(loc=1)

#plt.figure()
#plt.plot(lam,(flam-fitgas)/dflam, label="residual")

#if out.params['reds'].stderr == 0.: 
#    print ("errors reported as zero. running a MC simulation")
#
#    fluxmc = { }
#    for var in out.var_names: 
#        fluxmc[var] = sp.zeros(30, dtype=sp.float64)
#
#    for imc in range(30):
#        flam_mc = flam + sp.randn(len(flam))*dflam*300.
#        out_mc = lmfit.minimize(residual, lmpars, method=minimize_method, args=(lam, flam_mc, dflam, mask10))
#
#        for var in out_mc.var_names: 
#            fluxmc[var][imc] = out_mc.params[var].value
#
#        print (imc, fluxmc['fwhm'][imc])
#
#    for var in out.var_names: 
#        out.params[var].stderr = fluxmc[var].std()


print ("***Fit Results***")
p = out.params['reds']
print ("{:>20} : {:9.7f} +/- {:9.7f}".format("redshift", out.params['reds'].value, out.params['reds'].stderr))
print ("{:>20} : {:7.3f} +/- {:5.3f}".format("FWHM [km/s]", out.params['fwhm'].value, out.params['fwhm'].stderr))
for nebline in neblines.keys(): 
    print ("{:>20} : {:6.3f} +/- {:5.3f}".format("flux "+nebline, out.params[nebline].value, out.params[nebline].stderr ))

linevars = [ a for a in out.var_names if not a == 'fwhm' and not a == 'reds' ] 
linevars.sort()

fh = open(fnOutLines, 'w')

s1 = "#{:>19s}  {:>10s}  {:>10s}  {:>10s}  {:>10s}".format('galname', 'z', 'dz', 'fwhm', 'dfwhm' )
s2 = "#{:>19s}  {:>10s}  {:>10s}  {:>10s}  {:>10s}".format('', '',  '', 'km/s', 'km/s')
s3 = "#{:>19s}  {:>10s}  {:>10s}  {:>10s}  {:>10s}".format('(1)', '(2)', '(3)', '(4)', '(5)')
for iline, line in enumerate(linevars):
    s1 += "  {:>10s}  {:>10s}".format("f"+line.split("_")[0], "df"+line.split("_")[1])
    s2 += "  {:>10s}  {:>10s}".format("e-15cgs", "e-15cgs")
    s3 += "  {:>10s}  {:>10s}".format("("+str(iline*2+6)+")", "("+str(iline*2+7)+")" )
fh.write(s1+'\n')
fh.write(s2+'\n')
fh.write(s3+'\n')


s1 = "{:>20s}  {:>10.6f}  {:>10.4g}  {:>10.6f}  {:>10.6f}".format(
         galname, \
         out.params['reds'].value, \
         out.params['reds'].stderr, \
         out.params['fwhm'].value, \
         out.params['fwhm'].stderr)

for line in linevars:
    s1 += "  {:>10.4g}  {:>10.4g}".format(out.params[line].value, out.params[line].stderr)
fh.write(s1+'\n')

fh.close()



def compute_neb_quants(ha, hb, sii_6717, sii_6731, oiii_4363, oiii_5007, oii_3726, nii_6584): 
    hahb  = ha / hb
    hahb0 = 2.86
    
    RC = pn.RedCorr(law='CCM89')
    RC.setCorr(hahb / hahb0, neblines['HI_6563'][0], neblines['HI_4861'][0])
    ebv = RC.E_BV

    #print ("ha/hb & ebv", hahb, ebv)

    diags = pn.Diagnostics()
    diags.addDiag([ '[OIII] 4363/5007', '[SII] 6731/6716' ])

    hac        = ha        * RC.getCorr(neblines['HI_6563'  ][0]) 
    hbc        = hb        * RC.getCorr(neblines['HI_4861'  ][0]) 

    sii_6717c  = sii_6717  * RC.getCorr(neblines['SII_6717' ][0]) 
    sii_6731c  = sii_6731  * RC.getCorr(neblines['SII_6731' ][0])
    oiii_4363c = oiii_4363 * RC.getCorr(neblines['OIII_4363'][0])
    oiii_5007c = oiii_5007 * RC.getCorr(neblines['OIII_5007'][0])
    oii_3726c  = oii_3726  * RC.getCorr(neblines['OII_3726' ][0])
    nii_6584c  = nii_6584  * RC.getCorr(neblines['NII_6584' ][0])

    obs   = pn.Observation(corrected=True)
    obs.addLine(pn.EmissionLine('S', 2, 6716.0, obsIntens=sii_6717c ))
    obs.addLine(pn.EmissionLine('S', 2, 6731.0, obsIntens=sii_6731c ))
    obs.addLine(pn.EmissionLine('O', 3, 4363.0, obsIntens=oiii_4363c))
    obs.addLine(pn.EmissionLine('O', 3, 5007.0, obsIntens=oiii_5007c))

    Te, Ne = diags.getCrossTemDen('[OIII] 4363/5007', '[SII] 6731/6716', obs=obs)
    
    O3 = pn.Atom('O', 3)
    O2 = pn.Atom('O', 2)
    abundOiii_Te = O3.getIonAbundance(oiii_5007c, tem=Te, den=Ne, wave=5007, Hbeta=hbc)
    abundOii_Te  = O2.getIonAbundance(oii_3726c , tem=Te, den=Ne, wave=3726, Hbeta=hbc)
    
    n2 = sp.log10(nii_6584c/hac)
    abundOiii_PP04 = 9.37 + 2.03*n2 + 1.26*n2**2 + 0.32*n2**3

    #print ("ha/hb and ebv:", hahb, ebv)
    #print ("Te, ne:", Te, Ne)
    #print (" abund O++ Te:", 12+sp.log10(abundOiii_Te))
    #print ("  abund O+ Te:", 12+sp.log10(abundOii_Te))
    #print ("abund Otot Te:", 12+sp.log10(abundOii_Te+abundOiii_Te))
    #print (" abund O PP04:", abundOiii_PP04)

    return ebv, Te, Ne, 12+sp.log10(abundOii_Te+abundOiii_Te), abundOiii_PP04
#adict = diags.atomDict
#
#for atom in adict:
#    # Computes all the intensities of all the lines of all the ions considered
#    for line in pn.LINE_LABEL_LIST[atom]:
#        if line[-1] == 'm':
#            wavelength = float(line[:-1])*1e4
#        else:
#            wavelength = float(line[:-1])
#            elem, spec = parseAtom(atom)
#
##        print (line, wavelength, elem, spec)



fha        = out.params['HI_6563'  ].value
fhb        = out.params['HI_4861'  ].value
fsii_6717  = out.params['SII_6717' ].value
fsii_6731  = out.params['SII_6731' ].value
foiii_4363 = out.params['OIII_4363'].value
foiii_5007 = out.params['OIII_5007'].value
foii_3726  = out.params['OII_3726' ].value
fnii_6584  = out.params['NII_6584' ].value

ebv, te, ne, oh_te, oh_pp = compute_neb_quants(fha, fhb, fsii_6717, fsii_6731, foiii_4363, foiii_5007, foii_3726, fnii_6584)


mc_ha        = out.params['HI_6563'  ].value + sp.randn(Nmc) * out.params['HI_6563'  ].stderr
mc_hb        = out.params['HI_4861'  ].value + sp.randn(Nmc) * out.params['HI_4861'  ].stderr
mc_sii_6717  = out.params['SII_6717' ].value + sp.randn(Nmc) * out.params['SII_6717' ].stderr
mc_sii_6731  = out.params['SII_6731' ].value + sp.randn(Nmc) * out.params['SII_6731' ].stderr
mc_oiii_4363 = out.params['OIII_4363'].value + sp.randn(Nmc) * out.params['OIII_4363'].stderr
mc_oiii_5007 = out.params['OIII_5007'].value + sp.randn(Nmc) * out.params['OIII_5007'].stderr
mc_oii_3726  = out.params['OII_3726' ].value + sp.randn(Nmc) * out.params['OII_3726' ].stderr    
mc_nii_6584  = out.params['NII_6584' ].value + sp.randn(Nmc) * out.params['NII_6584' ].stderr
if 'HACKERROR' in sys.argv: 
    mc_ha        = out.params['HI_6563'  ].value + sp.randn(Nmc) * out.params['HI_6563'  ].value * 0.01
    mc_hb        = out.params['HI_4861'  ].value + sp.randn(Nmc) * out.params['HI_4861'  ].value * 0.01
    mc_sii_6717  = out.params['SII_6717' ].value + sp.randn(Nmc) * out.params['SII_6717' ].value * 0.07
    mc_sii_6731  = out.params['SII_6731' ].value + sp.randn(Nmc) * out.params['SII_6731' ].value * 0.07
    mc_oiii_4363 = out.params['OIII_4363'].value + sp.randn(Nmc) * out.params['OIII_4363'].value * 0.25
    mc_oiii_5007 = out.params['OIII_5007'].value + sp.randn(Nmc) * out.params['OIII_5007'].value * 0.01
    mc_oii_3726  = out.params['OII_3726' ].value + sp.randn(Nmc) * out.params['OII_3726' ].value * 0.03
    mc_nii_6584  = out.params['NII_6584' ].value + sp.randn(Nmc) * out.params['NII_6584' ].value * 0.10
    print ("***** WARNING ****** ERRORS WERE HACKED *****")

mc_ebv   = sp.zeros_like(mc_ha)
mc_te    = sp.zeros_like(mc_ha)
mc_ne    = sp.zeros_like(mc_ha)
mc_oh_te = sp.zeros_like(mc_ha)
mc_oh_pp = sp.zeros_like(mc_ha)

for imc in range(Nmc): 
    mc_ebv[imc], mc_te[imc], mc_ne[imc], mc_oh_te[imc], mc_oh_pp[imc] = \
        compute_neb_quants(mc_ha[imc], \
                           mc_hb[imc], \
                           mc_sii_6717[imc], \
                           mc_sii_6731[imc], \
                           mc_oiii_4363[imc], \
                           mc_oiii_5007[imc], \
                           mc_oii_3726[imc], \
                           mc_nii_6584[imc])


mc_ebv  [sp.isnan(mc_ebv  ) ] = -1. 
mc_ne   [sp.isnan(mc_ne   ) ] = -1. 
mc_te   [sp.isnan(mc_te   ) ] = -1. 
mc_oh_te[sp.isnan(mc_oh_te) ] = -1. 
mc_oh_pp[sp.isnan(mc_oh_pp) ] = -1. 

mc_ebv   = mc_ebv  [mc_ebv  >0]
mc_ne    = mc_ne   [mc_ne   >0]
mc_te    = mc_te   [mc_te   >0]
mc_oh_te = mc_oh_te[mc_oh_te>0] 
mc_oh_pp = mc_oh_pp[mc_oh_pp>0]

ebv_lo  , ebv_med  , ebv_hi   = sp.percentile(mc_ebv  , [15.8655, 50, 84.134 ])
te_lo   , te_med   , te_hi    = sp.percentile(mc_te   , [15.8655, 50, 84.134 ])
ne_lo   , ne_med   , ne_hi    = sp.percentile(mc_ne   , [15.8655, 50, 84.134 ])
oh_te_lo, oh_te_med, oh_te_hi = sp.percentile(mc_oh_te, [15.8655, 50, 84.134 ])
oh_pp_lo, oh_pp_med, oh_pp_hi = sp.percentile(mc_oh_pp, [15.8655, 50, 84.134 ])


fh = open(fnOutNebProps, 'w')

s1 = "#{:>19s}".format('galname' )
s2 = "#{:>19s}".format('')
s3 = "#{:>19s}".format('(1)')

s1 += "  {:>10s}  {:>10s}  {:>10s}  {:>10s}".format("ebv", "ebv_med", "ebv_lo", "ebv_hi")
s1 += "  {:>10s}  {:>10s}  {:>10s}  {:>10s}".format("te", "te_med", "te_lo", "te_hi")
s1 += "  {:>10s}  {:>10s}  {:>10s}  {:>10s}".format("ne", "ne_med", "ne_lo", "ne_hi")
s1 += "  {:>10s}  {:>10s}  {:>10s}  {:>10s}".format("oh_te", "oh_te_med", "oh_te_lo", "oh_te_hi")
s1 += "  {:>10s}  {:>10s}  {:>10s}  {:>10s}".format("oh_pp", "oh_pp_med", "oh_pp_lo", "oh_pp_hi")

s2 += "  {:>10s}  {:>10s}  {:>10s}  {:>10s}".format("mag", "mag", "mag", "mag")
s2 += "  {:>10s}  {:>10s}  {:>10s}  {:>10s}".format("K", "K", "K", "K")
s2 += "  {:>10s}  {:>10s}  {:>10s}  {:>10s}".format("cm-3", "cm-3", "cm-3", "cm-3")
s2 += "  {:>10s}  {:>10s}  {:>10s}  {:>10s}".format("12+O/H", "12+O/H", "12+O/H", "12+O/H")
s2 += "  {:>10s}  {:>10s}  {:>10s}  {:>10s}".format("12+O/H", "12+O/H", "12+O/H", "12+O/H")

for ii in range(4*5):
    s3 += "  {:>10s}".format("("+str(ii+2)+")" )
fh.write(s1+'\n')
fh.write(s2+'\n')
fh.write(s3+'\n')

s1 = "{:>20s}".format(galname)

s1 += "  {:>10.4f}  {:>10.4f}  {:>10.4f}  {:>10.4f}".format(ebv, ebv_med, ebv_lo, ebv_hi)
s1 += "  {:>10.4f}  {:>10.4f}  {:>10.4f}  {:>10.4f}".format(te, te_med, te_lo, te_hi)
s1 += "  {:>10.4f}  {:>10.4f}  {:>10.4f}  {:>10.4f}".format(ne, ne_med, ne_lo, ne_hi)
s1 += "  {:>10.4f}  {:>10.4f}  {:>10.4f}  {:>10.4f}".format(oh_te, oh_te_med, oh_te_lo, oh_te_hi)
s1 += "  {:>10.4f}  {:>10.4f}  {:>10.4f}  {:>10.4f}".format(oh_pp, oh_pp_med, oh_pp_lo, oh_pp_hi)

fh.write(s1+'\n')

fh.close()



fig = plt.figure(figsize=(10,10))
ax1 = fig.add_subplot(221)
ax2 = fig.add_subplot(222)
ax3 = fig.add_subplot(223)
ax4 = fig.add_subplot(224)

#ax1.hist(mc_ebv  , bins=sp.linspace(0, 1.5  , 30), normed=True)
#ax2.hist(mc_te   , bins=sp.linspace(0, 20000, 30), normed=True)
#ax3.hist(mc_ne   , bins=sp.linspace(0, 1000 , 30), normed=True)
#ax4.hist(mc_oh_te, bins=sp.linspace(6, 9    , 30), normed=True)
#ax4.hist(mc_oh_pp, bins=sp.linspace(6, 9    , 30), alpha=0.3, normed=True)

ax1.hist(mc_ebv  , bins=30, normed=True)
ax2.hist(mc_te   , bins=30, normed=True)
ax3.hist(mc_ne   , bins=30, normed=True)
ax4.hist(mc_oh_te, bins=30, normed=True)
ax4.hist(mc_oh_pp, bins=30, alpha=0.3, normed=True)

ax1.axvline(ebv  , c="k", ls="-")
ax2.axvline(te   , c="k", ls="-")
ax3.axvline(ne   , c="k", ls="-")
ax4.axvline(oh_te, c="k", ls="-")
ax4.axvline(oh_pp, c="r", ls="-")

ax1.axvline(ebv_med  , c="k", ls="--")
ax2.axvline(te_med   , c="k", ls="--")
ax3.axvline(ne_med   , c="k", ls="--")
ax4.axvline(oh_te_med, c="k", ls="--")
ax4.axvline(oh_pp_med, c="r", ls="--")

ax1.axvline(ebv_lo  , c="k", ls=":")
ax2.axvline(te_lo   , c="k", ls=":")
ax3.axvline(ne_lo   , c="k", ls=":")
ax4.axvline(oh_te_lo, c="k", ls=":")
ax4.axvline(oh_pp_lo, c="r", ls=":")

ax1.axvline(ebv_hi  , c="k", ls=":")
ax2.axvline(te_hi   , c="k", ls=":")
ax3.axvline(ne_hi   , c="k", ls=":")
ax4.axvline(oh_te_hi, c="k", ls=":")
ax4.axvline(oh_pp_hi, c="r", ls=":")

plt.show()
