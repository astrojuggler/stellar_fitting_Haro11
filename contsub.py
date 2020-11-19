import sys
import scipy as sp
from matplotlib import pyplot as plt 
from matplotlib.ticker import MultipleLocator
from filehandle import getlines 
from specfunc import resample_spline



if len(sys.argv)!=2: 
    print ("!!!!error.  usage:")
    print ("   ", sys.argv[0], " input.pars")
    sys.exit(1)


keys = [ a.split() for a in getlines(sys.argv[1]) if 
         ("_" in a) and not (a.strip().startswith("#")) ]

for key in keys: 
    if   key[0] == "FN_SSP_RES1"    : 
        d_ssp_fres1   = getlines(key[1])
        fn_ssp_fres1  = key[1]
        
    elif key[0] == "FN_SSP_RES2"    : 
        d_ssp_fres2   = getlines(key[1])
        fn_ssp_fres2  = key[1]
    elif key[0] == "FN_SSP_PRI"     : d_ssp_pri     = sp.loadtxt(key[1])
    elif key[0] == "FN_SSP_SEC"     : d_ssp_sec     = sp.loadtxt(key[1])
    elif key[0] == "FN_SSP_HST1"    : d_ssp_hst1    = sp.loadtxt(key[1])
    elif key[0] == "FN_SSP_HST2"    : d_ssp_hst2    = sp.loadtxt(key[1])
    elif key[0] == "FN_SSP_SECHIST" : d_ssp_sechist = getlines(key[1])
    elif key[0] == "FN_CSF_RES1"    : 
        d_csf_fres1   = getlines(key[1])
        fn_csf_fres1  = key[1]
    #elif key[0] == "FN_CSF_RES2"    : d_csf_fres2   = getlines(key[1])
    elif key[0] == "FN_CSF_PRI"     : d_csf_pri     = sp.loadtxt(key[1])
    elif key[0] == "FN_CSF_SEC"     : d_csf_sec     = sp.loadtxt(key[1])
    elif key[0] == "FN_CSF_HST1"    : d_csf_hst1    = sp.loadtxt(key[1])
    #elif key[0] == "FN_CSF_HST2"    : d_csf_hst2    = sp.loadtxt(key[1])
    elif key[0] == "FN_CSF_SECHIST" : d_csf_sechist = getlines(key[1])

    elif key[0] == "EBV_LO"   : ebv_lo   = float(key[1])
    elif key[0] == "EBV_HI"   : ebv_hi   = float(key[1])
    elif key[0] == "AGE_LO"   : age_lo   = float(key[1])
    elif key[0] == "AGE_HI"   : age_hi   = float(key[1])
    elif key[0] == "NORM_LO"  : norm_lo  = float(key[1])
    elif key[0] == "NORM_HI"  : norm_hi  = float(key[1])
    elif key[0] == "EMECH_LO" : emech_lo = float(key[1])
    elif key[0] == "EMECH_HI" : emech_hi = float(key[1])
    elif key[0] == "LMECH_LO" : lmech_lo = float(key[1])
    elif key[0] == "LMECH_HI" : lmech_hi = float(key[1])
    elif key[0] == "QHI_LO"   : qhi_lo   = float(key[1])
    elif key[0] == "QHI_HI"   : qhi_hi   = float(key[1])
    elif key[0] == "QHEI_LO"  : qhei_lo  = float(key[1])
    elif key[0] == "QHEI_HI"  : qhei_hi  = float(key[1])
    elif key[0] == "QHEII_LO" : qheii_lo = float(key[1])
    elif key[0] == "QHEII_HI" : qheii_hi = float(key[1])

    else: 
        print (key[0], "not an allowed keyword")
        sys.exit(1)



fnOutSSp1 = fn_ssp_fres1.replace(".txt", "_cs.txt")
fnOutSSp2 = fn_ssp_fres2.replace(".txt", "_cs.txt")
fnOutCsf1 = fn_csf_fres1.replace(".txt", "_cs.txt")
fhOutSSp1 = open(fnOutSSp1, "w")
fhOutSSp2 = open(fnOutSSp2, "w")
fhOutCsf1 = open(fnOutCsf1, "w")

sSsp1 = "#  {:>11s}  {:>12s}  {:>12s}  {:>12s}\n".format("lambda", "flam_cs", "flam_norm", "dflam_obs")
sSsp2 = "#  {:>11s}  {:>12s}  {:>12s}  {:>12s}\n".format("lambda", "flam_cs", "flam_norm", "dflam_obs")
sCsf1 = "#  {:>11s}  {:>12s}  {:>12s}  {:>12s}\n".format("lambda", "flam_cs", "flam_norm", "dflam_obs")
fhOutSSp1.write(sSsp1)
fhOutSSp2.write(sSsp2)
fhOutCsf1.write(sCsf1)
sSsp1 = "#  {:>11s}  {:>12s}  {:>12s}  {:>12s}\n".format("(1)", "(2)", "(3)", "(4)")
sSsp2 = "#  {:>11s}  {:>12s}  {:>12s}  {:>12s}\n".format("(1)", "(2)", "(3)", "(4)")
sCsf1 = "#  {:>11s}  {:>12s}  {:>12s}  {:>12s}\n".format("(1)", "(2)", "(3)", "(4)")
fhOutSSp1.write(sSsp1)
fhOutSSp2.write(sSsp2)
fhOutCsf1.write(sCsf1)

datSsp1 = sp.loadtxt(d_ssp_fres1)
datSsp2 = sp.loadtxt(d_ssp_fres2)
datCsf1 = sp.loadtxt(d_csf_fres1)

lamSsp1   = datSsp1[:,0] 
lamSsp2   = datSsp2[:,0] 
lamCsf1   = datCsf1[:,0] 
flamSsp1  = datSsp1[:,1] 
flamSsp2  = datSsp2[:,1] 
flamCsf1  = datCsf1[:,1] 
dflamSsp1 = datSsp1[:,2] 
dflamSsp2 = datSsp2[:,2] 
dflamCsf1 = datCsf1[:,2] 
mflamSsp1 = datSsp1[:,5] 
mflamSsp2 = datSsp2[:,6] 
mflamCsf1 = datCsf1[:,5] 

csflamSsp1 = flamSsp1 - mflamSsp1
csflamSsp2 = flamSsp2 - mflamSsp2
csflamCsf1 = flamCsf1 - mflamCsf1

nflamSsp1  = flamSsp1 / mflamSsp1
nflamSsp2  = flamSsp2 / mflamSsp2
nflamCsf1  = flamCsf1 / mflamCsf1

for ii in range(len(lamSsp1)):
    sSsp1 = "  {:12.5g}  {:12.5g}  {:12.5g}  {:12.5g}\n".format(lamSsp1[ii], csflamSsp1[ii], nflamSsp1[ii], dflamSsp1[ii])
    sSsp2 = "  {:12.5g}  {:12.5g}  {:12.5g}  {:12.5g}\n".format(lamSsp2[ii], csflamSsp2[ii], nflamSsp2[ii], dflamSsp2[ii])
    sCsf1 = "  {:12.5g}  {:12.5g}  {:12.5g}  {:12.5g}\n".format(lamCsf1[ii], csflamCsf1[ii], nflamCsf1[ii], dflamCsf1[ii])
        
    fhOutSSp1.write(sSsp1)
    fhOutSSp2.write(sSsp2)
    fhOutCsf1.write(sCsf1)

fhOutSSp1.close()
fhOutSSp2.close()
fhOutCsf1.close()


c_ssp_1   = "teal"
c_ssp_2   = "orange"
c_ssp_2pr = "orange"
c_ssp_2po = "maroon"
c_csf_1   = "purple"



fig = plt.figure(figsize=(10,10))
ax1 = fig.add_subplot(321, xlim=[1000,9300], ylim=[-5,20])
ax2 = fig.add_subplot(323, xlim=[1000,9300], ylim=[-5,20])
ax3 = fig.add_subplot(325, xlim=[1000,9300], ylim=[-5,20])
ax4 = fig.add_subplot(322, xlim=[1000,9300], ylim=[0,3])
ax5 = fig.add_subplot(324, xlim=[1000,9300], ylim=[0,3])
ax6 = fig.add_subplot(326, xlim=[1000,9300], ylim=[0,3])

ax1.plot(lamSsp1, flamSsp1)
ax1.plot(lamSsp1, mflamSsp1)
ax1.plot(lamSsp1, csflamSsp1)

ax2.plot(lamSsp2, flamSsp2)
ax2.plot(lamSsp2, mflamSsp2)
ax2.plot(lamSsp2, csflamSsp2)

ax3.plot(lamCsf1, flamCsf1)
ax3.plot(lamCsf1, mflamCsf1)
ax3.plot(lamCsf1, csflamCsf1)

ax4.plot(lamSsp1, nflamSsp1)

ax5.plot(lamSsp2, nflamSsp2)

ax6.plot(lamCsf1, nflamCsf1)

ax1.hlines(0, xmin=0, xmax=10000, linestyle="-", color="k")
ax2.hlines(0, xmin=0, xmax=10000, linestyle="-", color="k")
ax3.hlines(0, xmin=0, xmax=10000, linestyle="-", color="k")
ax4.hlines(1, xmin=0, xmax=10000, linestyle="-", color="k")
ax5.hlines(1, xmin=0, xmax=10000, linestyle="-", color="k")
ax6.hlines(1, xmin=0, xmax=10000, linestyle="-", color="k")
                                                           
#plt.show()



