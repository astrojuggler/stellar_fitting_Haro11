import sys
import scipy as sp
from matplotlib import pyplot as plt 
from matplotlib.ticker import MultipleLocator
from filehandle import getlines 
from specfunc import resample_spline


ebv_lo   = 0.
ebv_hi   = 1.
age_lo   = 6.
age_hi   = 7.7
met_lo   = 0.001
met_hi   = 0.040
norm_lo  = 7
norm_hi  = 9.5
emech_lo = 55
emech_hi = 57.5
lmech_lo = 40
lmech_hi = 44
qhi_lo   = 51
qhi_hi   = 56
qhei_lo  = 51
qhei_hi  = 56
qheii_lo = 47
qheii_hi = 53

if len(sys.argv)!=2: 
    print ("!!!!error.  usage:")
    print ("   ", sys.argv[0], " input.pars")
    sys.exit(1)


keys = [ a.split() for a in getlines(sys.argv[1]) if 
         ("_" in a) and not (a.strip().startswith("#")) ]

for key in keys: 
    if   key[0] == "FN_SSP_RES1"    : d_ssp_fres1   = getlines(key[1])
    elif key[0] == "FN_SSP_RES2"    : d_ssp_fres2   = getlines(key[1])
    elif key[0] == "FN_SSP_PRI"     : d_ssp_pri     = sp.loadtxt(key[1])
    elif key[0] == "FN_SSP_SEC"     : d_ssp_sec     = sp.loadtxt(key[1])
    elif key[0] == "FN_SSP_HST1"    : d_ssp_hst1    = sp.loadtxt(key[1])
    elif key[0] == "FN_SSP_HST2"    : d_ssp_hst2    = sp.loadtxt(key[1])
    elif key[0] == "FN_SSP_SECHIST" : d_ssp_sechist = getlines(key[1])
    elif key[0] == "FN_CSF_RES1"    : d_csf_fres1   = getlines(key[1])
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



def getposlog(s): 
    if float(s)<0:
        return -100.
    else: 
        return sp.log10(float(s))

def get_inrange(arr, lo, hi): 
    itake = (lo<arr) & (arr<hi)
    return arr[itake]



# individual best fit quantities 
# 1 SSP
ssp_ebv1      = float(d_ssp_fres1[5].split()[1])
ssp_ebv1_lo   = float(d_ssp_fres1[5].split()[3])
ssp_ebv1_hi   = float(d_ssp_fres1[5].split()[4])
ssp_age1      = float(d_ssp_fres1[7].split()[3])
ssp_age1_lo   = float(d_ssp_fres1[7].split()[5])
ssp_age1_hi   = float(d_ssp_fres1[7].split()[6])
ssp_met1      = float(d_ssp_fres1[9].split()[3])
ssp_met1_lo   = float(d_ssp_fres1[9].split()[5])
ssp_met1_hi   = float(d_ssp_fres1[9].split()[6])
ssp_norm1     = getposlog(d_ssp_fres1[11].split()[3])+6
ssp_norm1_lo  = getposlog(d_ssp_fres1[11].split()[5])+6
ssp_norm1_hi  = getposlog(d_ssp_fres1[11].split()[6])+6
ssp_qhi1      = getposlog(d_ssp_fres1[13].split()[3])
ssp_qhi1_lo   = getposlog(d_ssp_fres1[13].split()[5])
ssp_qhi1_hi   = getposlog(d_ssp_fres1[13].split()[6])
ssp_qhei1     = getposlog(d_ssp_fres1[15].split()[3])
ssp_qhei1_lo  = getposlog(d_ssp_fres1[15].split()[5])
ssp_qhei1_hi  = getposlog(d_ssp_fres1[15].split()[6])
ssp_qheii1    = getposlog(d_ssp_fres1[17].split()[3])
ssp_qheii1_lo = getposlog(d_ssp_fres1[17].split()[5])
ssp_qheii1_hi = getposlog(d_ssp_fres1[17].split()[6])
ssp_emech1    = getposlog(d_ssp_fres1[19].split()[3])
ssp_emech1_lo = getposlog(d_ssp_fres1[19].split()[5])
ssp_emech1_hi = getposlog(d_ssp_fres1[19].split()[6])
ssp_lmech1    = getposlog(d_ssp_fres1[21].split()[3])
ssp_lmech1_lo = getposlog(d_ssp_fres1[21].split()[5])
ssp_lmech1_hi = getposlog(d_ssp_fres1[21].split()[6])

# 2 SSP
ssp_ebv2        = float(d_ssp_fres2[5].split()[1])
ssp_ebv2_lo     = float(d_ssp_fres2[5].split()[3])
ssp_ebv2_hi     = float(d_ssp_fres2[5].split()[4])
ssp_age2pr      = float(d_ssp_fres2[7].split()[3])
ssp_age2pr_lo   = float(d_ssp_fres2[7].split()[5])
ssp_age2pr_hi   = float(d_ssp_fres2[7].split()[6])
ssp_age2po      = float(d_ssp_fres2[8].split()[3])
ssp_age2po_lo   = float(d_ssp_fres2[8].split()[5])
ssp_age2po_hi   = float(d_ssp_fres2[8].split()[6])
ssp_met2pr      = float(d_ssp_fres2[10].split()[3])
ssp_met2pr_lo   = float(d_ssp_fres2[10].split()[5])
ssp_met2pr_hi   = float(d_ssp_fres2[10].split()[6])
ssp_met2po      = float(d_ssp_fres2[11].split()[3])
ssp_met2po_lo   = float(d_ssp_fres2[11].split()[5])
ssp_met2po_hi   = float(d_ssp_fres2[11].split()[6])
ssp_norm2pr     = getposlog(d_ssp_fres2[13].split()[3])+6
ssp_norm2pr_lo  = getposlog(d_ssp_fres2[13].split()[5])+6
ssp_norm2pr_hi  = getposlog(d_ssp_fres2[13].split()[6])+6
ssp_norm2po     = getposlog(d_ssp_fres2[14].split()[3])+6
ssp_norm2po_lo  = getposlog(d_ssp_fres2[14].split()[5])+6
ssp_norm2po_hi  = getposlog(d_ssp_fres2[14].split()[6])+6

ssp_qhi2pr      = getposlog(d_ssp_fres2[16].split()[3])
ssp_qhi2pr_lo   = getposlog(d_ssp_fres2[16].split()[5])
ssp_qhi2pr_hi   = getposlog(d_ssp_fres2[16].split()[6])
ssp_qhi2po      = getposlog(d_ssp_fres2[17].split()[3])
ssp_qhi2po_lo   = getposlog(d_ssp_fres2[17].split()[5])
ssp_qhi2po_hi   = getposlog(d_ssp_fres2[17].split()[6])
ssp_qhei2pr     = getposlog(d_ssp_fres2[19].split()[3])
ssp_qhei2pr_lo  = getposlog(d_ssp_fres2[19].split()[5])
ssp_qhei2pr_hi  = getposlog(d_ssp_fres2[19].split()[6])
ssp_qhei2po     = getposlog(d_ssp_fres2[20].split()[3])
ssp_qhei2po_lo  = getposlog(d_ssp_fres2[20].split()[5])
ssp_qhei2po_hi  = getposlog(d_ssp_fres2[20].split()[6])
ssp_qheii2pr    = getposlog(d_ssp_fres2[22].split()[3])
ssp_qheii2pr_lo = getposlog(d_ssp_fres2[22].split()[5])
ssp_qheii2pr_hi = getposlog(d_ssp_fres2[22].split()[6])
ssp_qheii2po    = getposlog(d_ssp_fres2[23].split()[3])
ssp_qheii2po_lo = getposlog(d_ssp_fres2[23].split()[5])
ssp_qheii2po_hi = getposlog(d_ssp_fres2[23].split()[6])
ssp_emech2pr    = getposlog(d_ssp_fres2[25].split()[3])
ssp_emech2pr_lo = getposlog(d_ssp_fres2[25].split()[5])
ssp_emech2pr_hi = getposlog(d_ssp_fres2[25].split()[6])
ssp_emech2po    = getposlog(d_ssp_fres2[26].split()[3])
ssp_emech2po_lo = getposlog(d_ssp_fres2[26].split()[5])
ssp_emech2po_hi = getposlog(d_ssp_fres2[26].split()[6])
ssp_lmech2pr    = getposlog(d_ssp_fres2[28].split()[3])
ssp_lmech2pr_lo = getposlog(d_ssp_fres2[28].split()[5])
ssp_lmech2pr_hi = getposlog(d_ssp_fres2[28].split()[6])
ssp_lmech2po    = getposlog(d_ssp_fres2[29].split()[3])
ssp_lmech2po_lo = getposlog(d_ssp_fres2[29].split()[5])
ssp_lmech2po_hi = getposlog(d_ssp_fres2[29].split()[6])

# individual best fit quantities 
# 1 CSF
csf_ebv1      = float(d_csf_fres1[5].split()[1])
csf_ebv1_lo   = float(d_csf_fres1[5].split()[3])
csf_ebv1_hi   = float(d_csf_fres1[5].split()[4])
csf_age1      = float(d_csf_fres1[7].split()[3])
csf_age1_lo   = float(d_csf_fres1[7].split()[5])
csf_age1_hi   = float(d_csf_fres1[7].split()[6])
csf_met1      = float(d_csf_fres1[9].split()[3])
csf_met1_lo   = float(d_csf_fres1[9].split()[5])
csf_met1_hi   = float(d_csf_fres1[9].split()[6])
csf_norm1     = sp.log10 ( float(d_csf_fres1[11].split()[3]) * 10.**csf_age1    )
csf_norm1_lo  = sp.log10 ( float(d_csf_fres1[11].split()[5]) * 10.**csf_age1_lo )
csf_norm1_hi  = sp.log10 ( float(d_csf_fres1[11].split()[6]) * 10.**csf_age1_hi )
csf_qhi1      = getposlog(d_csf_fres1[13].split()[3])
csf_qhi1_lo   = getposlog(d_csf_fres1[13].split()[5])
csf_qhi1_hi   = getposlog(d_csf_fres1[13].split()[6])
csf_qhei1     = getposlog(d_csf_fres1[15].split()[3])
csf_qhei1_lo  = getposlog(d_csf_fres1[15].split()[5])
csf_qhei1_hi  = getposlog(d_csf_fres1[15].split()[6])
csf_qheii1    = getposlog(d_csf_fres1[17].split()[3])
csf_qheii1_lo = getposlog(d_csf_fres1[17].split()[5])
csf_qheii1_hi = getposlog(d_csf_fres1[17].split()[6])
csf_emech1    = getposlog(d_csf_fres1[19].split()[3])
csf_emech1_lo = getposlog(d_csf_fres1[19].split()[5])
csf_emech1_hi = getposlog(d_csf_fres1[19].split()[6])
csf_lmech1    = getposlog(d_csf_fres1[21].split()[3])
csf_lmech1_lo = getposlog(d_csf_fres1[21].split()[5])
csf_lmech1_hi = getposlog(d_csf_fres1[21].split()[6])


# monte carlo distributions
ssp_age1_mc     = d_ssp_pri[:,1]
ssp_age2pr_mc   = d_ssp_pri[:,5]
ssp_age2po_mc   = d_ssp_pri[:,8]
csf_age1_mc     = d_csf_pri[:,1]

ssp_met1_mc     = d_ssp_pri[:,2]
ssp_met2pr_mc   = d_ssp_pri[:,6]
ssp_met2po_mc   = d_ssp_pri[:,9]
csf_met1_mc     = d_csf_pri[:,2]

ssp_norm1_mc    = sp.log10(d_ssp_pri[:,3]*1.e6)
ssp_norm2pr_mc  = sp.log10(d_ssp_pri[:,7]*1.e6)
ssp_norm2po_mc  = sp.log10(d_ssp_pri[:,10]*1.e6)
csf_norm1_mc    = sp.log10(d_csf_pri[:,3]*10.**csf_age1_mc)

ssp_ebv1_mc     = d_ssp_pri[:,4]
ssp_ebv2_mc     = d_ssp_pri[:,11]
csf_ebv1_mc     = d_csf_pri[:,4]

ssp_qhi1_mc     = sp.log10(d_ssp_sec[:,1] )
ssp_qhi2pr_mc   = sp.log10(d_ssp_sec[:,6] )
ssp_qhi2po_mc   = sp.log10(d_ssp_sec[:,11])
csf_qhi1_mc     = sp.log10(d_csf_sec[:,1] )

ssp_qhei1_mc    = sp.log10(d_ssp_sec[:,2] )
ssp_qhei2pr_mc  = sp.log10(d_ssp_sec[:,7] )
ssp_qhei2po_mc  = sp.log10(d_ssp_sec[:,12])
csf_qhei1_mc    = sp.log10(d_csf_sec[:,2] )

ssp_qheii1_mc   = sp.log10(d_ssp_sec[:,3] )
ssp_qheii2pr_mc = sp.log10(d_ssp_sec[:,8] )
ssp_qheii2po_mc = sp.log10(d_ssp_sec[:,13])
csf_qheii1_mc   = sp.log10(d_csf_sec[:,3] )

ssp_lmech1_mc   = sp.log10(d_ssp_sec[:,4] )
ssp_lmech2pr_mc = sp.log10(d_ssp_sec[:,9] )
ssp_lmech2po_mc = sp.log10(d_ssp_sec[:,14])
csf_lmech1_mc   = sp.log10(d_csf_sec[:,4] )

ssp_emech1_mc   = sp.log10(d_ssp_sec[:,5] )
ssp_emech2pr_mc = sp.log10(d_ssp_sec[:,10])
ssp_emech2po_mc = sp.log10(d_ssp_sec[:,15])
csf_emech1_mc   = sp.log10(d_csf_sec[:,5] )

ssp_t = 10.**d_ssp_hst1[:,0]
ssp_agehst1   = ssp_t/1.e6
ssp_qhihst1   = d_ssp_hst1[:,1]
ssp_qheihst1  = d_ssp_hst1[:,2]
ssp_qheiihst1 = d_ssp_hst1[:,3]
ssp_lmechhst1 = d_ssp_hst1[:,4]
ssp_emechhst1 = d_ssp_hst1[:,5]

csf_t = 10.**d_csf_hst1[:,0]
csf_agehst1   = csf_t/1.e6
csf_qhihst1   = d_csf_hst1[:,1]
csf_qheihst1  = d_csf_hst1[:,2]
csf_qheiihst1 = d_csf_hst1[:,3]
csf_lmechhst1 = d_csf_hst1[:,4]
csf_emechhst1 = d_csf_hst1[:,5]


ssp_nelem1 = sp.argwhere(sp.diff(d_ssp_hst2[:,0])<0.)[0][0] +1
ssp_nelem2 = len(d_ssp_hst2[:,0])-ssp_nelem1
ssp_t1 = 10.**d_ssp_hst2[:ssp_nelem1,0]
ssp_t2 = 10.**d_ssp_hst2[ssp_nelem1:,0]
#agehst2pr   = (age2prmc-age2prmc[-1])/1.e6
#agehst2po   = (age2pomc-age2pomc[-1])/1.e6
ssp_agehst2pr   = ssp_t1/1.e6
ssp_agehst2po   = ssp_t2/1.e6
ssp_qhihst2pr   = d_ssp_hst2[:ssp_nelem1,1]
ssp_qhihst2po   = d_ssp_hst2[ssp_nelem1:,1]
ssp_qheihst2pr  = d_ssp_hst2[:ssp_nelem1,2]
ssp_qheihst2po  = d_ssp_hst2[ssp_nelem1:,2]
ssp_qheiihst2pr = d_ssp_hst2[:ssp_nelem1,3]
ssp_qheiihst2po = d_ssp_hst2[ssp_nelem1:,3]
ssp_lmechhst2pr = d_ssp_hst2[:ssp_nelem1,4]
ssp_lmechhst2po = d_ssp_hst2[ssp_nelem1:,4]
ssp_emechhst2pr = d_ssp_hst2[:ssp_nelem1,5]
ssp_emechhst2po = d_ssp_hst2[ssp_nelem1:,5]




def bin_interp(q, bins):
    binsCnt = (bins[:-1] + bins[1:])/2.
    xp      = sp.linspace(bins[1], bins[-2], 1000)
    y, x    = sp.histogram(q, bins=bins)
    qlo,qhi = sp.percentile(q, [13.59,86.41])
    yp      = resample_spline(binsCnt,y,xp,kind="quadratic")
    yp[yp<0.] = 0.
    return xp,yp,qlo,qhi


def plot_dist(ax, q, qbest, bins, color="k", label="none", hist=False):

    if len(q)>0:
        binsCnt = (bins[:-1] + bins[1:])/2.
        y, x    = sp.histogram(q, bins=bins)
        xp      = sp.linspace(bins[1], bins[-2], 1000)
        yp      = resample_spline(binsCnt,y,xp,kind="quadratic")
        qlo,qhi = sp.percentile(q, [13.59,86.41])
        yp[yp<0.] = 0.

        if hist == True: 
            ax.hist(q, bins=bins, fc=color, alpha=0.5, label=label)

        ix  = (qlo<=xp) & (xp<=qhi)
        ax.plot(xp, yp, c=color, label=label)
        ax.fill_between(xp[ix], yp[ix], color=color, alpha=0.1)
 
        plt.axvline(qbest, c=color, ls="-")
        #plt.axvline(qlo  , c=color, ls=":")
        #plt.axvline(qhi  , c=color, ls=":")

        ax.legend()

        return xp,yp,qlo,qhi




c_ssp_1   = "teal"
c_ssp_2   = "orange"
c_ssp_2pr = "orange"
c_ssp_2po = "maroon"
c_csf_1   = "purple"


# E(B-V)
fig = plt.figure()
ax  = fig.add_subplot(111)
bins = sp.linspace(ebv_lo, ebv_hi, 50)

plot_dist(ax, get_inrange(ssp_ebv1_mc, ebv_lo, ebv_hi), ssp_ebv1, bins=bins, color=c_ssp_1, label="1pop")
plot_dist(ax, get_inrange(ssp_ebv2_mc, ebv_lo, ebv_hi), ssp_ebv2, bins=bins, color=c_ssp_2, label="2pop")
plot_dist(ax, get_inrange(csf_ebv1_mc, ebv_lo, ebv_hi), csf_ebv1, bins=bins, color=c_csf_1, label="1pop")

ax.set_xlabel("E(B-V)")
ax.set_ylabel("frequency")
ax.set_yticks([])



# AGE
fig = plt.figure()
ax  = fig.add_subplot(111)
bins = sp.linspace(age_lo, age_hi, 100)

plot_dist(ax, get_inrange(ssp_age1_mc  , age_lo, age_hi), ssp_age1  , bins=bins, color=c_ssp_1  , label="1 SSP1"     )
plot_dist(ax, get_inrange(ssp_age2pr_mc, age_lo, age_hi), ssp_age2pr, bins=bins, color=c_ssp_2pr, label="2 SSP2 pre" )
plot_dist(ax, get_inrange(ssp_age2po_mc, age_lo, age_hi), ssp_age2po, bins=bins, color=c_ssp_2po, label="2 SSP2 post")
plot_dist(ax, get_inrange(csf_age1_mc  , age_lo, age_hi), csf_age1  , bins=bins, color=c_csf_1  , label="1 CSF1"     )

ax.set_xlabel("log age (yr)")
ax.set_ylabel("frequency")
ax.set_yticks([])



# METALLICITY
fig = plt.figure()
ax  = fig.add_subplot(111)
bins = sp.linspace(met_lo, met_hi, 100)

plot_dist(ax, get_inrange(ssp_met1_mc  , met_lo, met_hi), ssp_met1  , bins=bins, color=c_ssp_1  , label="1 SSP1"     )
plot_dist(ax, get_inrange(ssp_met2pr_mc, met_lo, met_hi), ssp_met2pr, bins=bins, color=c_ssp_2pr, label="2 SSP2 pre" )
plot_dist(ax, get_inrange(ssp_met2po_mc, met_lo, met_hi), ssp_met2po, bins=bins, color=c_ssp_2po, label="2 SSP2 post")
plot_dist(ax, get_inrange(csf_met1_mc  , met_lo, met_hi), csf_met1  , bins=bins, color=c_csf_1  , label="1 CSF1"     )
ax.set_xlabel("metallicity")
ax.set_ylabel("frequency")
ax.set_yticks([])



# NORM
fig = plt.figure()
ax  = fig.add_subplot(111)
bins = sp.linspace(norm_lo, norm_hi, 100)

plot_dist(ax, get_inrange(ssp_norm1_mc  , norm_lo, norm_hi), ssp_norm1  , bins=bins, color=c_ssp_1  , label="1 SSP"     )
plot_dist(ax, get_inrange(ssp_norm2pr_mc, norm_lo, norm_hi), ssp_norm2pr, bins=bins, color=c_ssp_2pr, label="2 SSP pre" )
plot_dist(ax, get_inrange(ssp_norm2po_mc, norm_lo, norm_hi), ssp_norm2po, bins=bins, color=c_ssp_2po, label="2 SSP post")
plot_dist(ax, get_inrange(csf_norm1_mc  , norm_lo, norm_hi), csf_norm1  , bins=bins, color=c_csf_1  , label="1 CSF"     )

ax.set_xlabel("log mass (Msun)")
ax.set_ylabel("frequency")
ax.set_yticks([])



# EMECH
fig = plt.figure()
ax  = fig.add_subplot(111)
bins = sp.linspace(emech_lo, emech_hi, 100)

plot_dist(ax, get_inrange(ssp_emech1_mc  , emech_lo, emech_hi), ssp_emech1  , bins=bins, color=c_ssp_1  , label="1 SSP"      )
plot_dist(ax, get_inrange(ssp_emech2pr_mc, emech_lo, emech_hi), ssp_emech2pr, bins=bins, color=c_ssp_2pr, label="2 SSP pre"  )
plot_dist(ax, get_inrange(ssp_emech2po_mc, emech_lo, emech_hi), ssp_emech2po, bins=bins, color=c_ssp_2po, label="2 SSP post" )
plot_dist(ax, get_inrange(csf_emech1_mc  , emech_lo, emech_hi), csf_emech1  , bins=bins, color=c_csf_1  , label="1 CSF"      )

ax.set_xlabel("Emech (erg)")
ax.set_ylabel("frequency")
ax.set_yticks([])




# LMECH
fig = plt.figure()
ax  = fig.add_subplot(111)
bins = sp.linspace(lmech_lo, lmech_hi, 30)

plot_dist(ax, get_inrange(ssp_lmech1_mc  , lmech_lo, lmech_hi), ssp_lmech1  , bins=bins, color=c_ssp_1  , label="1 SSP"      )
plot_dist(ax, get_inrange(ssp_lmech2pr_mc, lmech_lo, lmech_hi), ssp_lmech2pr, bins=bins, color=c_ssp_2pr, label="2 SSP pre"  )
plot_dist(ax, get_inrange(ssp_lmech2po_mc, lmech_lo, lmech_hi), ssp_lmech2po, bins=bins, color=c_ssp_2po, label="2 SSP post" )
plot_dist(ax, get_inrange(csf_lmech1_mc  , lmech_lo, lmech_hi), csf_lmech1  , bins=bins, color=c_csf_1  , label="1 CSF"      )

ax.set_xlabel("Lmech (erg/sec)")
ax.set_ylabel("frequency")
ax.set_yticks([])



# Q(H I)
fig = plt.figure()
ax  = fig.add_subplot(111)
bins = sp.linspace(qhi_lo, qhi_hi, 30)

plot_dist(ax, get_inrange(ssp_qhi1_mc  , qhi_lo, qhi_hi), ssp_qhi1  , bins=bins, color=c_ssp_1  , label="1 SSP"      )
plot_dist(ax, get_inrange(ssp_qhi2pr_mc, qhi_lo, qhi_hi), ssp_qhi2pr, bins=bins, color=c_ssp_2pr, label="2 SSP pre"  )
plot_dist(ax, get_inrange(ssp_qhi2po_mc, qhi_lo, qhi_hi), ssp_qhi2po, bins=bins, color=c_ssp_2po, label="2 SSP post" )
plot_dist(ax, get_inrange(csf_qhi1_mc  , qhi_lo, qhi_hi), csf_qhi1  , bins=bins, color=c_csf_1  , label="1 CSF"      )

ax.set_xlabel("Q(H I) (photons/sec)")
ax.set_ylabel("frequency")
ax.set_yticks([])



# Q(He I)
fig = plt.figure()
ax  = fig.add_subplot(111)
bins = sp.linspace(qhei_lo, qhei_hi, 30)

plot_dist(ax, get_inrange(ssp_qhei1_mc  , qhei_lo, qhei_hi), ssp_qhei1  , bins=bins, color=c_ssp_1  , label="1 SSP"      )
plot_dist(ax, get_inrange(ssp_qhei2pr_mc, qhei_lo, qhei_hi), ssp_qhei2pr, bins=bins, color=c_ssp_2pr, label="2 SSP pre"  )
#plot_dist(ax, get_inrange(ssp_qhei2po_mc, qhei_lo, qhei_hi), ssp_qhei2po, bins=bins, color=c_ssp_2po, label="2 SSP post" )
plot_dist(ax, get_inrange(csf_qhei1_mc  , qhei_lo, qhei_hi), csf_qhei1  , bins=bins, color=c_csf_1  , label="1 CSF"      )

ax.set_xlabel("Q(He I) (photons/sec)")
ax.set_ylabel("frequency")
ax.set_yticks([])



# Q(He II)
fig = plt.figure()
ax  = fig.add_subplot(111)
bins = sp.linspace(qheii_lo, qheii_hi, 30)

plot_dist(ax, get_inrange(ssp_qheii1_mc  , qheii_lo, qheii_hi), ssp_qheii1  , bins=bins, color=c_ssp_1  , label="1 SSP"      )
plot_dist(ax, get_inrange(ssp_qheii2pr_mc, qheii_lo, qheii_hi), ssp_qheii2pr, bins=bins, color=c_ssp_2pr, label="2 SSP pre"  )
plot_dist(ax, get_inrange(ssp_qheii2po_mc, qheii_lo, qheii_hi), ssp_qheii2po, bins=bins, color=c_ssp_2po, label="2 SSP post" )
plot_dist(ax, get_inrange(csf_qheii1_mc  , qheii_lo, qheii_hi), csf_qheii1  , bins=bins, color=c_csf_1  , label="1 CSF"      )

ax.set_xlabel("Q(He II) (photons/sec)")
ax.set_ylabel("frequency")
ax.set_yticks([])











agemax = sp.concatenate( (ssp_agehst1, ssp_agehst2pr, ssp_agehst2po, csf_agehst1 ) ).max()
agemin = sp.concatenate( (ssp_agehst1, ssp_agehst2pr, ssp_agehst2po, csf_agehst1 ) ).min()
agevec = sp.linspace(agemin*1.0001, agemax*0.9999, 200)

ssp_agein1   = ssp_agehst1   - ssp_agehst1  [-1] + agemax
ssp_agein2pr = ssp_agehst2pr - ssp_agehst2pr[-1] + agemax
ssp_agein2po = ssp_agehst2po - ssp_agehst2po[-1] + agemax
csf_agein1   = csf_agehst1   - csf_agehst1  [-1] + agemax


# 1 SSP
if ssp_agein1[0] > 1: 
    ssp_agein1     = sp.concatenate( (sp.array([agemin, ssp_agein1  .min()*0.9999]), ssp_agein1  ) )
    ssp_qhiin1     = sp.concatenate( (sp.array([0,0]), ssp_qhihst1 ) )
    ssp_emechin1   = sp.concatenate( (sp.array([0,0]), ssp_emechhst1   ) )
    ssp_lmechin1   = sp.concatenate( (sp.array([0,0]), ssp_lmechhst1   ) )
else: 
    ssp_qhiin1     = ssp_qhihst1 
    ssp_emechin1   = ssp_emechhst1
    ssp_lmechin1   = ssp_lmechhst1

# 2 SSP pre
if ssp_agein2pr[0] > 1: 
    ssp_agein2pr   = sp.concatenate( (sp.array([agemin, ssp_agein2pr.min()*0.9999]), ssp_agein2pr) )
    ssp_qhiin2pr   = sp.concatenate( (sp.array([0,0]), ssp_qhihst2pr ) )
    ssp_emechin2pr = sp.concatenate( (sp.array([0,0]), ssp_emechhst2pr ) )
    ssp_lmechin2pr = sp.concatenate( (sp.array([0,0]), ssp_lmechhst2pr ) )
else: 
    ssp_qhiin2pr   = ssp_qhihst2pr 
    ssp_emechin2pr = ssp_emechhst2pr
    ssp_lmechin2pr = ssp_lmechhst2pr

# 2 SSP pre
if ssp_agein2po[0] > 1: 
    ssp_agein2po   = sp.concatenate( (sp.array([agemin, ssp_agein2po.min()*0.9999]), ssp_agein2po) )
    ssp_qhiin2po   = sp.concatenate( (sp.array([0,0]), ssp_qhihst2po ) )
    ssp_emechin2po = sp.concatenate( (sp.array([0,0]), ssp_emechhst2po ) )
    ssp_lmechin2po = sp.concatenate( (sp.array([0,0]), ssp_lmechhst2po ) )
else: 
    ssp_qhiin2po   = ssp_qhihst2po 
    ssp_emechin2po = ssp_emechhst2po
    ssp_lmechin2po = ssp_lmechhst2po

# 1 CSF
if csf_agein1[0] > 1: 
    csf_agein1     = sp.concatenate( (sp.array([agemin, csf_agein1  .min()*0.9999]), csf_agein1  ) )
    csf_qhiin1     = sp.concatenate( (sp.array([0,0]), csf_qhihst1 ) )
    csf_emechin1   = sp.concatenate( (sp.array([0,0]), csf_emechhst1   ) )
    csf_lmechin1   = sp.concatenate( (sp.array([0,0]), csf_lmechhst1   ) )
else: 
    csf_qhiin1     = csf_qhihst1 
    csf_emechin1   = csf_emechhst1
    csf_lmechin1   = csf_lmechhst1


ssp_qhivec1     = resample_spline(ssp_agein1  , ssp_qhiin1    , agevec, kind="slinear") 
ssp_qhivec2pr   = resample_spline(ssp_agein2pr, ssp_qhiin2pr  , agevec, kind="slinear") 
ssp_qhivec2po   = resample_spline(ssp_agein2po, ssp_qhiin2po  , agevec, kind="slinear") 
csf_qhivec1     = resample_spline(csf_agein1  , csf_qhiin1    , agevec, kind="slinear") 

ssp_emechvec1   = resample_spline(ssp_agein1  , ssp_emechin1  , agevec, kind="slinear") 
ssp_emechvec2pr = resample_spline(ssp_agein2pr, ssp_emechin2pr, agevec, kind="slinear") 
ssp_emechvec2po = resample_spline(ssp_agein2po, ssp_emechin2po, agevec, kind="slinear") 
csf_emechvec1   = resample_spline(csf_agein1  , csf_emechin1  , agevec, kind="slinear") 

ssp_lmechvec1   = resample_spline(ssp_agein1  , ssp_lmechin1  , agevec, kind="slinear") 
ssp_lmechvec2pr = resample_spline(ssp_agein2pr, ssp_lmechin2pr, agevec, kind="slinear") 
ssp_lmechvec2po = resample_spline(ssp_agein2po, ssp_lmechin2po, agevec, kind="slinear") 
csf_lmechvec1   = resample_spline(csf_agein1  , csf_lmechin1  , agevec, kind="slinear") 

ssp_qhivec2tot   = ssp_qhivec2pr   + ssp_qhivec2po
ssp_emechvec2tot = ssp_emechvec2pr + ssp_emechvec2po
ssp_lmechvec2tot = ssp_lmechvec2pr + ssp_lmechvec2po



cphotons = "teal"
ckinetic = "maroon"
fontsize = 14

#HISTORY
#########
fig = plt.figure(figsize=(10,4))

ax1l = fig.add_subplot(121)#, position=[0.05, 0.1, 0.4, 0.8])
ax2l = fig.add_subplot(122)#, position=[0.55, 0.1, 0.4, 0.8])

### left plot with integrated energy
ax1l.plot(agevec, sp.log10(ssp_qhivec1   ), c=c_ssp_1, ls="-", lw=3, label="1 SSP"    )       # c=cphotons, ls="-" , 
ax1l.plot(agevec, sp.log10(ssp_qhivec2pr ), c=c_ssp_2, ls=":", lw=1)#, label="2 SSP,pr" )     # c=cphotons, ls=":" , 
ax1l.plot(agevec, sp.log10(ssp_qhivec2po ), c=c_ssp_2, ls=":", lw=1)#, label="2 SSP,po" )     # c=cphotons, ls=":" , 
ax1l.plot(agevec, sp.log10(ssp_qhivec2tot), c=c_ssp_2, ls="-", lw=3, label="2 SSP,tot")       # c=cphotons, ls="--", 
ax1l.plot(agevec, sp.log10(csf_qhivec1   ), c=c_csf_1, ls="-", lw=3, label="1 CSF"    )       # c=cphotons, ls="-.", 

ax1l.set_xlabel("Time since t=0 (Myr)", fontsize=fontsize)
ax1l.set_ylabel("log Q(HI) (photons/sec)", fontsize=fontsize)#, color=cphotons)
ax1l.tick_params('y')#, fontsize=fontsize)#, colors=cphotons)

ax1l.legend(loc=3, ncol=1)

ax1r = ax1l.twinx()
ax1r.plot(agevec, sp.log10(ssp_emechvec1   ), c=c_ssp_1, ls="--", lw=3, label="1 SSP"    )   # c=ckinetic, ls="-" , 
ax1r.plot(agevec, sp.log10(ssp_emechvec2pr ), c=c_ssp_2, ls=":" , lw=1)#, label="2 SSP,pr" ) # c=ckinetic, ls=":" , 
ax1r.plot(agevec, sp.log10(ssp_emechvec2po ), c=c_ssp_2, ls=":" , lw=1)#, label="2 SSP,po" ) # c=ckinetic, ls=":" , 
ax1r.plot(agevec, sp.log10(ssp_emechvec2tot), c=c_ssp_2, ls="--", lw=3, label="2 SSP,tot")   # c=ckinetic, ls="--", 
ax1r.plot(agevec, sp.log10(csf_emechvec1   ), c=c_csf_1, ls="--", lw=3, label="1 CSF"    )   # c=ckinetic, ls="-.", 
ax1r.set_ylabel("log Emech (erg)", fontsize=fontsize)#, color=ckinetic)
ax1r.tick_params('y')#, fontsize=fontsize)#, colors=ckinetic)

### right plot with luminosity 
lab1 = ax2l.plot(agevec, sp.log10(ssp_qhivec1   ),  c=c_ssp_1, ls="-", lw=3, label="Photons"    )   # c=cphotons, ls="-" ,
ax2l.plot(agevec, sp.log10(ssp_qhivec2pr ),  c=c_ssp_2, ls=":", lw=1)#, label="2 SSP,pr" ) # c=cphotons, ls=":" ,
ax2l.plot(agevec, sp.log10(ssp_qhivec2po ),  c=c_ssp_2, ls=":", lw=1)#, label="2 SSP,po" ) # c=cphotons, ls=":" ,
ax2l.plot(agevec, sp.log10(ssp_qhivec2tot),  c=c_ssp_2, ls="-", lw=3)#, label="2 SSP,tot")   # c=cphotons, ls="--",
ax2l.plot(agevec, sp.log10(csf_qhivec1   ),  c=c_csf_1, ls="-", lw=3)#, label="1 CSF"    )   # c=cphotons, ls="-.",
ax2l.set_xlabel("Time since t=0 (Myr)", fontsize=fontsize)
ax2l.set_ylabel("log Q(HI) (photons/sec)", fontsize=fontsize)#, color=cphotons)
ax2l.tick_params('y')#, fontsize=fontsize)#, colors=cphotons)

ax2r = ax2l.twinx()
lab2 = ax2r.plot(agevec, sp.log10(ssp_lmechvec1   ), c=c_ssp_1, ls="--", lw=3, label="Mechanics")   # c=ckinetic, ls="-" ,
ax2r.plot(agevec, sp.log10(ssp_lmechvec2pr ), c=c_ssp_2, ls=":" , lw=1)#, label="SSP,pr" ) # c=ckinetic, ls=":" ,
ax2r.plot(agevec, sp.log10(ssp_lmechvec2po ), c=c_ssp_2, ls=":" , lw=1)#, label="SSP,po" ) # c=ckinetic, ls=":" ,
ax2r.plot(agevec, sp.log10(ssp_lmechvec2tot), c=c_ssp_2, ls="--", lw=3)#, label="SSP,tot")   # c=ckinetic, ls="--",
ax2r.plot(agevec, sp.log10(csf_lmechvec1   ), c=c_csf_1, ls="--", lw=3)#, label="CSF"    )   # c=ckinetic, ls="-.",
ax2r.set_ylabel("log Lmech (erg/sec)", fontsize=fontsize)#, color=ckinetic)
ax2r.tick_params('y')#, fontsize=fontsize)#, colors=ckinetic)

#ax2l.legend((lab1, lab2), ("Photons", "Mechanics"),  loc=3, ncol=1)
#ax2l.legend(loc=3)

for ax in [ ax1l, ax1r, ax2l, ax2r ]:
    ax.xaxis.set_tick_params(which='both', labelsize=fontsize) 
    ax.yaxis.set_tick_params(which='both', labelsize=fontsize) 
    ax.xaxis.set_major_locator(MultipleLocator(5))
    ax.xaxis.set_minor_locator(MultipleLocator(1))
    ax.yaxis.set_major_locator(MultipleLocator(1))
    ax.yaxis.set_minor_locator(MultipleLocator(0.2))

fig.subplots_adjust(left=0.09, right=0.91, hspace=0.5, wspace=0.4, bottom=0.15, top=0.98)
fig.savefig("evolqe.pdf")



#USING THE RATIOS
#########
fig = plt.figure(figsize=(10,4))


ax1l = fig.add_subplot(121)#, position=[0.05, 0.1, 0.4, 0.8])
ax2l = fig.add_subplot(122)#, position=[0.55, 0.1, 0.4, 0.8])

### left plot with integrated energy
ax1l.plot(agevec, sp.log10(ssp_emechvec1   /ssp_qhivec1   ), c=c_ssp_1, ls="-", lw=3, label="1 SSP"    )       # c=cphotons, ls="-" , 
ax1l.plot(agevec, sp.log10(ssp_emechvec2pr /ssp_qhivec2pr ), c=c_ssp_2, ls=":", lw=1)#, label="2 SSP,pr" )     # c=cphotons, ls=":" , 
ax1l.plot(agevec, sp.log10(ssp_emechvec2po /ssp_qhivec2po ), c=c_ssp_2, ls=":", lw=1)#, label="2 SSP,po" )     # c=cphotons, ls=":" , 
ax1l.plot(agevec, sp.log10(ssp_emechvec2tot/ssp_qhivec2tot), c=c_ssp_2, ls="-", lw=3, label="2 SSP,tot")       # c=cphotons, ls="--", 
ax1l.plot(agevec, sp.log10(csf_emechvec1   /csf_qhivec1   ), c=c_csf_1, ls="-", lw=3, label="1 CSF"    )       # c=cphotons, ls="-.", 

ax1l.set_xlabel("Time since t=0 (Myr)", fontsize=fontsize)
ax1l.set_ylabel("log [ Emech / Q(HI) ] (erg/sec phot)", fontsize=fontsize)#, color=cphotons)
ax1l.tick_params('y')#, fontsize=fontsize)#, colors=cphotons)

ax1l.legend(loc=3, ncol=1)


### right plot with luminosity 
ax2l.plot(agevec, sp.log10(ssp_lmechvec1   /ssp_qhivec1   ),  c=c_ssp_1, ls="-", lw=3, label="1 SSP"    )   # c=cphotons, ls="-" ,
ax2l.plot(agevec, sp.log10(ssp_lmechvec2pr /ssp_qhivec2pr ),  c=c_ssp_2, ls=":", lw=1)#, label="2 SSP,pr" ) # c=cphotons, ls=":" ,
ax2l.plot(agevec, sp.log10(ssp_lmechvec2po /ssp_qhivec2po ),  c=c_ssp_2, ls=":", lw=1)#, label="2 SSP,po" ) # c=cphotons, ls=":" ,
ax2l.plot(agevec, sp.log10(ssp_lmechvec2tot/ssp_qhivec2tot),  c=c_ssp_2, ls="-", lw=3, label="2 SSP,tot")   # c=cphotons, ls="--",
ax2l.plot(agevec, sp.log10(csf_lmechvec1   /csf_qhivec1   ),  c=c_csf_1, ls="-", lw=3, label="1 CSF"    )   # c=cphotons, ls="-.",
ax2l.set_xlabel("Time since t=0 (Myr)", fontsize=fontsize)
ax2l.set_ylabel("log [ Lmech / Q(HI) ]  (erg / phot)", fontsize=fontsize)#, color=cphotons)
ax2l.tick_params('y')#, fontsize=fontsize)#, colors=cphotons)


#ax2l.legend(loc=3)

for ax in [ ax1l, ax2l ]:
    ax.xaxis.set_tick_params(which='both', labelsize=fontsize) 
    ax.yaxis.set_tick_params(which='both', labelsize=fontsize) 
    ax.xaxis.set_major_locator(MultipleLocator(5))
    ax.xaxis.set_minor_locator(MultipleLocator(1))
    ax.yaxis.set_major_locator(MultipleLocator(1))
    ax.yaxis.set_minor_locator(MultipleLocator(0.2))

fig.subplots_adjust(left=0.09, right=0.91, hspace=0.5, wspace=0.4, bottom=0.15, top=0.98)
fig.savefig("evolrat.pdf")

plt.show()
