import scipy as sp
from matplotlib import pyplot as plt 
#import pickle


def get_histpoints(vals, bins):
    
    v = sp.histogram(vals, bins=bins)
    n, x = v[0], v[1]
    xc = (x[0:-1]+x[1:])/2
    return xc, n




def get_norm_histpoints(vals, binlo, binhi, Nb, funcform):
    
    itake = (binlo < vals) & (vals < binhi)
    
    if funcform == "lin": 
        bins  = sp.linspace(binlo, binhi, Nb)
        vtake = vals[itake]
    elif funcform == "log": 
        bins  = sp.linspace(sp.log10(binlo), sp.log10(binhi), Nb)
        vtake = sp.log10(vals[itake])
            
    cent, norm = get_histpoints(vtake, bins)
    integral   = sp.trapz(norm, x=cent )
    return cent, norm/integral



def plot_pops_1and2_old(fitres1ssp, fitres2ssp, fitres1csf, syndatSsp, syndatCsf, \
    ebvlo=0., ebvhi=1., agelo=6., agehi=7.3, masslo=0., masshi=1.e4, fwhmlo=0., fwhmhi=1000.,  \
    qhilo=1.e51, qhihi=1.e56, qheilo=1.e51, qheihi=1.e56, qheiilo=1.e50, qheiihi=1.e53, \
    llo=1.e40, lhi=1.e44, elo=1.e51, ehi=1.e57, \
    Npt=100, fnroot="delme"): 
    
    # get the chains of each quantity 
    oSsp1_ebv  = fitres1ssp.flatchain['ebv']     .values
    oSsp1_a    = fitres1ssp.flatchain['onepop_a'].values
    oSsp1_n    = fitres1ssp.flatchain['onepop_n'].values
    oSsp1_fwhm = fitres1ssp.flatchain['fwhm']    .values
    
    oSsp2_ebv  = fitres2ssp.flatchain['ebv']     .values
    oSsp2pr_a  = fitres2ssp.flatchain['PreSn_a'] .values
    oSsp2po_a  = fitres2ssp.flatchain['PostSn_a'].values
    oSsp2pr_n  = fitres2ssp.flatchain['PreSn_n'] .values
    oSsp2po_n  = fitres2ssp.flatchain['PostSn_n'].values
    oSsp2_fwhm = fitres2ssp.flatchain['fwhm']    .values
    
    oCsf1_ebv  = fitres1csf.flatchain['ebv']     .values
    oCsf1_a    = fitres1csf.flatchain['onepop_a'].values
    oCsf1_n    = fitres1csf.flatchain['onepop_n'].values
    oCsf1_fwhm = fitres1csf.flatchain['fwhm']    .values
                     
    # compute the normalized pdfs
    bSsp1c_ebv , bSsp1n_ebv  = get_norm_histpoints(oSsp1_ebv , ebvlo , ebvhi , Npt, "lin")
    bSsp1c_a   , bSsp1n_a    = get_norm_histpoints(oSsp1_a   , agelo , agehi , Npt, "lin")
    bSsp1c_n   , bSsp1n_n    = get_norm_histpoints(oSsp1_n   , masslo, masshi, Npt, "log")
    bSsp1c_fwhm, bSsp1n_fwhm = get_norm_histpoints(oSsp1_fwhm, fwhmlo, fwhmhi, Npt, "lin")
    
    bSsp2c_ebv , bSsp2n_ebv  = get_norm_histpoints(oSsp2_ebv , ebvlo , ebvhi , Npt, "lin")
    bSsp2cpr_a , bSsp2npr_a  = get_norm_histpoints(oSsp2pr_a , agelo , agehi , Npt, "lin")
    bSsp2cpo_a , bSsp2npo_a  = get_norm_histpoints(oSsp2po_a , agelo , agehi , Npt, "lin")
    bSsp2cpr_n , bSsp2npr_n  = get_norm_histpoints(oSsp2pr_n , masslo, masshi, Npt, "log")
    bSsp2cpo_n , bSsp2npo_n  = get_norm_histpoints(oSsp2po_n , masslo, masshi, Npt, "log")
    bSsp2c_fwhm, bSsp2n_fwhm = get_norm_histpoints(oSsp2_fwhm, fwhmlo, fwhmhi, Npt, "lin")
    
    bCsf1c_ebv , bCsf1n_ebv  = get_norm_histpoints(oCsf1_ebv , ebvlo , ebvhi , Npt, "lin")
    bCsf1c_a   , bCsf1n_a    = get_norm_histpoints(oCsf1_a   , agelo , agehi , Npt, "lin")
    bCsf1c_n   , bCsf1n_n    = get_norm_histpoints(oCsf1_n   , masslo, masshi, Npt, "log")
    bCsf1c_fwhm, bCsf1n_fwhm = get_norm_histpoints(oCsf1_fwhm, fwhmlo, fwhmhi, Npt, "lin")
    
    # plot the pdfs 
    f = plt.figure(figsize=(10,4))
    ae = f.add_subplot(141)
    aa = f.add_subplot(142)
    an = f.add_subplot(143)
    af = f.add_subplot(144)
    
    ae.plot(bSsp1c_ebv , bSsp1n_ebv , label="1 SSP"        , c="indigo", ls="-" )
    ae.plot(bSsp2c_ebv , bSsp2n_ebv , label="2 SSP"        , c="darkgreen" , ls="-.")
    ae.plot(bCsf1c_ebv , bCsf1n_ebv , label="1 CSF"        , c="darkorange", ls="--")
       
    aa.plot(bSsp1c_a   , bSsp1n_a   , label="1 SSP"        , c="indigo", ls="-" )
    aa.plot(bSsp2cpr_a , bSsp2npr_a , label="2 SSP, preSN" , c="darkgreen" , ls="-.")
    aa.plot(bSsp2cpo_a , bSsp2npo_a , label="2 SSP, postSN", c="darkgreen" , ls=":" )
    aa.plot(bCsf1c_a   , bCsf1n_a   , label="1 CSF"        , c="darkorange", ls="--")
       
    an.plot(bSsp1c_n   , bSsp1n_n   , label="1 SSP"        , c="indigo", ls="-" )
    an.plot(bSsp2cpr_n , bSsp2npr_n , label="2 SSP, preSN" , c="darkgreen" , ls="-.")
    an.plot(bSsp2cpo_n , bSsp2npo_n , label="2 SSP, postSN", c="darkgreen" , ls=":" )
    an.plot(bCsf1c_n   , bCsf1n_n   , label="1 CSF"        , c="darkorange", ls="--")
    
    af.plot(bSsp1c_fwhm, bSsp1n_fwhm, label="1 SSP"        , c="indigo", ls="-" )
    af.plot(bSsp2c_fwhm, bSsp2n_fwhm, label="2 SSP"        , c="darkgreen" , ls="-.")
    af.plot(bCsf1c_fwhm, bCsf1n_fwhm, label="1 CSF"        , c="darkorange", ls="--")
    
    #aa.semilogx()
    #an.semilogx()
    
    ae.legend(loc=2)
    aa.legend(loc=2)
    an.legend(loc=2)
    af.legend(loc=2)
    
    ae.set_yticklabels([])
    aa.set_yticklabels([])
    an.set_yticklabels([])
    af.set_yticklabels([])
    
    ae.set_ylabel(r"Normalized conditional pdf")
    ae.set_xlabel(r"E$_\mathrm{B-V}$ [ mag ]"   )
    aa.set_xlabel(r"log Age [ yr ]"      )
    an.set_xlabel(r"log M$_\mathrm{stell}$ [ 10$^6$ M$_\odot$ ]")
    af.set_xlabel(r"FWHM [ km/s ]")
    
    f.savefig(fnroot+"_pri.pdf")
    
    #pob = open(fnroot+"_pri.pkl"    , "wb")
    #pickle.dump(f,  pob)
    #pob.close()


    # compute the secondary quantities from the synthetic data
    oSsp1_qhi     = sp.zeros_like(oSsp1_ebv)
    oSsp2pr_qhi   = sp.zeros_like(oSsp1_ebv)
    oSsp2po_qhi   = sp.zeros_like(oSsp1_ebv)
    oCsf1_qhi     = sp.zeros_like(oSsp1_ebv)
    
    oSsp1_qhei    = sp.zeros_like(oSsp1_ebv)
    oSsp2pr_qhei  = sp.zeros_like(oSsp1_ebv)
    oSsp2po_qhei  = sp.zeros_like(oSsp1_ebv)
    oCsf1_qhei    = sp.zeros_like(oSsp1_ebv)
    
    oSsp1_qheii   = sp.zeros_like(oSsp1_ebv)
    oSsp2pr_qheii = sp.zeros_like(oSsp1_ebv)
    oSsp2po_qheii = sp.zeros_like(oSsp1_ebv)
    oCsf1_qheii   = sp.zeros_like(oSsp1_ebv)
    
    oSsp1_lmech   = sp.zeros_like(oSsp1_ebv)
    oSsp2pr_lmech = sp.zeros_like(oSsp1_ebv)
    oSsp2po_lmech = sp.zeros_like(oSsp1_ebv)
    oCsf1_lmech   = sp.zeros_like(oSsp1_ebv)
    
    oSsp1_emech   = sp.zeros_like(oSsp1_ebv)
    oSsp2pr_emech = sp.zeros_like(oSsp1_ebv)
    oSsp2po_emech = sp.zeros_like(oSsp1_ebv)
    oCsf1_emech   = sp.zeros_like(oSsp1_ebv)
    
    for ireal in range(len(oSsp1_ebv)):
        oSsp1_qhi  [ireal], oSsp1_qhei  [ireal], oSsp1_qheii  [ireal], oSsp1_lmech  [ireal], oSsp1_emech  [ireal] = syndatSsp.calc_secondprop(oSsp1_a  [ireal], oSsp1_n  [ireal])
        oSsp2pr_qhi[ireal], oSsp2pr_qhei[ireal], oSsp2pr_qheii[ireal], oSsp2pr_lmech[ireal], oSsp2pr_emech[ireal] = syndatSsp.calc_secondprop(oSsp2pr_a[ireal], oSsp2pr_n[ireal])
        oSsp2po_qhi[ireal], oSsp2po_qhei[ireal], oSsp2po_qheii[ireal], oSsp2po_lmech[ireal], oSsp2po_emech[ireal] = syndatSsp.calc_secondprop(oSsp2po_a[ireal], oSsp2po_n[ireal])
        oCsf1_qhi  [ireal], oCsf1_qhei  [ireal], oCsf1_qheii  [ireal], oCsf1_lmech  [ireal], oCsf1_emech  [ireal] = syndatCsf.calc_secondprop(oCsf1_a  [ireal], oCsf1_n  [ireal])

        print (ireal, oSsp1_qhi  [ireal], oSsp1_qhei  [ireal], oSsp1_qheii  [ireal], oSsp1_lmech  [ireal], oSsp1_emech  [ireal])
    
    bSsp1c_qhi    , bSsp1n_qhi     = get_norm_histpoints(oSsp1_qhi    , qhilo  , qhihi  , Npt, "log")
    bSsp1c_qhei   , bSsp1n_qhei    = get_norm_histpoints(oSsp1_qhei   , qheilo , qheihi , Npt, "log")
    bSsp1c_qheii  , bSsp1n_qheii   = get_norm_histpoints(oSsp1_qheii  , qheiilo, qheiihi, Npt, "log")
    bSsp1c_lmech  , bSsp1n_lmech   = get_norm_histpoints(oSsp1_lmech  , llo    , lhi    , Npt, "log")
    bSsp1c_emech  , bSsp1n_emech   = get_norm_histpoints(oSsp1_emech  , elo    , ehi    , Npt, "log")
    
    bSsp2cpr_qhi  , bSsp2npr_qhi   = get_norm_histpoints(oSsp2pr_qhi  , qhilo  , qhihi  , Npt, "log")
    bSsp2cpr_qhei , bSsp2npr_qhei  = get_norm_histpoints(oSsp2pr_qhei , qheilo , qheihi , Npt, "log")
    bSsp2cpr_qheii, bSsp2npr_qheii = get_norm_histpoints(oSsp2pr_qheii, qheiilo, qheiihi, Npt, "log")
    bSsp2cpr_lmech, bSsp2npr_lmech = get_norm_histpoints(oSsp2pr_lmech, llo    , lhi    , Npt, "log")
    bSsp2cpr_emech, bSsp2npr_emech = get_norm_histpoints(oSsp2pr_emech, elo    , ehi    , Npt, "log")
    
    bSsp2cpo_qhi  , bSsp2npo_qhi   = get_norm_histpoints(oSsp2po_qhi  , qhilo  , qhihi  , Npt, "log")
    bSsp2cpo_qhei , bSsp2npo_qhei  = get_norm_histpoints(oSsp2po_qhei , qheilo , qheihi , Npt, "log")
    bSsp2cpo_qheii, bSsp2npo_qheii = get_norm_histpoints(oSsp2po_qheii, qheiilo, qheiihi, Npt, "log")
    bSsp2cpo_lmech, bSsp2npo_lmech = get_norm_histpoints(oSsp2po_lmech, llo    , lhi    , Npt, "log")
    bSsp2cpo_emech, bSsp2npo_emech = get_norm_histpoints(oSsp2po_emech, elo    , ehi    , Npt, "log")
    
    bCsf1c_qhi    , bCsf1n_qhi     = get_norm_histpoints(oCsf1_qhi    , qhilo  , qhihi  , Npt, "log")
    bCsf1c_qhei   , bCsf1n_qhei    = get_norm_histpoints(oCsf1_qhei   , qheilo , qheihi , Npt, "log")
    bCsf1c_qheii  , bCsf1n_qheii   = get_norm_histpoints(oCsf1_qheii  , qheiilo, qheiihi, Npt, "log")
    bCsf1c_lmech  , bCsf1n_lmech   = get_norm_histpoints(oCsf1_lmech  , llo    , lhi    , Npt, "log")
    bCsf1c_emech  , bCsf1n_emech   = get_norm_histpoints(oCsf1_emech  , elo    , ehi    , Npt, "log")
    
    
    f = plt.figure(figsize=(15,4))
    aq1 = f.add_subplot(151)
    aq2 = f.add_subplot(152)
    aq3 = f.add_subplot(153)
    al  = f.add_subplot(154)
    ae  = f.add_subplot(155)
    
    aq1.plot(bSsp1c_qhi    , bSsp1n_qhi     , label="1 SSP"        , c="indigo", ls="-" )
    aq1.plot(bSsp2cpr_qhi  , bSsp2npr_qhi   , label="2 SSP, preSN" , c="darkgreen" , ls="-.")
    aq1.plot(bSsp2cpo_qhi  , bSsp2npo_qhi   , label="2 SSP, postSN", c="darkgreen" , ls=":" )
    aq1.plot(bCsf1c_qhi    , bCsf1n_qhi     , label="1 CSF"        , c="darkorange", ls="--")
       
    aq2.plot(bSsp1c_qhei   , bSsp1n_qhei    , label="1 SSP"        , c="indigo", ls="-" )
    aq2.plot(bSsp2cpr_qhei , bSsp2npr_qhei  , label="2 SSP, preSN" , c="darkgreen" , ls="-.")
    aq2.plot(bSsp2cpo_qhei , bSsp2npo_qhei  , label="2 SSP, postSN", c="darkgreen" , ls=":" )
    aq2.plot(bCsf1c_qhei   , bCsf1n_qhei    , label="1 CSF"        , c="darkorange", ls="--")
       
    aq3.plot(bSsp1c_qheii  , bSsp1n_qheii   , label="1 SSP"        , c="indigo", ls="-" )
    aq3.plot(bSsp2cpr_qheii, bSsp2npr_qheii , label="2 SSP, preSN" , c="darkgreen" , ls="-.")
    aq3.plot(bSsp2cpo_qheii, bSsp2npo_qheii , label="2 SSP, postSN", c="darkgreen" , ls=":" )
    aq3.plot(bCsf1c_qheii  , bCsf1n_qheii   , label="1 CSF"        , c="darkorange", ls="--")
       
    al.plot(bSsp1c_lmech   , bSsp1n_lmech   , label="1 SSP"        , c="indigo", ls="-" )
    al.plot(bSsp2cpr_lmech , bSsp2npr_lmech , label="2 SSP, preSN" , c="darkgreen" , ls="-.")
    al.plot(bSsp2cpo_lmech , bSsp2npo_lmech , label="2 SSP, postSN", c="darkgreen" , ls=":" )
    al.plot(bCsf1c_lmech   , bCsf1n_lmech   , label="1 CSF"        , c="darkorange", ls="--")
       
    ae.plot(bSsp1c_emech   , bSsp1n_emech   , label="1 SSP"        , c="indigo", ls="-" )
    ae.plot(bSsp2cpr_emech , bSsp2npr_emech , label="2 SSP, preSN" , c="darkgreen" , ls="-.")
    ae.plot(bSsp2cpo_emech , bSsp2npo_emech , label="2 SSP, postSN", c="darkgreen" , ls=":" )
    ae.plot(bCsf1c_emech   , bCsf1n_emech   , label="1 CSF"        , c="darkorange", ls="--")
    
    
    aq1.legend(loc=2)
    aq2.legend(loc=2)
    aq3.legend(loc=2)
    al .legend(loc=2)
    ae .legend(loc=2)
    
    aq1.set_yticklabels([])
    aq2.set_yticklabels([])
    aq3.set_yticklabels([])
    al .set_yticklabels([])
    ae .set_yticklabels([])
    
    aq1.set_ylabel(r"Normalized conditional pdf"         )
    aq1.set_xlabel(r"log ( Q(H I) [ photons/s ] )"       )
    aq2.set_xlabel(r"log ( Q(He I) [ photons/s ] )"      )
    aq3.set_xlabel(r"log ( Q(He II) [ photons/s ] )"     )
    al .set_xlabel(r"log ( L$_\mathrm{mech}$ [ erg/s ] )")
    ae .set_xlabel(r"log ( E$_\mathrm{mech}$ [ erg ] )"  )
    
    f.savefig(fnroot+"_sec.pdf")
    
    #pob = open(fnroot+"_sec.pkl"    , "wb")
    #pickle.dump(f,  pob)
    #pob.close()
    
    
    # first 3 are QHI, then QHeI, ... QHeII,  Lmech,  and Emech
    return  oSsp1_qhi  , oSsp2pr_qhi  , oSsp2po_qhi  , oCsf1_qhi  , \
        oSsp1_qhei , oSsp2pr_qhei , oSsp2po_qhei , oCsf1_qhei , \
        oSsp1_qheii, oSsp2pr_qheii, oSsp2po_qheii, oCsf1_qheii, \
        oSsp1_lmech, oSsp2pr_lmech, oSsp2po_lmech, oCsf1_lmech, \
        oSsp1_emech, oSsp2pr_emech, oSsp2po_emech, oCsf1_emech



def plot_pops_1and2(fitres1, fitres2, syndat, \
    ebvlo=0., ebvhi=1., agelo=6., agehi=7.3, masslo=0., masshi=1.e4, fwhmlo=0., fwhmhi=1000.,  \
    qhilo=1.e51, qhihi=1.e56, qheilo=1.e51, qheihi=1.e56, qheiilo=1.e50, qheiihi=1.e53, \
    llo=1.e40, lhi=1.e44, elo=1.e51, ehi=1.e57, \
    Npt=100, fnroot="delme"): 
    
    # get the chains of each quantity 
    o1_ebv  = fitres1.flatchain['ebv']     .values
    o1_a    = fitres1.flatchain['onepop_a'].values
    o1_n    = fitres1.flatchain['onepop_n'].values
    #o1_fwhm = fitres1.flatchain['fwhm']    .values
    
    o2_ebv  = fitres2.flatchain['ebv']     .values
    o2pr_a  = fitres2.flatchain['PreSn_a'] .values
    o2po_a  = fitres2.flatchain['PostSn_a'].values
    o2pr_n  = fitres2.flatchain['PreSn_n'] .values
    o2po_n  = fitres2.flatchain['PostSn_n'].values
    #o2_fwhm = fitres2.flatchain['fwhm']    .values
    
    # compute the normalized pdfs
    b1c_ebv , b1n_ebv  = get_norm_histpoints(o1_ebv , ebvlo , ebvhi , Npt, "lin")
    b1c_a   , b1n_a    = get_norm_histpoints(o1_a   , agelo , agehi , Npt, "lin")
    b1c_n   , b1n_n    = get_norm_histpoints(o1_n   , masslo, masshi, Npt, "log")
    #b1c_fwhm, b1n_fwhm = get_norm_histpoints(o1_fwhm, fwhmlo, fwhmhi, Npt, "lin")
    
    b2c_ebv , b2n_ebv  = get_norm_histpoints(o2_ebv , ebvlo , ebvhi , Npt, "lin")
    b2cpr_a , b2npr_a  = get_norm_histpoints(o2pr_a , agelo , agehi , Npt, "lin")
    b2cpo_a , b2npo_a  = get_norm_histpoints(o2po_a , agelo , agehi , Npt, "lin")
    b2cpr_n , b2npr_n  = get_norm_histpoints(o2pr_n , masslo, masshi, Npt, "log")
    b2cpo_n , b2npo_n  = get_norm_histpoints(o2po_n , masslo, masshi, Npt, "log")
    #b2c_fwhm, b2n_fwhm = get_norm_histpoints(o2_fwhm, fwhmlo, fwhmhi, Npt, "lin")
    
    # plot the pdfs 
    f = plt.figure(figsize=(10,4))
    ae = f.add_subplot(131)
    aa = f.add_subplot(132)
    an = f.add_subplot(133)
    #af = f.add_subplot(144)
    
    ae.plot(b1c_ebv , b1n_ebv , label="1 pop"        , c="indigo", ls="-" )
    ae.plot(b2c_ebv , b2n_ebv , label="2 pop"        , c="darkgreen" , ls="-.")
       
    aa.plot(b1c_a   , b1n_a   , label="1 pop"        , c="indigo", ls="-" )
    aa.plot(b2cpr_a , b2npr_a , label="2 pop, preSN" , c="darkgreen" , ls="-.")
    aa.plot(b2cpo_a , b2npo_a , label="2 pop, postSN", c="darkgreen" , ls=":" )
       
    an.plot(b1c_n   , b1n_n   , label="1 pop"        , c="indigo", ls="-" )
    an.plot(b2cpr_n , b2npr_n , label="2 pop, preSN" , c="darkgreen" , ls="-.")
    an.plot(b2cpo_n , b2npo_n , label="2 pop, postSN", c="darkgreen" , ls=":" )
    
    #af.plot(b1c_fwhm, b1n_fwhm, label="1 pop"        , c="indigo", ls="-" )
    #af.plot(b2c_fwhm, b2n_fwhm, label="2 pop"        , c="darkgreen" , ls="-.")
    
    #aa.semilogx()
    #an.semilogx()
    
    ae.legend(loc=2)
    aa.legend(loc=2)
    an.legend(loc=2)
    #af.legend(loc=2)
    
    ae.set_yticklabels([])
    aa.set_yticklabels([])
    an.set_yticklabels([])
    #af.set_yticklabels([])
    
    ae.set_ylabel(r"Normalized conditional pdf")
    ae.set_xlabel(r"E$_\mathrm{B-V}$ [ mag ]"   )
    aa.set_xlabel(r"log Age [ yr ]"      )
    an.set_xlabel(r"log M$_\mathrm{stell}$ [ 10$^6$ M$_\odot$ ]")
    #af.set_xlabel(r"FWHM [ km/s ]")
    
    f.savefig(fnroot+"_pri.pdf")
    
    #pob = open(fnroot+"_pri.pkl"    , "wb")
    #pickle.dump(f,  pob)
    #pob.close()


    # compute the secondary quantities from the synthetic data
    o1_qhi     = sp.zeros_like(o1_ebv)
    o2pr_qhi   = sp.zeros_like(o1_ebv)
    o2po_qhi   = sp.zeros_like(o1_ebv)
    
    o1_qhei    = sp.zeros_like(o1_ebv)
    o2pr_qhei  = sp.zeros_like(o1_ebv)
    o2po_qhei  = sp.zeros_like(o1_ebv)
    
    o1_qheii   = sp.zeros_like(o1_ebv)
    o2pr_qheii = sp.zeros_like(o1_ebv)
    o2po_qheii = sp.zeros_like(o1_ebv)
    
    o1_lmech   = sp.zeros_like(o1_ebv)
    o2pr_lmech = sp.zeros_like(o1_ebv)
    o2po_lmech = sp.zeros_like(o1_ebv)
    
    o1_emech   = sp.zeros_like(o1_ebv)
    o2pr_emech = sp.zeros_like(o1_ebv)
    o2po_emech = sp.zeros_like(o1_ebv)
    
    for ireal in range(len(o1_ebv)):
        o1_qhi  [ireal], o1_qhei  [ireal], o1_qheii  [ireal], o1_lmech  [ireal], o1_emech  [ireal] = syndat.calc_secondprop(o1_a  [ireal], o1_n  [ireal])
        o2pr_qhi[ireal], o2pr_qhei[ireal], o2pr_qheii[ireal], o2pr_lmech[ireal], o2pr_emech[ireal] = syndat.calc_secondprop(o2pr_a[ireal], o2pr_n[ireal])
        o2po_qhi[ireal], o2po_qhei[ireal], o2po_qheii[ireal], o2po_lmech[ireal], o2po_emech[ireal] = syndat.calc_secondprop(o2po_a[ireal], o2po_n[ireal])

        print (ireal, o1_qhi  [ireal], o1_qhei  [ireal], o1_qheii  [ireal], o1_lmech  [ireal], o1_emech  [ireal])
    
    b1c_qhi    , b1n_qhi     = get_norm_histpoints(o1_qhi    , qhilo  , qhihi  , Npt, "log")
    b1c_qhei   , b1n_qhei    = get_norm_histpoints(o1_qhei   , qheilo , qheihi , Npt, "log")
    b1c_qheii  , b1n_qheii   = get_norm_histpoints(o1_qheii  , qheiilo, qheiihi, Npt, "log")
    b1c_lmech  , b1n_lmech   = get_norm_histpoints(o1_lmech  , llo    , lhi    , Npt, "log")
    b1c_emech  , b1n_emech   = get_norm_histpoints(o1_emech  , elo    , ehi    , Npt, "log")
    
    b2cpr_qhi  , b2npr_qhi   = get_norm_histpoints(o2pr_qhi  , qhilo  , qhihi  , Npt, "log")
    b2cpr_qhei , b2npr_qhei  = get_norm_histpoints(o2pr_qhei , qheilo , qheihi , Npt, "log")
    b2cpr_qheii, b2npr_qheii = get_norm_histpoints(o2pr_qheii, qheiilo, qheiihi, Npt, "log")
    b2cpr_lmech, b2npr_lmech = get_norm_histpoints(o2pr_lmech, llo    , lhi    , Npt, "log")
    b2cpr_emech, b2npr_emech = get_norm_histpoints(o2pr_emech, elo    , ehi    , Npt, "log")
    
    b2cpo_qhi  , b2npo_qhi   = get_norm_histpoints(o2po_qhi  , qhilo  , qhihi  , Npt, "log")
    b2cpo_qhei , b2npo_qhei  = get_norm_histpoints(o2po_qhei , qheilo , qheihi , Npt, "log")
    b2cpo_qheii, b2npo_qheii = get_norm_histpoints(o2po_qheii, qheiilo, qheiihi, Npt, "log")
    b2cpo_lmech, b2npo_lmech = get_norm_histpoints(o2po_lmech, llo    , lhi    , Npt, "log")
    b2cpo_emech, b2npo_emech = get_norm_histpoints(o2po_emech, elo    , ehi    , Npt, "log")
    
    
    f = plt.figure(figsize=(15,4))
    aq1 = f.add_subplot(151)
    aq2 = f.add_subplot(152)
    aq3 = f.add_subplot(153)
    al  = f.add_subplot(154)
    ae  = f.add_subplot(155)
    
    aq1.plot(b1c_qhi    , b1n_qhi     , label="1 pop"        , c="indigo", ls="-" )
    aq1.plot(b2cpr_qhi  , b2npr_qhi   , label="2 pop, preSN" , c="darkgreen" , ls="-.")
    aq1.plot(b2cpo_qhi  , b2npo_qhi   , label="2 pop, postSN", c="darkgreen" , ls=":" )
       
    aq2.plot(b1c_qhei   , b1n_qhei    , label="1 pop"        , c="indigo", ls="-" )
    aq2.plot(b2cpr_qhei , b2npr_qhei  , label="2 pop, preSN" , c="darkgreen" , ls="-.")
    aq2.plot(b2cpo_qhei , b2npo_qhei  , label="2 pop, postSN", c="darkgreen" , ls=":" )
       
    aq3.plot(b1c_qheii  , b1n_qheii   , label="1 pop"        , c="indigo", ls="-" )
    aq3.plot(b2cpr_qheii, b2npr_qheii , label="2 pop, preSN" , c="darkgreen" , ls="-.")
    aq3.plot(b2cpo_qheii, b2npo_qheii , label="2 pop, postSN", c="darkgreen" , ls=":" )
       
    al.plot(b1c_lmech   , b1n_lmech   , label="1 pop"        , c="indigo", ls="-" )
    al.plot(b2cpr_lmech , b2npr_lmech , label="2 pop, preSN" , c="darkgreen" , ls="-.")
    al.plot(b2cpo_lmech , b2npo_lmech , label="2 pop, postSN", c="darkgreen" , ls=":" )
       
    ae.plot(b1c_emech   , b1n_emech   , label="1 pop"        , c="indigo", ls="-" )
    ae.plot(b2cpr_emech , b2npr_emech , label="2 pop, preSN" , c="darkgreen" , ls="-.")
    ae.plot(b2cpo_emech , b2npo_emech , label="2 pop, postSN", c="darkgreen" , ls=":" )
    
    
    aq1.legend(loc=2)
    aq2.legend(loc=2)
    aq3.legend(loc=2)
    al .legend(loc=2)
    ae .legend(loc=2)
    
    aq1.set_yticklabels([])
    aq2.set_yticklabels([])
    aq3.set_yticklabels([])
    al .set_yticklabels([])
    ae .set_yticklabels([])
    
    aq1.set_ylabel(r"Normalized conditional pdf"         )
    aq1.set_xlabel(r"log ( Q(H I) [ photons/s ] )"       )
    aq2.set_xlabel(r"log ( Q(He I) [ photons/s ] )"      )
    aq3.set_xlabel(r"log ( Q(He II) [ photons/s ] )"     )
    al .set_xlabel(r"log ( L$_\mathrm{mech}$ [ erg/s ] )")
    ae .set_xlabel(r"log ( E$_\mathrm{mech}$ [ erg ] )"  )
    
    f.savefig(fnroot+"_sec.pdf")
    
    #pob = open(fnroot+"_sec.pkl"    , "wb")
    #pickle.dump(f,  pob)
    #pob.close()
    
    
    # first 3 are QHI, then QHeI, ... QHeII,  Lmech,  and Emech
    return  o1_qhi  , o2pr_qhi  , o2po_qhi  ,  \
        o1_qhei , o2pr_qhei , o2po_qhei ,  \
        o1_qheii, o2pr_qheii, o2po_qheii,  \
        o1_lmech, o2pr_lmech, o2po_lmech,  \
        o1_emech, o2pr_emech, o2po_emech



def write_mc_pdf(fitres1, fitres2, syndat, fnroot="delme"): 
   
    fn1 = fnroot+"_mcpdf_pri.txt" 
    fn2 = fnroot+"_mcpdf_sec.txt" 

    # get the chains of each quantity 
    #MS using a different extinction for each population
    o1_ebv = fitres1.flatchain['onepop_ebv'].values
    o1_a   = fitres1.flatchain['onepop_a'  ].values
    o1_z   = fitres1.flatchain['onepop_z'  ].values
    o1_n   = fitres1.flatchain['onepop_n'  ].values
    
    o2pr_ebv = fitres2.flatchain['PreSn_ebv' ].values
    o2po_ebv = fitres2.flatchain['PostSn_ebv'].values
    o2pr_a   = fitres2.flatchain['PreSn_a'   ].values
    o2po_a   = fitres2.flatchain['PostSn_a'  ].values
    o2pr_z   = fitres2.flatchain['PreSn_z'   ].values
    o2po_z   = fitres2.flatchain['PostSn_z'  ].values
    o2pr_n   = fitres2.flatchain['PreSn_n'   ].values
    o2po_n   = fitres2.flatchain['PostSn_n'  ].values
    
    # compute the secondary quantities from the synthetic data
    o1_qhi     = sp.zeros_like(o1_ebv)
    o2pr_qhi   = sp.zeros_like(o1_ebv)
    o2po_qhi   = sp.zeros_like(o1_ebv)
    o1_qhei    = sp.zeros_like(o1_ebv)
    o2pr_qhei  = sp.zeros_like(o1_ebv)
    o2po_qhei  = sp.zeros_like(o1_ebv)
    o1_qheii   = sp.zeros_like(o1_ebv)
    o2pr_qheii = sp.zeros_like(o1_ebv)
    o2po_qheii = sp.zeros_like(o1_ebv)
    o1_lmech   = sp.zeros_like(o1_ebv)
    o2pr_lmech = sp.zeros_like(o1_ebv)
    o2po_lmech = sp.zeros_like(o1_ebv)
    o1_emech   = sp.zeros_like(o1_ebv)
    o2pr_emech = sp.zeros_like(o1_ebv)
    o2po_emech = sp.zeros_like(o1_ebv)
    
    fh_pri = open(fn1, "w")
    fh_sec = open(fn2, "w")

    spri  = "#{:>5}  {:>8}  {:>8}  {:>11}  {:>8}  ".format("Nmc", "age1", "met1", "norm1", "ebv1")
    spri += "{:>8}  {:>8}  {:>11}  {:>8}"          .format("age2pr", "met2pr", "norm2pr", "ebv2pr")
    spri += "{:>8}  {:>8}  {:>11}  {:>8}\n"        .format("age2po", "met2po", "norm2po", "ebv2po")
    spri += "#{:>5}  {:>8}  {:>8}  {:>11}  {:>8}  ".format("", "yr", "", "", "mag")
    spri += "{:>8}  {:>8}  {:>11}  "               .format("yr", "", "")
    spri += "{:>8}  {:>8}  {:>11}  {:>8}\n"        .format("yr", "", "", "mag")
    spri += "#{:>5}  {:>8}  {:>8}  {:>11}  {:>8}  ".format("(1)", "(2)", "(3)", "(4)", "(5)")
    spri += "{:>8}  {:>8}  {:>11}  "               .format("(6)", "(7)", "(8)")
    spri += "{:>8}  {:>8}  {:>11}  {:>8}\n"        .format("(9)", "(10)", "(11)", "(12)")

    ssec  = "#{:>5}  ".format("Nmc")
    ssec += "{:>11}  {:>11}  {:>11}  {:>11}  {:>11}  ".format("qhi1"  , "qhei1"  , "qheii1"  , "lmech1"  , "emech1"  )
    ssec += "{:>11}  {:>11}  {:>11}  {:>11}  {:>11}  ".format("qhi2pr", "qhei2pr", "qheii2pr", "lmech2pr", "emech2pr")
    ssec += "{:>11}  {:>11}  {:>11}  {:>11}  {:>11}\n".format("qhi2po", "qhei2po", "qheii2po", "lmech2po", "emech2po")
    ssec += "#{:>5}  ".format("")
    ssec += "{:>11}  {:>11}  {:>11}  {:>11}  {:>11}  ".format("phot/sec", "phot/sec", "phot/sec", "erg/sec", "erg")
    ssec += "{:>11}  {:>11}  {:>11}  {:>11}  {:>11}  ".format("phot/sec", "phot/sec", "phot/sec", "erg/sec", "erg")
    ssec += "{:>11}  {:>11}  {:>11}  {:>11}  {:>11}\n".format("phot/sec", "phot/sec", "phot/sec", "erg/sec", "erg")
    ssec += "#{:>5}  ".format("(1)")
    ssec += "{:>11}  {:>11}  {:>11}  {:>11}  {:>11}  ".format("(2)" , "(3)" , "(4)" , "(5)" , "(6)" )
    ssec += "{:>11}  {:>11}  {:>11}  {:>11}  {:>11}  ".format("(7)" , "(8)" , "(9)" , "(10)", "(11)")
    ssec += "{:>11}  {:>11}  {:>11}  {:>11}  {:>11}\n".format("(12)", "(13)", "(14)", "(15)", "(16)")

    fh_pri.write(spri)
    fh_sec.write(ssec)

    for ireal in range(len(o1_ebv)):
        d1 = [ o1_n  [ireal] * 10.**d for d in syndat.gen_sec_from_splines(o1_z  [ireal], o1_a  [ireal]) ]
        d2 = [ o2pr_n[ireal] * 10.**d for d in syndat.gen_sec_from_splines(o2pr_z[ireal], o2pr_a[ireal]) ]
        d3 = [ o2po_n[ireal] * 10.**d for d in syndat.gen_sec_from_splines(o2po_z[ireal], o2po_a[ireal]) ]
        o1_qhi  , o1_qhei  , o1_qheii  , o1_lmech  , o1_emech   = d1[0], d1[1], d1[2], d1[3], d1[4] 
        o2pr_qhi, o2pr_qhei, o2pr_qheii, o2pr_lmech, o2pr_emech = d2[0], d2[1], d2[2], d2[3], d2[4] 
        o2po_qhi, o2po_qhei, o2po_qheii, o2po_lmech, o2po_emech = d3[0], d3[1], d3[2], d3[3], d3[4] 

        spri  = "{:6d}  {:1.5f}  {:1.5f}  {:1.5e}  {:1.5f}".format(ireal, o1_a[ireal], o1_z[ireal], o1_n[ireal], o1_ebv[ireal])
        spri += "  {:1.5f}  {:1.5f}  {:1.5e}  {:1.5f}"     .format(o2pr_a[ireal], o2pr_z[ireal], o2pr_n[ireal], o2pr_ebv[ireal])
        spri += "  {:1.5f}  {:1.5f}  {:1.5e}  {:1.5f}\n"   .format(o2po_a[ireal], o2po_z[ireal], o2po_n[ireal], o2po_ebv[ireal])

        ssec  = "{:6d}  ".format(ireal)
        ssec += "{:1.5e}  {:1.5e}  {:1.5e}  {:1.5e}  {:1.5e}  ".format(o1_qhi  , o1_qhei  , o1_qheii  , o1_lmech  , o1_emech  )
        ssec += "{:1.5e}  {:1.5e}  {:1.5e}  {:1.5e}  {:1.5e}  ".format(o2pr_qhi, o2pr_qhei, o2pr_qheii, o2pr_lmech, o2pr_emech)
        ssec += "{:1.5e}  {:1.5e}  {:1.5e}  {:1.5e}  {:1.5e}\n".format(o2po_qhi, o2po_qhei, o2po_qheii, o2po_lmech, o2po_emech)

        fh_pri.write(spri)
        fh_sec.write(ssec)
    
    
    fh_pri.close()
    fh_sec.close()

    return  o1_qhi  , o2pr_qhi  , o2po_qhi  ,  \
        o1_qhei , o2pr_qhei , o2po_qhei ,  \
        o1_qheii, o2pr_qheii, o2po_qheii,  \
        o1_lmech, o2pr_lmech, o2po_lmech,  \
        o1_emech, o2pr_emech, o2po_emech
