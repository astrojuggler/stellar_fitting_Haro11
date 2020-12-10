# This script estimates the flux errors of COS 1d extracted spectra 
# by calculating the local (window of 50 px) standard deviation

# This is needed because the COS pipeline overestimates the errors for low S/N (<10)sources

# The script then creates txt files with the fixed-errors spectra

from astropy.io import fits
import numpy as np
import scipy as sp 
import matplotlib.pyplot as plt

import glob
import os

# I iterate the error correction over all the directories and files of the CLUES spectra
# The COS 1d extracted spectra have always the suffix "x1dsum"

files = glob.glob('*coarse.ascii*', recursive=True)
for filePath in files:
    
    lam, flam, dflam, dq = np.loadtxt(filePath, unpack=True)

    fnOut = filePath.replace('.ascii', '_errfix_trueNoise.txt') #name of the output file
    
    NpixWind = 50 #spectral window width over which std will be computed

    head  = "# Wavelenght Flux Error Data_Quality\n" #first line of the output file

    NpixSpec = len(lam) #number of spectral bins

    fhOut = open(fnOut, 'w') #output file
    fhOut.write(head)

    fhOut.write('#  error vector has been scaled to appxorximate local rms\n')

    # define regions that are not bright geocoronal lines, absorption lines,
    # or region where the error is not formally zero and the pixel worthless
    inonz = (dflam>0.) & ((lam<1213) | (1219<lam)) & ((lam<1300) | (1306<lam))
    fclo, fmed, fchi = sp.percentile(flam[inonz], [1., 50., 99.]) #values below which you have 2, 50, 99.9 % of the data
    iclip = (fclo < flam) & (flam < fchi)
    imeas = inonz & iclip
    print (fclo, fmed, fchi)

    # compute the wavelength-specific standard deviation over a window of NpixWind
    stdvec = sp.zeros_like(flam)

    for ipix in range(NpixSpec):
        ilo = sp.maximum(0,ipix-int(NpixWind/2))
        ihi = sp.minimum(NpixSpec,ipix+int(NpixWind/2))

        flamw = flam [ilo:ihi]
        iw    = imeas[ilo:ihi]
        stdvec[ipix] = flamw[iw].std()

    # fit polynomials to the standard deviation vector and the pipeline error vector
    polyord = 7

    polyvalStd1  = sp.polyfit(lam[imeas], stdvec[imeas], polyord)
    stdVecNorm   = stdvec / sp.polyval(polyvalStd1, lam)
    iStdTake     = stdVecNorm < 2
    polyvalStd2  = sp.polyfit(lam[imeas&iStdTake], stdvec[imeas&iStdTake], polyord)

    iPipeTake    = 1130<lam
    polyvalPipe1 = sp.polyfit(lam[imeas&iPipeTake], dflam [imeas&iPipeTake],
    polyord)

    vecPolyStd   = sp.polyval(polyvalStd2 , lam)
    vecPolyPipe  = sp.polyval(polyvalPipe1, lam)

    vecPolyScale = vecPolyStd / vecPolyPipe
    vecPolyScale[vecPolyScale<0.] = 1. 

    dflamScale = dflam * vecPolyScale #new estimated error

    #dflamScale = dflam / sp.polyval(polyvalPipe, lam) * sp.polyval(polyvalStd , lam)

    for ipix in range(NpixSpec):
        s = '   {:4.4f}  {:1.6e}  {:1.6e}  {:d}\n'.format(lam[ipix], flam[ipix], dflamScale[ipix], int(dq[ipix]))
        fhOut.write(s)

    s = '#\n'
    s+= '# file {:s} fixed to {:s}\n'.format(filePath, fnOut)
    s+= '# fraction of total pixels with error==0: {:1.5f}\n'.format( (dflam == 0.).sum() / len(dflam))
    s+= '# minimum error value: {:1.6g}\n'.format(dflam.min())

    print (s)
    fhOut.write(s)

    fhOut.close() 
    
    #plotting the estimated error compared to the pipeline error
    plt.figure()
    plotName =  fnOut = filePath.replace('.ascii', '_err_OLDvsNEW_plot.pdf') #name of the plot file
    plt.plot(lam, dflam, label='pipeline error')
    plt.plot(lam, dflamScale, label='estimated error')
    plt.legend()
    plt.ylim(0, 1.5*dflam[imeas].max())
    plt.savefig(plotName, format='pdf')
    
    #plotting the estimated error compared to flux (noise and signal)
    plt.figure()
    plotName =  fnOut = filePath.replace('.ascii', '_ERRvsFLUX_plot.pdf') #name of the plot file
    plt.plot(lam, flam, label='flux')
    plt.plot(lam, dflamScale, label='estimated error')
    plt.legend()
    plt.ylim(0, 1.5*flam[imeas].max())
    plt.savefig(plotName, format='pdf')
    
    
