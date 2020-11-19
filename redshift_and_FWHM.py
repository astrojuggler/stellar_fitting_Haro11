""" This is a module to use for the FUV analysis of the CLUES sample.

Useful for deriving redshift and braodening from a photospheric absorption line. This is possible by fitting the line with a gaussian.

"""

import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import math


def profile(x, constant, ctr, amp, wid):
    """Model of the photospheric line.
    
    The line is modeled as a constant continuum minus a gaussian to account for the absorption.
    
    Parameters
    -----------
    x : float
        variable of the function (wavelength)
    constant: float
              continuum level
    ctr: float
         centre of the gaussian
    amp: float
         amplitude of the gaussian
    wid: float
         width of the gaussian (sigma)
    
    Returns
    -----------
    y : float
        model evaluated at x
    """
    
    y = constant - amp * np.exp( -((x-ctr)/wid)**2)
    return y

def plot_and_print_z_and_FWHM(olam, flux, m1, m2, m3, m4, line):
    """Calculates and prints the redshift and the stellar lines broadening.
    
    Takes as input the observed spectrum (wavelengths and fluxes) and the intervals of the masks.
    
    Parameters
    -----------
    olam : ndarray
        wavelengths
    flux: ndarray
        observed spectrum
    m1: float
        left boundary of mask for the continuum at the blue side of the line
    m2: float
        right boundary of mask for the continuum at the blue side of the line
    m3: float
        left boundary of mask for the continuum at the red side of the line
    m4: float
        right boundary of mask for the continuum at the blue side of the line
    line: string 
          specifying the photospheric line to fit, options are CIII1247 or CIII1776
    
    Returns
    ---------
    z: float
       redshift best-fit value
    dz: float
        redshift error
    FWHM: float
          broadening best-fit value 
    dFWHM: float
           broadening error
    """
    
    if len(olam) != len(flux):
        raise Exception("Error: check the input spectrum, size of wavelenghts different from size of fluxes")

    if line == "CIII1247":
        ref = 1247.4
    elif line == "CIII1776":
        ref = 1175.53
    else:
        raise Exception("Error: this is not a valid photospheric line, please choose between CIII1247 and CIII1776")
    
    # I determine the guess of the constant averaging the spectrum at the sides of the absorpiton line
    const_mask_blue = (olam > m1) & (olam < m2)
    const_mask_red = (olam > m3) & (olam < m4)
    const_mask = const_mask_red + const_mask_blue
    fit_mask = (olam > m2) & (olam < m3)
    const = np.mean(flux[const_mask])
    tot_mask = const_mask + fit_mask


    #fitting and plotting results
    guess = [const, np.average([m2,m3]), 3e-14, 2]
    popt, pcov = curve_fit(profile, olam[tot_mask], flux[tot_mask], p0=guess)
    fit = profile(olam[tot_mask], *popt)
    plt.figure()
    plt.plot(olam[tot_mask], flux[tot_mask])
    plt.plot(olam[tot_mask], fit)

    #printing estimated redshift and broadening
    z = popt[1]/ref - 1
    dz = np.sqrt(pcov[1,1]) / ref
    print("Redshift measurement with error:")
    print(z, dz)
    
    FWHM = math.fabs(popt[3])
    dFWHM = np.sqrt(pcov[3,3])
    print("FWHM and error:")
    print(FWHM, dFWHM)
    
    return z, dz, FWHM, dFWHM
    
def update_setup_file(path, z, FWHM, line):
    """Updates the setup file for fitting a cluster.
    
    Takes as input the path to the file to be edited and changes the redshift and FWHM values.
    
    Parameters
    -------------
    path: string
          path to the file to be edited
    z: float
       redshift value to update
    FWHM: float
          broadening value to update
    line: string 
          specifying the photospheric line to fit, options are CIII1247 or CIII1776
          
    """
    
    setup_f = open(path, 'r')

    lines = setup_f.readlines()

    if line == "CIII1247":
        pass
    elif line == "CIII1776":
        pass
    else:
        raise Exception("Error: this is not a valid photospheric line, please choose between CIII1247 and CIII1776")


    for i in range(len(lines)):
        if lines[i][0:6] == 'FWHMOP':
            lines[i] = "FWHMOP          " + "{:.2f}".format(FWHM) + "   # G130M-1291 grating  (effective resolution measured from the width of" + line + "photospheric line)\n"
        if lines[i][0:6] == 'FWHMUV':
            lines[i] = "FWHMUV          " + "{:.2f}".format(FWHM) + "   # G130M-1291 grating  (effective resolution measured from the width of CIII1247 photospheric line)\n"
        if lines[i][0:8] == 'REDSHIFT':
            lines[i] = "REDSHIFT        " + "{:.6f}".format(z) + "    # measured from the Doppler shift of CIII1247 photospheric line)\n"
        
    setup_f = open(path, 'w')
    setup_f.writelines(lines)
    setup_f.close()
    
    
    
    
    
    
    
