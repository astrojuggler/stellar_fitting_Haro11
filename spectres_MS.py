import numpy as np 
import sys
from spectres import spectres

# Function to call spectres in order to resample and combine gratings G130M and G160M
# spec-wavs must cover full range of resampling with a margin of 100px = 1A
def call_spectres(spec_wavs, spec_fluxes, resampling, spec_errs=None):
    if (spec_wavs.min() <= resampling.min())  & (resampling.max() <= spec_wavs.max()): 
        #then all is good, and call noramlly
        #print("ciao")
        spectres(spec_wavs, spec_fluxes, resampling, spec_errs=None)
    else:
        if resampling.max() > spec_wavs.max():
            # I should pad spec_wavs with the rest of G160
            dx = np.diff(spec_wavs)[-1]
            pad_lam = np.linspace(spec_wavs.max()+dx, resampling.max()+200*dx, int((resampling.max()+200*dx-spec_wavs.max())/dx))
            lamr = np.append(spec_wavs, pad_lam)
            padf = np.full_like(pad_lam, np.nan)
            flamr = np.append(spec_fluxes, padf)       
            print("padding 1 alright")
            if spec_errs is not None:
                pade = np.full_like(pad_lam, np.nan)
                spec_errs_r = np.append(spec_errs, pade)
                return spectres(lamr, flamr, resampling, spec_errs_r)
            else:
                return spectres(lamr, flamr, resampling, spec_errs=None)
        if resampling.min() < spec_wavs.min():
            # I should pad spec_wavs with the rest of G130
            dx = np.diff(spec_wavs)[0]
            print("dx is equal to:")
            print(dx)
            pad_lam = np.linspace(resampling.min()-200*dx, spec_wavs.min(), int((spec_wavs.min()-resampling.min()+200*dx)/dx), endpoint=False)
            lamr = np.append(pad_lam, spec_wavs)
            padf = np.full_like(pad_lam, np.nan)
            flamr = np.append(padf, spec_fluxes)
            print("padding 2 alright")
            if spec_errs is not None:
                pade = np.full_like(pad_lam, np.nan)
                spec_errs_r = np.append(pade, spec_errs)
                return spectres(lamr, flamr, resampling, spec_errs_r)
            else:
                return spectres(lamr, flamr, resampling, spec_errs=None)
    # implement spec_errs part...? Yes!

