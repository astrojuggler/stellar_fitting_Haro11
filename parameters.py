import os
import sys
import scipy as sp 
from astroconv import z2area
from filehandle import ReadStripTok as rst

class params: 

    def __init__(self, args):
  
        if (len(args) != 2) and (len(args) != 3): 
            print ("!!!!error!!!!  USAGE:")
            print ("   ", sys.argv[0], "parameter_file.pars")
            sys.exit(1)
        
        self.fnPars   = sys.argv[1]
        self.homePath = os.environ['HOME']
        
        print ("reading parameter setup file", self.fnPars)
        print ("homepath = ", self.homePath)
        
        parList = rst(self.fnPars)
        Npars = len(parList)
        
        self.windows = {}
        self.manmask = False
        
        for iPar in range(Npars):
            par  = parList[iPar]
            lPar = par[0]
            rPar = par[1]
            
            if   lPar == "NOVERSAMPCOS" : self.NoversampCos  = int(rPar)
            elif lPar == "NEWBIN"       : self.newBin        = float(rPar)
            elif lPar == "FWHMUV130"    : self.fwhmUv130     = float(rPar)
            elif lPar == "FWHMUV160"    : self.fwhmUv160     = float(rPar)
            elif lPar == "FWHMOP"       : self.fwhmOp        = float(rPar)
            elif lPar == "MAXSNR"       : self.maxSnr        = float(rPar)
            
            elif lPar == "FNOBSUV130"      : 
                self.fnObsUv130 = rPar
                if len(par) == 3: self.scaleErrUv130 = float(par[2])
                else            : self.scaleErrUv130 = 1.
            elif lPar == "FNOBSUV160"      : 
                self.fnObsUv160 = rPar
                if len(par) == 3: self.scaleErrUv160 = float(par[2])
                else            : self.scaleErrUv160 = 1.
            elif lPar == "FNOBSOP"      : 
                self.fnObsOp = rPar
                if len(par) == 3: self.scaleErrOp = float(par[2])
                else            : self.scaleErrOp = 1.

            elif lPar == "RUNNAME"      : self.runName       = rPar
            elif lPar == "REDSHIFT"     : self.redshift      = float(rPar)
            elif lPar == "EBVMW"        : self.ebvMw         = float(rPar)
            
            elif lPar == "PATH2MODELS"  : self.path2models   = rPar
            elif lPar == "SFH"          : self.sfh           = rPar
            elif lPar == "TRACKS"       : self.tracks        = rPar
            elif lPar == "METALLO"      : self.metalLo       = float(rPar)
            elif lPar == "METALHI"      : self.metalHi       = float(rPar)
            elif lPar == "METALVALS"    : 
               self.metalstrs = par[1:]
               self.metalvals = sp.array([ float(l) for l in self.metalstrs ])
            
            elif lPar == "LOGAGELO"     : self.logAgeLo      = float(rPar)
            elif lPar == "LOGAGEHI"     : self.logAgeHi      = float(rPar)
            
            elif lPar == "EBVLO"        : self.ebvLo         = float(rPar)
            elif lPar == "EBVHI"        : self.ebvHi         = float(rPar)
            
            elif lPar == "FWHMLO"       : self.fwhmLo        = float(rPar)
            elif lPar == "FWHMHI"       : self.fwhmHi        = float(rPar)
            elif lPar == "NFWHM"        : self.Nfwhm         = int(rPar)
            
            elif lPar == "NSTEP"        : self.Nstep         = int(rPar)
            elif lPar == "NWALKER"      : self.Nwalker       = int(rPar)
            elif lPar == "NBURN"        : self.Nburn         = int(rPar)
            
            elif lPar == "WINDOW"       : self.windows[rPar] = par[2]
            elif lPar == "MANMASK"      : self.manmask       = rPar
            
            else:
                print ("!!!!error!!!!")
                print ("   ", lPar, "in", self.fnPars, "is not a valid parameter")
                sys.exit(1)
            
        self.LoverF = z2area(self.redshift)
       
        self.outRoot      = self.runName
        self.fnProps1     = self.outRoot+"_props1.txt"    
        self.fnProps2     = self.outRoot+"_props2.txt"    
        self.fnRes1       = self.outRoot+"_fitres1.txt"    
        self.fnRes2       = self.outRoot+"_fitres2.txt"
        self.fnSecHist1   = self.outRoot+"_sechist1.txt"    
        self.fnSecHist2   = self.outRoot+"_sechist2.txt"
        self.fnResSec     = self.outRoot+"_secres.txt"
        self.fnFigroot1   = self.outRoot+"_fitfig1"
        self.fnFigroot2   = self.outRoot+"_fitfig2"
        self.fnMcRes      = self.outRoot+"_mcres"
        self.fnFigPdfRoot = self.outRoot+"_pdffig"
        #MS classic monte carlo: file with store best fit values
        self.fnClassic    = self.outRoot+"_classic.txt"
        self.fnClassicDe  = self.outRoot+"_classic_de.txt"
        self.fnClassicErr = self.outRoot+"_classicErr.txt"
        #MS output file with final spectrum and mask
        self.fnMaskedSpectrum = self.outRoot+"_maskedSpectrum.txt"

