import numpy as np
import pandas as pd
import os
import subprocess
from astropy.io import fits

X = pd.read_csv('spectra0_qwsaz123.csv')

instrument = X['instrument'].values
mjd = X['mjd'].values
plate = X['plate'].values
fiber = X['fiberID'].values
run = X['run2d'].values
specobj = X['specObjID']

req_url= [] 
for i in range(60170,len(instrument)):
    req_url.append('http://dr12.sdss.org/sas/dr12/{}/spectro/redux/{}/spectra/{}/spec-{}-{}-{}.fits'.format(instrument[i].lower(),
                                                                                                                 str(run[i]),
                                                                                                                 str(plate[i]).zfill(4),
                                                                                                                 str(plate[i]).zfill(4),
                                                                                                                 str(mjd[i]),
                                                                                                                 str(fiber[i]).zfill(4)))
    out_path = '/home/awe2/scratch/Spectra_fits/'+str(specobj[i])+'.fits'
    subprocess.run(['wget', req_url[i], '-O', out_path, '-q'])
    
    with fits.open(out_path) as file:
        spectra = file[1].data['flux']
        if np.any(np.isnan(spectra)) == False:
            np.save('Spectra/' + str(specobj[i]) + '.npy',spectra)