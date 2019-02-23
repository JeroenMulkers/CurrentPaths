import numpy as np
from struct import unpack
from re import match

def readovf(filename):
    """ read ovf files as outputted by mumax3 """
    meta = {}
    with open(filename,'rb') as f:
        for line in iter(f.readline, b''):
            line = line.decode("utf-8")
            if line[0] == '#':
                mm = match('# (.*): (.*)',line)
                if mm:
                    key, value = mm.group(1), mm.group(2)
                    if key == 'Begin' and value == 'Data Binary 4':
                        nx = int(meta['xnodes'])
                        ny = int(meta['ynodes'])
                        nz = int(meta['znodes'])
                        ncomp = int(meta['valuedim'])
                        f.read(4) #nothing to read at start
                        data = np.frombuffer(f.read(4*ncomp*nx*ny*nz),dtype=np.float32)
                        data = data.reshape(nz,ny,nx,ncomp)
                    if key != 'begin' and key != 'end':
                        meta[key] = value
    return data, meta


def AMRconductivity(m, conductivity0 = 1.0, AMRratio = 0.0):
    """ calculate amr conductivity for the magnetization m """
    X,Y,Z = 0,1,2
    nz,ny,nx,_ = m.shape
    AMR = np.zeros((nz,ny,nx,3,3))
    f = (6*AMRratio)/(6+AMRratio)
    for a in [X,Y,Z]:
        AMR[:,:,:,a,a] += 1 + f/3 
        for b in [X,Y,Z]:
            AMR[:,:,:,a,b] -= f*m[:,:,:,a]*m[:,:,:,b]
    return conductivity0*AMR
