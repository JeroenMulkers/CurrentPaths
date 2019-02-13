import numpy as np
from scipy.ndimage import shift
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import spsolve
import matplotlib.pyplot as plt

def geomMean(a,b):
    a,b = np.abs(a),np.abs(b)
    return np.sqrt(a*b)

class Poisson():

    def __init__(self, nx, ny, xrange=[0.,1.], yrange=[0.,1.], sigma=0):
        self.nx = nx
        self.ny = ny
        self.dx = (xrange[1]-xrange[0])/nx
        self.dy = (yrange[1]-yrange[0])/ny
        self.extent = [xrange[0],xrange[1],yrange[0],yrange[1]]
        x = np.linspace(xrange[0]+self.dx/2,xrange[1]-self.dx/2,nx)
        y = np.linspace(yrange[0]+self.dy/2,yrange[1]-self.dy/2,ny)
        self.xgrid, self.ygrid = np.meshgrid(x,y)
        self.sigma = sigma*np.ones((nx,ny))
        self.pot = np.nan*np.ones((nx,ny))

    def setPotential(self,U,regionFunc):
        for ix in range(self.nx):
            for iy in range(self.ny):
                if regionFunc(self.xgrid[ix,iy],self.ygrid[ix,iy]):
                    self.pot[ix,iy] = U
                    
    def setSigma(self,sigma,regionFunc=lambda x,y: True):
        for ix in range(self.nx):
            for iy in range(self.ny):
                if regionFunc(self.xgrid[ix,iy],self.ygrid[ix,iy]):
                    self.sigma[ix,iy] = sigma

    def calcPotential(self):

        nx,ny = self.nx,self.ny
        N = nx*ny
        sigma = self.sigma
        
        # loop over all cells to construct system of equations

        Coef = lil_matrix((N,N))
        rhs = np.zeros(N)

        for iy in range(ny):
            for ix in range(nx):

                I = iy*nx+ix # vectorized index

                # set Dirichlet BC on cells with a not nan potential
                if not np.isnan(self.pot[iy,ix]):
                    Coef[I,I] = 1
                    rhs[I] = self.pot[iy,ix]
                    continue

                # don't calculate the potential in isolator regions
                if sigma[iy,ix] == 0.0:
                    Coef[I,I] = 1
                    continue

                
                # loop over neighbours
                for ix_,iy_ in [ (ix-1,iy), (ix+1,iy), (ix,iy-1), (ix,iy+1) ]:

                    # check if neighbor exists
                    if ix_ >= 0 and iy_ >= 0 and ix_ < nx and iy_ < ny: 

                        I_ = iy_*nx+ix_ # vectorized index of neighbor

                        h = abs(ix-ix_)*self.dx + abs(iy-iy_)*self.dy 

                        sigma_ = geomMean(sigma[iy,ix], sigma[iy_,ix_]) 

                        Coef[I,I]  -= sigma_/h**2
                        Coef[I,I_] += sigma_/h**2
        
        # solve the system of equations
        phi_RAW = spsolve(Coef.tocsr(),rhs)
        phi = phi_RAW.reshape(nx,ny)

        return phi

    def calcCurrent(self):
        sigma = self.sigma
        phi = self.calcPotential()
        jx = 0*sigma
        jy = 0*sigma
    
        shft = [1,0]
        sigma_ = geomMean(sigma, shift(sigma,shft))
        dphi_ = phi - shift(phi,shft)
        jy += sigma_*dphi_
    
        shft = [-1,0]
        sigma_ = geomMean(sigma, shift(sigma,shft))
        dphi_ = phi - shift(phi,shft)
        jy -= sigma_*dphi_
    
        shft = [0,1]
        sigma_ = geomMean(sigma, shift(sigma,shft))
        dphi_ = phi - shift(phi,shft)
        jx += sigma_*dphi_
    
        shft = [0,-1]
        sigma_ = geomMean(sigma, shift(sigma,shft))
        dphi_ = phi - shift(phi,shft)
        jx -= sigma_*dphi_
    
        return jx,jy

    def showPotential(self):
        pass

    def showSigma(self):
        plt.imshow(self.sigma,origin='lower',extent=self.extent)
        plt.show()

    def showCurrent(self):
        jx,jy = self.calcCurrent()
        j     = np.sqrt(jx**2+jy**2)

        appliedPotRegion = np.logical_not(np.isnan(self.pot))
        j[appliedPotRegion] = np.nan
        jx[appliedPotRegion] = np.nan
        jy[appliedPotRegion] = np.nan

        cmap = plt.cm.magma
        cmap.set_bad('gray',0.8)
        
        plt.figure(figsize=(10,10))
        plt.imshow(j,cmap=cmap,vmin=0,origin='lower',extent=self.extent)
        plt.quiver(self.xgrid,self.ygrid,jx,jy)
        plt.show()



