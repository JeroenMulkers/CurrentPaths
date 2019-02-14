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
        self.sigma = sigma*np.ones((ny,nx))
        self.pot = np.nan*np.ones((ny,nx))

    def setPotential(self,U,regionFunc):
        for ix in range(self.nx):
            for iy in range(self.ny):
                if regionFunc(self.xgrid[iy,ix],self.ygrid[iy,ix]):
                    self.pot[iy,ix] = U
                    
    def setSigma(self,sigma,regionFunc=lambda x,y: True):
        for ix in range(self.nx):
            for iy in range(self.ny):
                if regionFunc(self.xgrid[iy,ix],self.ygrid[iy,ix]):
                    self.sigma[iy,ix] = sigma

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
        phi = phi_RAW.reshape(ny,nx)

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
        plt.figure(figsize=(10,10))
        plt.grid(False)
        plt.imshow(self.sigma,origin='lower',extent=self.extent)
        plt.show()

    def showCurrent(self,quiverColor=None,streamplotColor=None):
        jx,jy = self.calcCurrent()
        j     = np.sqrt(jx**2+jy**2)

        appliedPotRegion = np.logical_not(np.isnan(self.pot))
        j[appliedPotRegion] = np.nan
        jx[appliedPotRegion] = np.nan
        jy[appliedPotRegion] = np.nan

        cmap = plt.cm.magma
        cmap.set_bad('gray',0.8)
        
        plt.figure(figsize=(10,10))
        plt.grid(False)
        plt.imshow(j,cmap=cmap,vmin=0,origin='lower',extent=self.extent)
        if streamplotColor:
            plt.streamplot(self.xgrid,self.ygrid,jx,jy,color=streamplotColor,linewidth=1)
        if quiverColor:
            plt.quiver(self.xgrid,self.ygrid,jx,jy,color=quiverColor)
        plt.show()


class PoissonAnisotropic():

    def __init__(self, nx, ny, xrange=[0.,1.], yrange=[0.,1.], sigma=[0,0,0]):
        self.nx = nx
        self.ny = ny
        self.dx = (xrange[1]-xrange[0])/nx
        self.dy = (yrange[1]-yrange[0])/ny
        self.extent = [xrange[0],xrange[1],yrange[0],yrange[1]]
        x = np.linspace(xrange[0]+self.dx/2,xrange[1]-self.dx/2,nx)
        y = np.linspace(yrange[0]+self.dy/2,yrange[1]-self.dy/2,ny)
        self.xgrid, self.ygrid = np.meshgrid(x,y)
        self.sigma = np.zeros((ny,nx,3))
        self.sigma[:,:,0:2] = sigma
        self.pot = np.nan*np.ones((ny,nx))

    def setPotential(self,U,regionFunc):
        for ix in range(self.nx):
            for iy in range(self.ny):
                if regionFunc(self.xgrid[iy,ix],self.ygrid[iy,ix]):
                    self.pot[iy,ix] = U
                    
    def setSigma(self,sigma,regionFunc=lambda x,y: True):
        for ix in range(self.nx):
            for iy in range(self.ny):
                if regionFunc(self.xgrid[iy,ix],self.ygrid[iy,ix]):
                    self.sigma[iy,ix] = sigma

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
                if sigma[iy,ix,0] == 0.0:
                    Coef[I,I] = 1
                    continue


                IW, IE, IS, IN = (ix-1,iy), (ix+1,iy), (ix,iy-1), (ix,iy+1)
                ISW, INW, ISE, INE = (ix-1,iy-1), (ix-1,iy+1), (ix+1,iy-1), (ix+1,iy+1)

                # xx

                ix_,iy_ = (ix-1,iy) if ix>0 else (ix,iy)
                I_ = iy_*nx+ix_ # vectorized index of neighbor
                sigma_ = geomMean(sigma[iy,ix,0], sigma[iy_,ix_,0]) 
                Coef[I,I]  -= sigma_/self.dx**2
                Coef[I,I_] += sigma_/self.dx**2

                ix_,iy_ = (ix+1,iy) if ix<nx-1 else (ix,iy)
                I_ = iy_*nx+ix_ # vectorized index of neighbor
                sigma_ = geomMean(sigma[iy,ix,0], sigma[iy_,ix_,0]) 
                Coef[I,I]  -= sigma_/self.dx**2
                Coef[I,I_] += sigma_/self.dx**2

                # yy

                ix_,iy_ = (ix,iy-1) if iy>0 else (ix,iy)
                I_ = iy_*nx+ix_ # vectorized index of neighbor
                sigma_ = geomMean(sigma[iy,ix,1], sigma[iy_,ix_,1]) 
                Coef[I,I]  -= sigma_/self.dy**2
                Coef[I,I_] += sigma_/self.dy**2

                ix_,iy_ = (ix,iy+1) if iy<ny-1 else (ix,iy)
                I_ = iy_*nx+ix_ # vectorized index of neighbor
                sigma_ = geomMean(sigma[iy,ix,1], sigma[iy_,ix_,1]) 
                Coef[I,I]  -= sigma_/self.dy**2
                Coef[I,I_] += sigma_/self.dy**2

                # xy

                ix_,iy_ = (ix-1,iy) if ix>0 else (ix,iy)
                iy__ = iy-1 if iy > 0 else iy
                I_ = iy_*nx+ix_ # vectorized index of neighbor
                sigma_ = geomMean(sigma[iy,ix,1], sigma[iy_,ix_,1]) 
                
                ## loop over neighbours
                #for ix_,iy_ in [ (ix-1,iy), (ix+1,iy), (ix,iy-1), (ix,iy+1) ]:

                #    # check if neighbor exists
                #    if ix_ >= 0 and iy_ >= 0 and ix_ < nx and iy_ < ny: 

                #        I_ = iy_*nx+ix_ # vectorized index of neighbor

                #        h = abs(ix-ix_)*self.dx + abs(iy-iy_)*self.dy 

                #        sigma_ = geomMean(sigma[iy,ix,0], sigma[iy_,ix_,0]) 

                #        Coef[I,I]  -= sigma_/h**2
                #        Coef[I,I_] += sigma_/h**2
        
        # solve the system of equations
        phi_RAW = spsolve(Coef.tocsr(),rhs)
        phi = phi_RAW.reshape(ny,nx)

        return phi

    def calcCurrent(self):
        sigma = self.sigma
        phi = self.calcPotential()
        jx = 0*sigma[:,:,0]
        jy = 0*sigma[:,:,0]
    
        shft = [1,0]
        sigma_ = geomMean(sigma[:,:,1], shift(sigma[:,:,1],shft))
        dphi_ = phi - shift(phi,shft)
        jy += sigma_*dphi_
    
        shft = [-1,0]
        sigma_ = geomMean(sigma[:,:,1], shift(sigma[:,:,1],shft))
        dphi_ = phi - shift(phi,shft)
        jy -= sigma_*dphi_
    
        shft = [0,1]
        sigma_ = geomMean(sigma[:,:,0], shift(sigma[:,:,0],shft))
        dphi_ = phi - shift(phi,shft)
        jx += sigma_*dphi_
    
        shft = [0,-1]
        sigma_ = geomMean(sigma[:,:,0], shift(sigma[:,:,0],shft))
        dphi_ = phi - shift(phi,shft)
        jx -= sigma_*dphi_



    
        return jx,jy

    def showPotential(self):
        pass

    def showSigma(self):
        plt.figure(figsize=(10,10))
        plt.grid(False)
        plt.imshow(self.sigma,origin='lower',extent=self.extent)
        plt.show()

    def showCurrent(self,quiverColor=None,streamplotColor=None):
        jx,jy = self.calcCurrent()
        j     = np.sqrt(jx**2+jy**2)

        appliedPotRegion = np.logical_not(np.isnan(self.pot))
        j[appliedPotRegion] = np.nan
        jx[appliedPotRegion] = np.nan
        jy[appliedPotRegion] = np.nan

        cmap = plt.cm.magma
        cmap.set_bad('gray',0.8)
        
        plt.figure(figsize=(10,10))
        plt.grid(False)
        plt.imshow(j,cmap=cmap,vmin=0,origin='lower',extent=self.extent)
        if streamplotColor:
            plt.streamplot(self.xgrid,self.ygrid,jx,jy,color=streamplotColor,linewidth=1)
        if quiverColor:
            plt.quiver(self.xgrid,self.ygrid,jx,jy,color=quiverColor)
        plt.show()


def example():
    p = PoissonAnisotropic( nx=140, ny=100, xrange=[-0.7,0.7], yrange=[0.,1.], sigma=1.)
    
    # Let's use the same contact points as in example 1
    region = lambda x,y: -0.45<x<-0.40 and 0.4<y<0.6
    p.setPotential(regionFunc=region, U=1.0)
    region = lambda x,y:  0.40<x< 0.45 and 0.4<y<0.6
    p.setPotential(regionFunc=region, U=-1.0)
    
    # Create rectangular region in the center with a low conductivity
    # (you can also set it to zero to have an isolating material at the center)
    sigmaCenter = [0.7,0,0.1]
    region = lambda x,y: -0.1<x<0.1 and 0.3<y<0.7
    p.setSigma(regionFunc=region, sigma=sigmaCenter)
    
    p.showCurrent(quiverColor='black',streamplotColor='gray')
