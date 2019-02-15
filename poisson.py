import numpy as np
from scipy.ndimage import shift
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import spsolve
import matplotlib.pyplot as plt
import time

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

    ############################################################################

    def __init__(self, nx, ny, xrange=[0.,1.], yrange=[0.,1.], sigma=[0,0,0,0]):
        self.nx = nx
        self.ny = ny
        self.dx = (xrange[1]-xrange[0])/nx
        self.dy = (yrange[1]-yrange[0])/ny
        self.extent = [xrange[0],xrange[1],yrange[0],yrange[1]]
        x = np.linspace(xrange[0]+self.dx/2,xrange[1]-self.dx/2,nx)
        y = np.linspace(yrange[0]+self.dy/2,yrange[1]-self.dy/2,ny)
        self.xgrid, self.ygrid = np.meshgrid(x,y)
        self.sigma = np.zeros((ny,nx,4))
        self.sigma[:,:,0:4] = sigma
        self.pot = np.nan*np.ones((ny,nx))
        self.phi = None

    ############################################################################

    def setPotential(self,U,regionFunc):
        for ix in range(self.nx):
            for iy in range(self.ny):
                if regionFunc(self.xgrid[iy,ix],self.ygrid[iy,ix]):
                    self.pot[iy,ix] = U
                    
    ############################################################################

    def setSigma(self,sigma,regionFunc=lambda x,y: True):
        for ix in range(self.nx):
            for iy in range(self.ny):
                if regionFunc(self.xgrid[iy,ix],self.ygrid[iy,ix]):
                    self.sigma[iy,ix] = sigma

    ############################################################################

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

                # helper function to set the coefficients
                def addCoeffDiff(a,ix1,iy1,ix2,iy2,sigmacomp):
                    if 0 > iy1 or iy1 > ny-1 or 0 > ix1 or ix1 > nx-1: return
                    if 0 > iy2 or iy2 > ny-1 or 0 > ix2 or ix2 > nx-1: return
                    sigma_ = (sigma[iy1,ix1,sigmacomp] + sigma[iy2,ix2,sigmacomp])/2
                    h = self.dy*(iy2-iy1) + self.dx*(ix2-ix1)
                    Coef[I,iy1*nx+ix1] -= a*sigma_/h
                    Coef[I,iy2*nx+ix2] += a*sigma_/h

                # from left
                if ix>0:
                    # isotropic
                    addCoeffDiff( self.dy,   ix, iy, ix-1, iy , 0)
                    # anisotropic
                    addCoeffDiff( self.dy/4, ix,   iy, ix,   iy+1 , 2 )
                    addCoeffDiff( self.dy/4, ix,   iy, ix,   iy-1 , 2 )
                    addCoeffDiff( self.dy/4, ix-1, iy, ix-1, iy+1 , 2 )
                    addCoeffDiff( self.dy/4, ix-1, iy, ix-1, iy-1 , 2 )

                # from right
                if ix<nx-1:
                    # isotropic
                    addCoeffDiff(-self.dy, ix, iy, ix+1, iy , 0)
                    # anisotropic
                    addCoeffDiff(-self.dy/4, ix,   iy, ix,   iy+1 , 2 )
                    addCoeffDiff(-self.dy/4, ix,   iy, ix,   iy-1 , 2 )
                    addCoeffDiff(-self.dy/4, ix+1, iy, ix+1, iy+1 , 2 )
                    addCoeffDiff(-self.dy/4, ix+1, iy, ix+1, iy-1 , 2 )

                # from below
                if iy>0:
                    # isotropic
                    addCoeffDiff( self.dx, ix, iy-1, ix, iy , 1)
                    # anisotropic
                    addCoeffDiff( self.dx/4, ix, iy,   ix+1, iy   , 3 )
                    addCoeffDiff( self.dx/4, ix, iy,   ix-1, iy   , 3 )
                    addCoeffDiff( self.dx/4, ix, iy-1, ix+1, iy-1 , 3 )
                    addCoeffDiff( self.dx/4, ix, iy-1, ix-1, iy-1 , 3 )

                # from above
                if iy<ny-1:
                    # isotropic
                    addCoeffDiff(-self.dx, ix, iy+1, ix, iy , 1)
                    # anisotropic
                    addCoeffDiff(-self.dx/4, ix, iy,   ix+1, iy   , 3 )
                    addCoeffDiff(-self.dx/4, ix, iy,   ix-1, iy   , 3 )
                    addCoeffDiff(-self.dx/4, ix, iy+1, ix+1, iy+1 , 3 )
                    addCoeffDiff(-self.dx/4, ix, iy+1, ix-1, iy+1 , 3 )

        # solve the system of equations
        Coef = Coef.tocsr()
        print("Solve system")
        ti = time.time()
        phi_RAW = spsolve(Coef,rhs)
        tf = time.time()
        print("System solved in %.2f ms"%(1000*(tf-ti)))
        phi = phi_RAW.reshape(ny,nx)
        return phi

    ############################################################################

    def calcCurrent(self,phi=None):
        sigma = self.sigma
        if phi is None:
            phi = self.calcPotential()
        jx = 0*sigma[:,:,0]
        jy = 0*sigma[:,:,0]
        dphidy,dphidx = np.gradient(phi,self.dy,self.dx)
        jx = sigma[:,:,0]*dphidx + sigma[:,:,2]*dphidy
        jy = sigma[:,:,1]*dphidy + sigma[:,:,3]*dphidx
        return jx,jy

    ############################################################################

    def solve(self):
        plt.figure(figsize=(15,8))

        phi   = self.calcPotential()
        jx,jy = self.calcCurrent(phi)
        j     = np.sqrt(jx**2+jy**2)
        appliedPotRegion = np.logical_not(np.isnan(self.pot))
        j[appliedPotRegion]  = np.nan
        jx[appliedPotRegion] = np.nan
        jy[appliedPotRegion] = np.nan

        plt.subplot(121,title='Potential')
        plt.imshow(phi,extent=self.extent,cmap = 'RdBu',origin='lower')
        plt.contourf(self.xgrid,self.ygrid,phi,20,cmap = 'RdBu')
        plt.streamplot(self.xgrid,self.ygrid,jx,jy,color='gray')

        plt.subplot(122,title='Current density')
        plt.grid(False)

        cmap = plt.cm.magma
        cmap.set_bad('gray',0.8)
        
        plt.imshow(j,cmap=cmap,vmin=0,origin='lower',extent=self.extent)
        plt.streamplot(self.xgrid,self.ygrid,jx,jy,color='gray')
        plt.quiver(self.xgrid,self.ygrid,jx,jy,color='k')

        plt.show()

        return phi,jx,jy


def example():
    p = PoissonAnisotropic( nx=70, ny=50, xrange=[-0.7,0.7], yrange=[0.,1.], sigma=[1,1,0.8,-0.8])
    
    # Let's use the same contact points as in example 1
    region = lambda x,y: -0.55<x<-0.5  and 0.2<y<0.8
    p.setPotential(regionFunc=region, U=1.0)
    region = lambda x,y:  0.5 <x<0.55 and 0.2<y<0.8
    p.setPotential(regionFunc=region, U=-1.0)
    
    #Create rectangular region in the center with a low conductivity
    #(you can also set it to zero to have an isolating material at the center)
    #sigmaCenter = [0.7,0,0.1]
    #region = lambda x,y: x<-0.65 or x >0.65 or y<0.1 or y >0.9
    #region = lambda x,y: -0.1 < x < 0.1 and 0.3 < y < 0.7
    #p.setSigma(regionFunc=region, sigma=[0.001,0.001,0.001,0.001])
    
    #p.showCurrent(quiverColor='black',streamplotColor='gray')
    p.solve()

#example()
