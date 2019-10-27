import time
import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import spsolve


class Poisson():

    """ Describes a generalized Poisson problem and solves the problem """

    def __init__(self, gridsize, xrange=[0.,1.], yrange=[0.,1.], conductivity=np.array([[0,0],[0,0]])):

        self.nx, self.ny = gridsize
        self.dx = (xrange[1]-xrange[0])/self.nx
        self.dy = (yrange[1]-yrange[0])/self.ny
        self.extent = [xrange[0],xrange[1],yrange[0],yrange[1]]
        self.x = np.linspace(xrange[0]+self.dx/2,xrange[1]-self.dx/2,self.nx)
        self.y = np.linspace(yrange[0]+self.dy/2,yrange[1]-self.dy/2,self.ny)
        self.xgrid, self.ygrid = np.meshgrid(self.x,self.y)

        self.conductivity = np.zeros((self.ny,self.nx,2,2))
        self.conductivity[:,:] = conductivity
        self.pot = np.nan*np.ones((self.ny,self.nx))


    def __setRegionValue(self, field, value, regionFunc):
        for ix in range(self.nx):
            for iy in range(self.ny):
                if regionFunc(self.xgrid[iy,ix], self.ygrid[iy,ix]):
                    field[iy,ix] = value


    def setPotential(self, U, regionFunc):
        self.__setRegionValue(self.pot, U, regionFunc)


    def setConductivity(self, conductivity, regionFunc=lambda x,y: True):
        self.__setRegionValue(self.conductivity, conductivity, regionFunc)


    def isAnisotropic(self):
        xyNonZero = np.count_nonzero(self.conductivity[:,:,0,1]) > 0 
        yxNonZero = np.count_nonzero(self.conductivity[:,:,1,0]) > 0 
        return xyNonZero or yxNonZero


    def conductivityBetween(self,ixa,iya,ixb,iyb):
        """ Mixed arithmetic and geometric average 
            This average is designed to have the following properties
            - works for mixed signs of a and b 
            - avg(a,b) = avg(b,a)
            - min(a,b) < avg(a,b) < max(a,b)
            - avg(a,b) -> 0 if a -> -b
            - avg(a,b) -> 0 if a -> 0 or b -> 0
        """
        Ca = self.conductivity[iya,ixa]
        Cb = self.conductivity[iyb,ixb]
        return np.sign(Ca+Cb)*np.sqrt(0.5*np.abs(Ca+Cb)*np.sqrt(np.abs(Ca*Cb)))


    def calcPotential(self):

        isAnisotropic = self.isAnisotropic() # avoid checking it every time
        nx,ny = self.nx,self.ny
        Ncells = nx*ny

        # loop over all cells to construct system of equations
        Coef = lil_matrix((Ncells,Ncells))
        rhs = np.zeros(Ncells)

        # set Dirichlet BC on cells with a not nan potential
        not_nan_values = np.argwhere(np.isnan(self.pot)-1)
        I = not_nan_values[:, 0]*nx + not_nan_values[:, 1]
        rhs[I] = self.pot[not_nan_values[:, 0], not_nan_values[:, 1]]
        Coef[I, I] = 1

        # don't calculate the potential in isolator regions
        nonzero = np.argwhere(self.conductivity == 0)
        Coef[nonzero[:, 0], nonzero[:, 1]] = 1

        for iy in range(ny):
            for ix in range(nx):

                I = iy*nx+ix # vectorized index
                ID = lambda x,y: (iy+y)*nx+(ix+x)

                # from the left
                if ix>0:
                    conductivity = self.conductivityBetween(ix,iy,ix-1,iy)
                    Cxx = conductivity[0,0] * self.dy/self.dx
                    Coef[I,ID(-1,0)] -= Cxx
                    Coef[I,ID( 0,0)] += Cxx
                    if isAnisotropic:
                        Cxy = (1/4)*conductivity[1,0] * self.dy/self.dy
                        if iy < ny-1:
                            Coef[I,ID( 0, 0)] -= Cxy
                            Coef[I,ID( 0, 1)] += Cxy
                            Coef[I,ID(-1, 0)] -= Cxy
                            Coef[I,ID(-1, 1)] += Cxy
                        if iy > 0:
                            Coef[I,ID( 0,-1)] -= Cxy
                            Coef[I,ID( 0, 0)] += Cxy
                            Coef[I,ID(-1,-1)] -= Cxy
                            Coef[I,ID(-1, 0)] += Cxy

                # from right
                if ix<nx-1:
                    conductivity = self.conductivityBetween(ix,iy,ix+1,iy)
                    Cxx = conductivity[0,0] *self.dy/self.dx
                    Coef[I,ID(0,0)]+= Cxx
                    Coef[I,ID(1,0)]-= Cxx
                    if isAnisotropic:
                        Cxy = (1/4)*conductivity[1,0] * self.dy/self.dy
                        if iy < ny-1:
                            Coef[I,ID( 0, 0)] += Cxy
                            Coef[I,ID( 0, 1)] -= Cxy
                            Coef[I,ID(+1, 0)] += Cxy
                            Coef[I,ID(+1, 1)] -= Cxy
                        if iy > 0:
                            Coef[I,ID( 0,-1)] += Cxy
                            Coef[I,ID( 0, 0)] -= Cxy
                            Coef[I,ID(+1,-1)] += Cxy
                            Coef[I,ID(+1, 0)] -= Cxy

                # from below
                if iy>0:
                    conductivity = self.conductivityBetween(ix,iy,ix,iy-1)
                    Cyy = conductivity[1,1] * self.dx/self.dy
                    Coef[I,ID(0,-1)] -= Cyy
                    Coef[I,ID(0, 0)] += Cyy
                    if isAnisotropic:
                        Cyx = (1/4)*conductivity[0,1] * self.dx/self.dx
                        if ix < nx-1:
                            Coef[I,ID( 0, 0)] -= Cyx
                            Coef[I,ID( 1, 0)] += Cyx
                            Coef[I,ID( 0,-1)] -= Cyx
                            Coef[I,ID( 1,-1)] += Cyx
                        if ix > 0:
                            Coef[I,ID(-1, 0)] -= Cyx
                            Coef[I,ID( 0, 0)] += Cyx
                            Coef[I,ID(-1,-1)] -= Cyx
                            Coef[I,ID( 0,-1)] += Cyx

                # from above
                if iy<ny-1:
                    conductivity = self.conductivityBetween(ix,iy,ix,iy+1)
                    Cyy = conductivity[1,1] * self.dx/self.dy
                    Coef[I,ID(0,0)] += Cyy
                    Coef[I,ID(0,1)] -= Cyy
                    if isAnisotropic:
                        Cyx = (1/4)*conductivity[0,1] * self.dx/self.dx
                        if ix < nx-1:
                            Coef[I,ID( 0, 0)] += Cyx
                            Coef[I,ID( 1, 0)] -= Cyx
                            Coef[I,ID( 0, 1)] += Cyx
                            Coef[I,ID( 1, 1)] -= Cyx
                        if ix > 0:
                            Coef[I,ID(-1, 0)] += Cyx
                            Coef[I,ID( 0, 0)] -= Cyx
                            Coef[I,ID(-1, 1)] += Cyx
                            Coef[I,ID( 0, 1)] -= Cyx

        # solve the system of equations
        Coef = Coef.tocsr()
        ti = time.time()
        phi_RAW = spsolve(Coef,rhs)
        tf = time.time()
        print("System solved in %.2f ms"%(1000*(tf-ti)))
        phi = phi_RAW.reshape(ny,nx)
        return phi


    def calcCurrent(self,phi=None):
        if phi is None:
            phi = self.calcPotential()

        nx,ny=self.nx,self.ny

        def fd(ix1,iy1,ix2,iy2):
            if 0 > iy1 or iy1 > ny-1 or 0 > ix1 or ix1 > nx-1: return 0.0
            if 0 > iy2 or iy2 > ny-1 or 0 > ix2 or ix2 > nx-1: return 0.0
            h = self.dy*(iy2-iy1) + self.dx*(ix2-ix1)
            return (phi[iy2,ix2] - phi[iy1,ix1])/h

        # current density buffer
        Jx = 0*phi
        Jy = 0*phi

        for iy in range(ny):
            for ix in range(nx):
                jx,jy = 0.0,0.0

                if ix>0:
                    conductivity = self.conductivityBetween(ix,iy,ix-1,iy)
                    DIV = 8 if ( 0 < iy < ny -1 ) else 4 # TODO: check why this is necessary?
                    jx += conductivity[0,0]*fd( ix   , iy , ix-1 , iy   )/2    # isotropic
                    jx += conductivity[1,0]*fd( ix-1 , iy , ix-1 , iy-1 )/DIV  # anisotropic
                    jx += conductivity[1,0]*fd( ix-1 , iy , ix-1 , iy+1 )/DIV
                    jx += conductivity[1,0]*fd( ix   , iy , ix   , iy-1 )/DIV
                    jx += conductivity[1,0]*fd( ix   , iy , ix   , iy+1 )/DIV

                if ix<nx-1:
                    conductivity = self.conductivityBetween(ix,iy,ix+1,iy)
                    DIV = 8 if ( 0 < iy < ny -1 ) else 4
                    jx += conductivity[0,0]*fd( ix   , iy , ix+1 , iy   )/2    # isotropic
                    jx += conductivity[1,0]*fd( ix   , iy , ix   , iy-1 )/DIV  # anisotropic
                    jx += conductivity[1,0]*fd( ix   , iy , ix   , iy+1 )/DIV
                    jx += conductivity[1,0]*fd( ix+1 , iy , ix+1 , iy-1 )/DIV
                    jx += conductivity[1,0]*fd( ix+1 , iy , ix+1 , iy+1 )/DIV

                if iy>0:
                    conductivity = self.conductivityBetween(ix,iy,ix,iy-1)
                    DIV = 8 if ( 0 < ix < nx-1 ) else 4
                    jy += conductivity[1,1]*fd( ix , iy   , ix   , iy-1 )/2    # isotropic
                    jy += conductivity[0,1]*fd( ix , iy-1 , ix-1 , iy-1 )/DIV  # anisotropic
                    jy += conductivity[0,1]*fd( ix , iy-1 , ix+1 , iy-1 )/DIV
                    jy += conductivity[0,1]*fd( ix , iy   , ix-1 , iy   )/DIV
                    jy += conductivity[0,1]*fd( ix , iy   , ix-1 , iy   )/DIV

                if iy<ny-1:
                    conductivity = self.conductivityBetween(ix,iy,ix,iy+1)
                    DIV = 8 if ( 0 < ix < nx-1 ) else 4
                    jy += conductivity[1,1]*fd( ix , iy   , ix   , iy+1 ) /2   # isotropic
                    jy += conductivity[0,1]*fd( ix , iy   , ix+1 , iy   )/DIV  # anisotropic
                    jy += conductivity[0,1]*fd( ix , iy   , ix+1 , iy   )/DIV
                    jy += conductivity[0,1]*fd( ix , iy+1 , ix-1 , iy+1 )/DIV
                    jy += conductivity[0,1]*fd( ix , iy+1 , ix+1 , iy+1 )/DIV

                Jx[iy,ix] = jx
                Jy[iy,ix] = jy

        return Jx,Jy

    ############################################################################

    def solve(self, visualize=True):
        phi   = self.calcPotential()
        jx,jy = self.calcCurrent(phi)
        j     = np.sqrt(jx**2+jy**2)
        appliedPotRegion = np.logical_not(np.isnan(self.pot))
        j[appliedPotRegion]  = np.nan
        jx[appliedPotRegion] = np.nan
        jy[appliedPotRegion] = np.nan

        if visualize:
            cmap = plt.cm.magma
            cmap.set_bad('gray',0.8)
            plt.figure(figsize=(15,8))
            plt.grid(False)
            plt.imshow(j,cmap=cmap,vmin=0,origin='lower',extent=self.extent)
            plt.streamplot(self.xgrid,self.ygrid,jx,jy,color='lightslategray',density=1,linewidth=1)
            plt.quiver(self.xgrid,self.ygrid,jx/j,jy/j,color='k',pivot='middle',alpha=0.4)
            plt.show()

        return phi,jx,jy
