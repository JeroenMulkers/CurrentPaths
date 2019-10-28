import sys
sys.path.insert(0, "/home/szymag/python/CurrentPaths/")
import unittest
import numpy as np
from poisson import Poisson


class HomogeneousConductivityTestCases(unittest.TestCase):
    # Construct a 'Poisson object'
    p = Poisson(gridsize=(70,70), xrange=[0.,1.], yrange=[0.,1.],
                conductivity=[[1.,0],[0,1]])
    # Add a first contact point with a certain voltage
    region = lambda x,y: 0.15>x>0.1 and 0.4<y<0.6
    p.setPotential(regionFunc=region, U=1.0)
    # Of course we need a second contact
    region = lambda x,y: 0.9<x<0.95 and 0.4<y<0.6
    p.setPotential(regionFunc=region, U=-1.0)

    pot, cond_x, cond_y = p.solve(visualize=False)

    @staticmethod
    def test_potential():
        loaded_pot = np.loadtxt('./tst/homogeneous_conductivity_pot.txt')
        np.testing.assert_array_almost_equal(HomogeneousConductivityTestCases.pot,
                                             loaded_pot, decimal=5)

    @staticmethod
    def test_conductivity_x():
        loaded_cond_x = np.loadtxt('./tst/homogeneous_conductivity_cond_x.txt')
        np.testing.assert_array_almost_equal(HomogeneousConductivityTestCases.cond_x,
                                             loaded_cond_x, decimal=5)

    @staticmethod
    def test_conductivity_x():
        loaded_cond_y = np.loadtxt('./tst/homogeneous_conductivity_cond_y.txt')
        np.testing.assert_array_almost_equal(HomogeneousConductivityTestCases.cond_y,
                                             loaded_cond_y, decimal=5)

class HightResistivityTestCases(unittest.TestCase):
    p = Poisson( gridsize=(70,50), xrange=[-0.7,0.7], yrange=[0.,1.], conductivity=[[1,0],[0,1]])
    # Let's use the same contact points as in example 1
    region = lambda x,y: -0.45<x<-0.40 and 0.4<y<0.6
    p.setPotential(regionFunc=region, U=1.0)
    region = lambda x,y:  0.40<x< 0.45 and 0.4<y<0.6
    p.setPotential(regionFunc=region, U=-1.0)
    # Create rectangular region in the center with a low conductivity
    # (you can also set it to zero to have an isolating material at the center)
    region = lambda x,y: -0.1<x<0.1 and 0.3<y<0.7
    p.setConductivity(regionFunc=region, conductivity=[[0.3,0],[0,0.3]])

    pot, cond_x, cond_y = p.solve(visualize=False)

    @staticmethod
    def test_potential():
        loaded_pot = np.loadtxt('./tst/homogeneous_conductivity_pot.txt')
        np.testing.assert_array_almost_equal(HomogeneousConductivityTestCases.pot,
                                             loaded_pot, decimal=5)

    @staticmethod
    def test_conductivity_x():
        loaded_cond_x = np.loadtxt('./tst/homogeneous_conductivity_cond_x.txt')
        np.testing.assert_array_almost_equal(HomogeneousConductivityTestCases.cond_x,
                                             loaded_cond_x, decimal=5)

    @staticmethod
    def test_conductivity_x():
        loaded_cond_y = np.loadtxt('./tst/homogeneous_conductivity_cond_y.txt')
        np.testing.assert_array_almost_equal(HomogeneousConductivityTestCases.cond_y,
                                             loaded_cond_y, decimal=5)


if __name__ == '__main__':
    unittest.main()
