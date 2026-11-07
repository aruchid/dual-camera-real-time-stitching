import unittest
import numpy as np
from src.blend import blend

class BlendTests(unittest.TestCase):
    def test_blend_geometry_and_uint8_arithmetic(self):
        out=blend(np.full((2,3),200,dtype=np.uint8),np.full((2,4),100,dtype=np.uint8),2)
        self.assertEqual(out.shape,(2,5))
        np.testing.assert_allclose(out[0],[200,500/3,400/3,100,100])
    def test_color_and_single_pixel_overlap(self):
        out=blend(np.zeros((2,2,3)),np.ones((2,2,3)),1)
        np.testing.assert_allclose(out[:,1,:],.5)
    def test_invalid_overlap_and_geometry(self):
        for overlap in (0,3,1.5,True):
            with self.assertRaises(ValueError):blend(np.zeros((2,2)),np.zeros((2,2)),overlap)
        with self.assertRaises(ValueError):blend(np.zeros((2,2)),np.zeros((3,2)),1)
