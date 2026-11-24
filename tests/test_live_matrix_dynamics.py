"""Numerical/interface contracts for a data-only live-state diagnostic."""
import io
import math
import pickle
import unittest

import numpy as np

from scripts.audit_live_matrix_dynamics import DataOnly, force, quiet


class MatrixDiagnosticContracts(unittest.TestCase):
    def test_force_is_the_derivative_of_the_reported_potential(self):
        rng = np.random.default_rng(41)
        s = rng.normal(size=(2, 3, 4, 4))
        direction = rng.normal(size=s.shape)
        epsilon = 1e-5
        numerical = (force(s+epsilon*direction, .005)[0] -
                     force(s-epsilon*direction, .005)[0])/(2*epsilon)
        analytic = np.sum(force(s, .005)[1]*direction, axis=(-3, -2, -1))
        np.testing.assert_allclose(numerical, analytic, rtol=1e-8, atol=1e-9)

    def test_zero_coupling_is_an_exact_harmonic_rotation(self):
        rng = np.random.default_rng(42)
        s, p = rng.normal(size=(2, 3, 4, 4)), rng.normal(size=(2, 3, 4, 4))
        a, b = quiet(s, p, .01, .8, 0.)
        np.testing.assert_allclose(a, math.cos(.008)*s+math.sin(.008)/.8*p)
        np.testing.assert_allclose(np.sum(b*b+.8**2*a*a), np.sum(p*p+.8**2*s*s))

    def test_metadata_reader_rejects_arbitrary_callables(self):
        with self.assertRaisesRegex(pickle.UnpicklingError, 'Unsupported'):
            DataOnly(io.BytesIO(pickle.dumps(eval))).load()


if __name__ == '__main__':
    unittest.main()
