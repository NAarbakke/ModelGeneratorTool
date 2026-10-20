"""mesh_gen: Poisson trimming math - no Open3D."""

import numpy as np

from mesh_gen.meshing import low_density_mask


def test_low_density_mask():
    densities = np.arange(100, dtype=float)
    mask = low_density_mask(densities, quantile=0.05)
    assert mask.dtype == bool
    assert mask.sum() == 5
    assert mask[:5].all()
