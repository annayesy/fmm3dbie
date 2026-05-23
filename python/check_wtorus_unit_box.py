#!/usr/bin/env python
"""Check that the fmm3dbie wiggly torus lives in the unit box."""

import numpy as np

import fmm3dbie as h3


def main():
    radii = np.array([1.0, 2.0, 0.25], dtype=np.float64)
    scales = np.array([1.2, 1.0, 1.7], dtype=np.float64)
    nosc = 5
    nu = 64
    nv = 64
    norder = 3
    npols = int((norder + 1) * (norder + 2) / 2)
    npatches = int(2 * nu * nv)
    npts = int(npatches * npols)

    _, _, _, srcvals, _, _ = h3.get_wtorus_geom(
        radii, scales, nosc, nu, nv, npatches, norder, npts
    )
    xyz = np.asarray(srcvals[:3, :])
    lo = xyz.min(axis=1)
    hi = xyz.max(axis=1)

    print("wtorus bbox:")
    print("  x: [% .16e, % .16e]" % (lo[0], hi[0]))
    print("  y: [% .16e, % .16e]" % (lo[1], hi[1]))
    print("  z: [% .16e, % .16e]" % (lo[2], hi[2]))
    print("  spans:", hi - lo)

    tol = 5e-3
    assert lo[0] >= -0.5 - tol and hi[0] <= 0.5 + tol
    assert lo[1] >= -0.5 - tol and hi[1] <= 0.5 + tol
    assert lo[2] >= -0.5 - tol and hi[2] <= 0.5 + tol
    assert np.max(hi - lo) > 0.98


if __name__ == "__main__":
    main()
