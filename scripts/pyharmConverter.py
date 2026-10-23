from pathlib import Path

import h5py
import numpy as np
import pyharm
from pyharm.grid import Grid
from pyharm.defs import Loci

"""
Add the metric quantities needed by the Julia resizer / divB cleaner to a
KHARMA restart (.rhdf) file.

Works for any number of meshblocks (uniform mesh, no AMR). The global mesh is
taken from the restart's Info/RootGridSize and Info/RootGridDomain, which are
independent of how the mesh is split into blocks, so the arrays written are
always on the full global grid without ghost zones, in the same format as before:

    gcov_new, gcon_new   (4, 4, Nx1, Nx2, Nx3)
    gdet_new_center      (Nx1, Nx2, Nx3)
    gdet_new_face        (Nx1+1, Nx2+1, Nx3+1, 3)   (NaN in unused slots)
    gamma_ad             scalar
    [*_old versions on the original global grid if include_old_metric=True]
"""


def packed_face_gdet(grid):
    # Grids here are always built with ng=0, so GN == N (no ghost cells).
    n1, n2, n3 = (int(grid.GN[d]) for d in (1, 2, 3))

    ic = np.arange(n1)
    jc = np.arange(n2)
    kc = np.arange(n3)

    # The stored face field uses a common padded shape for all 3 components.
    gdet = np.full((n1 + 1, n2 + 1, n3 + 1, 3), np.nan, dtype=float)

    # Each component is valid on faces normal to its own axis.
    x_f1 = grid.coord(np.arange(n1 + 1), jc, kc, loc=Loci.FACE1)
    x_f2 = grid.coord(ic, np.arange(n2 + 1), kc, loc=Loci.FACE2)
    x_f3 = grid.coord(ic, jc, np.arange(n3 + 1), loc=Loci.FACE3)

    gdet[:n1 + 1, :n2,     :n3,     0] = grid.coords.gdet(x_f1)
    gdet[:n1,     :n2 + 1, :n3,     1] = grid.coords.gdet(x_f2)
    gdet[:n1,     :n2,     :n3 + 1, 2] = grid.coords.gdet(x_f3)

    return gdet


def read_global_mesh(rhdf_path):
    """
    Global mesh info from a Parthenon restart, valid for any meshblock count.

    Returns (root_size, ranges, n_blocks):
        root_size = (Nx1, Nx2, Nx3)                 interior cells of the whole mesh
        ranges    = ((x1min, x1max), (x2min, x2max), (x3min, x3max))
        n_blocks  = Info/NumMeshBlocks (informational)
    """
    with h5py.File(rhdf_path, "r") as f:
        info = f["Info"].attrs
        root_size = tuple(int(n) for n in info["RootGridSize"])
        dom = np.asarray(info["RootGridDomain"], dtype=float)
        n_blocks = int(info["NumMeshBlocks"])
        if "Multilevel" in info and int(info["Multilevel"]) != 0:
            raise NotImplementedError("Refined (multilevel) meshes are not supported")

    # RootGridDomain = [x1min, x1max, x1rat, x2min, x2max, x2rat, x3min, x3max, x3rat]
    if not np.allclose(dom[[2, 5, 8]], 1.0):
        raise NotImplementedError(f"Non-uniform root grid (ratios {dom[[2, 5, 8]]})")
    ranges = ((dom[0], dom[1]), (dom[3], dom[4]), (dom[6], dom[7]))
    return root_size, ranges, n_blocks


def make_grid(base_params, counts, ranges):
    """Build a pyharm Grid on the full global domain with the given cell counts and no ghosts."""
    params = base_params.copy()
    params.update({
        "n1tot": counts[0], "n2tot": counts[1], "n3tot": counts[2],
        "ng": 0,
    })
    # Grid prioritizes xNmin over startxN, so remove old minima and set starts.
    for axis, (lo, hi), n in zip((1, 2, 3), ranges, counts):
        params.pop(f"x{axis}min", None)
        params[f"startx{axis}"] = float(lo)
        params[f"dx{axis}"] = float((hi - lo) / n)
    return Grid(params)


def metric_arrays(grid, counts):
    n1, n2, n3 = counts
    gcov = np.broadcast_to(grid["gcov"], (4, 4, n1, n2, n3))
    gcon = np.broadcast_to(grid["gcon"], (4, 4, n1, n2, n3))
    gdet_center = np.broadcast_to(grid["gdet"], (n1, n2, n3))
    gdet_face = packed_face_gdet(grid)
    return gcov, gcon, gdet_center, gdet_face


def resize_rhdf(input_rhdf_file, input_phdf_file, metric_file,
                Nx1, Nx2, Nx3, include_old_metric=False):
    
    INPUT = Path(input_phdf_file)
    CONVERTED = Path(input_rhdf_file)

    if not CONVERTED.exists():
        raise FileNotFoundError(
            f"Converted file not found: {CONVERTED}. "
            "Check the filename printed by pyharm."
        )

    # Global mesh straight from the restart, independent of the meshblock count
    old_counts, ranges, n_blocks = read_global_mesh(CONVERTED)
    new_counts = (Nx1, Nx2, Nx3)
    print(f"Restart mesh: {old_counts[0]}x{old_counts[1]}x{old_counts[2]} "
          f"in {n_blocks} meshblock(s) -> new {Nx1}x{Nx2}x{Nx3}")

    # The phdf is used only for the coordinate system parameters and gamma
    dump = pyharm.load_dump(str(INPUT))
    base_params = dump.grid.params.copy()
    gamma_ad = float(dump.params.get("gam", dump.params.get("gamma", 1.666667)))

    # Sanity check: the phdf must describe the same global mesh as the restart
    phdf_counts = tuple(int(dump.grid.N[d]) for d in (1, 2, 3))
    if phdf_counts != old_counts:
        raise ValueError(
            f"phdf global mesh {phdf_counts} != rhdf RootGridSize {old_counts}. "
            "Are these files from the same run?"
        )
    del dump

    # New grid
    grid_new = make_grid(base_params, new_counts, ranges)
    gcov_new, gcon_new, gdet_new_center, gdet_new_face = metric_arrays(grid_new, new_counts)

    values = {
        "gcov_new": gcov_new,
        "gcon_new": gcon_new,
        "gdet_new_center": gdet_new_center,
        "gdet_new_face": gdet_new_face,
        "gamma_ad": gamma_ad,
    }

    if include_old_metric:
        # Old metric on the full global grid (all meshblocks merged)
        grid_old = make_grid(base_params, old_counts, ranges)
        gcov_old, gcon_old, gdet_old_center, gdet_old_face = metric_arrays(grid_old, old_counts)
        values["gcov_old"] = gcov_old
        values["gcon_old"] = gcon_old
        values["gdet_old_center"] = gdet_old_center
        values["gdet_old_face"] = gdet_old_face


    # The metric doesn't vary with x3, so gzip + shuffle compresses these
    # repeated values to a few MB instead of ~1 GB.
    with h5py.File(metric_file, "w") as h5file:
        for name, data in values.items():
            if np.ndim(data) == 0:
                h5file.create_dataset(name, data=data)
            else:
                h5file.create_dataset(name, data=data, compression="gzip",
                                      compression_opts=4, shuffle=True)
            print(f"Added {name} to {metric_file}")


    # with h5py.File(CONVERTED, "r+") as h5file:
    #     for name, data in values.items():
    #         if name in h5file:
    #             del h5file[name]
    #         h5file.create_dataset(name, data=data)
    #         print(f"Added {name} to {CONVERTED}")

