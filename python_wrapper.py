from pathlib import Path
import subprocess

import h5py
import pyharm
from pyharm.grid import Grid
from pyharm.defs import Loci
import numpy as np

import numpy as np
from pyharm.defs import Loci
import subprocess

def packed_face_gdet(grid):
    # GN includes ghost cells when grid was constructed with ng=the RHDF NGhost.
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

    gdet_f1 = grid.coords.gdet(x_f1)
    gdet_f2 = grid.coords.gdet(x_f2)
    gdet_f3 = grid.coords.gdet(x_f3)

    gdet[:n1 + 1, :n2,     :n3,     0] = gdet_f1
    gdet[:n1,     :n2 + 1, :n3,     1] = gdet_f2
    gdet[:n1,     :n2,     :n3 + 1, 2] = gdet_f3

    return gdet


def resize_rhdf(input_rhdf_file,input_phdf_file,output_rhdf_file,Nx1,Nx2,Nx3,boundary_conditions,method,include_old_metric=False):
    INPUT = Path(input_phdf_file)
    CONVERTED = Path(input_rhdf_file)
    dump = pyharm.load_dump(str(INPUT))

    nx1_new, nx2_new, nx3_new = Nx1,Nx2,Nx3


    #----------------------------------
    params = dump.grid.params.copy()
    grid_old = Grid(params)
    corners = grid_old.coord_bulk(mesh=True)
    nx1_old, nx2_old, nx3_old = corners.shape[1]-1, corners.shape[2]-1, corners.shape[3]-1
    # center = grid_old.coord_bulk(loc=Loci.CENT)
    # center = Loci.CENT.value
    # print(center)
    gcov_all = grid_old["gcov"]
    gcon_all = grid_old["gcon"]
    gdet_all = grid_old["gdet"]
    gcov_old = np.broadcast_to(gcov_all, (4, 4, nx1_old, nx2_old, nx3_old))
    gcon_old = np.broadcast_to(gcon_all, (4, 4, nx1_old, nx2_old, nx3_old))
    gdet_old_center = np.broadcast_to(gdet_all, (nx1_old, nx2_old, nx3_old))
    gdet_old_face = packed_face_gdet(grid_old)


    x1_faces_old = corners[1, :, 0, 0]
    x2_faces_old = corners[2, 0, :, 0]
    x3_faces_old = corners[3, 0, 0, :]


    ranges = [
        (x1_faces_old[0], x1_faces_old[-1]),
        (x2_faces_old[0], x2_faces_old[-1]),
        (x3_faces_old[0], x3_faces_old[-1]),
    ]
    x1_faces = np.linspace(ranges[0][0], ranges[0][1], nx1_new + 1)
    x2_faces = np.linspace(ranges[1][0], ranges[1][1], nx2_new + 1)
    x3_faces = np.linspace(ranges[2][0], ranges[2][1], nx3_new + 1)

    # These are the face arrays produced by your resize code.
    # They must be in pyharm's native coordinates.
    faces = (x1_faces, x2_faces, x3_faces)
    counts = (nx1_new, nx2_new, nx3_new)

    params = dump.grid.params.copy()
    params.update({
        "n1tot": nx1_new, "n2tot": nx2_new, "n3tot": nx3_new,
        "ng": 0,
    })


    # Grid prioritizes xNmin over startxN, so remove old minima and set starts.
    for axis, face, n in zip((1, 2, 3), faces, counts):
        params.pop(f"x{axis}min", None)
        params[f"startx{axis}"] = float(face[0])
        params[f"dx{axis}"] = float((face[-1] - face[0]) / n)

    grid_new = Grid(params)

    gcov_all = grid_new["gcov"]
    gcon_all = grid_new["gcon"]
    gdet_all = grid_new["gdet"]
    gcov_new = np.broadcast_to(gcov_all, (4, 4, nx1_new, nx2_new, nx3_new))
    gcon_new = np.broadcast_to(gcon_all, (4, 4, nx1_new, nx2_new, nx3_new))
    gdet_new_center = np.broadcast_to(gdet_all, (nx1_new, nx2_new, nx3_new))
    gdet_new_face = packed_face_gdet(grid_new)


    gamma_ad = float(dump.params.get("gam", dump.params.get("gamma", 1.666667)))

    values = {
        "gcov_new": gcov_new,
        "gcon_new": gcon_new,
        "gdet_new_center": gdet_new_center,
        "gdet_new_face": gdet_new_face,
        # "gcov_old": gcov_old,

        "gamma_ad": gamma_ad,
    }

    if include_old_metric:
        values["gcov_old"] = gcov_old
        values["gcon_old"] = gcon_old,
        values["gdet_old_center"] = gdet_old_center,
        values["gdet_old_face"] = gdet_old_face,


    del dump

    if not CONVERTED.exists():
        raise FileNotFoundError(
            f"Converted file not found: {CONVERTED}. "
            "Check the filename printed by pyharm."
        )

    with h5py.File(CONVERTED, "r+") as h5file:
        for name, data in values.items():
            if name in h5file:
                del h5file[name]
            h5file.create_dataset(name, data=data)
            print(f"Added {name} to {CONVERTED}")
# ---------------------------------------------------------------------------------
# params --------------------------------------------------------------------------
# file paths
input_rhdf_file = "/home/june/Desktop/Code/Repos/kharmaResize/fullPipelineTest/torus.out1.00024.rhdf"
input_phdf_file = "/home/june/Desktop/Code/Repos/kharmaResize/fullPipelineTest/torus.out0.00012.phdf"
output_rhdf_file = "/home/june/Desktop/Code/Repos/kharmaResize/fullPipelineTest/kharma_resampled_r24.rhdf"
# new resolution
Nx1 = 256
Nx2 = 128
Nx3 = 128
# 1=Dirichlet, 2=Neumann, 3=periodic, 4=transmitting
boundary_conditions = [
    (1,1),
    (4,4),
    (3,3)
]
# choice for interpolation type (1:linear or 2:nearest neighbor)
method = 1 
# number of threads for julia
nthread = 10
# ---------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------

print("Adding needed values to rhdf file from phdf file...")

resize_rhdf(input_rhdf_file,input_phdf_file,output_rhdf_file,Nx1,Nx2,Nx3,boundary_conditions,method,include_old_metric=False)



# Runs the command through the system shell

command = "julia -t "+str(nthread)+" resize_rst.jl " + input_rhdf_file + " " + output_rhdf_file

args = " "
args += "--method " + str(method) + " "
args += "--bc1inner " + str(boundary_conditions[0][0]) + " "
args += "--bc1outer " + str(boundary_conditions[0][1]) + " "
args += "--bc2inner " + str(boundary_conditions[1][0]) + " "
args += "--bc2outer " + str(boundary_conditions[1][1]) + " "
args += "--bc3inner " + str(boundary_conditions[2][0]) + " "
args += "--bc3outer " + str(boundary_conditions[2][1]) + " "
args += "--nx1 " + str(Nx1) + " "
args += "--nx2 " + str(Nx2) + " "
args += "--nx3 " + str(Nx3) + " "

command = command + args

# --------
print("Running Julia Resizing Script...")

subprocess.run(command, shell=True)

