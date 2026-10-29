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

import pyharmConverter

"""

"""

path_to_JULIARMA = "/home/june/Desktop/Code/Repos/kharmaResize/"
# ---------------------------------------------------------------------------------
# params --------------------------------------------------------------------------
# file paths
input_rhdf_file = "/home/june/Desktop/Code/Repos/kharmaResize/products/GhostBuster/torus.out1.00049.rhdf"
input_phdf_file = "/home/june/Desktop/Code/Repos/kharmaResize/products/GhostBuster/torus.out0.00024.phdf"
output_rhdf_file = "/home/june/Desktop/Code/Repos/kharmaResize/products/256x128x128/192x96x96_r30_resized-accretion_start.rhdf"
# new resolution
Nx1 = 256
Nx2 = 128
Nx3 = 128
Mesh_Nx1 = 128
Mesh_Nx2 = 64
Mesh_Nx3 = 128
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

print("creating metric values from phdf file and storing in metric file...")

metric_file = str(Path(output_rhdf_file).with_suffix("")) + "_metric.h5"

pyharmConverter.resize_rhdf(input_rhdf_file, input_phdf_file, metric_file,
                            Nx1, Nx2, Nx3, include_old_metric=False)



# Runs the command through the system shell

command = "julia -t "+str(nthread)+ " " + path_to_JULIARMA +"/scripts/resize_rst.jl " + input_rhdf_file + " " + output_rhdf_file

args = " "
args += "--metricfile " + metric_file + " "
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
args += "--meshnx1 " + str(Mesh_Nx1) + " "
args += "--meshnx2 " + str(Mesh_Nx2) + " "
args += "--meshnx3 " + str(Mesh_Nx3) + " "

command = command + args

# --------
print("Running Julia Resizing Script...")
subprocess.run(command, shell=True, check=True)

print("removing created metric file...")
Path(metric_file).unlink(missing_ok=True)