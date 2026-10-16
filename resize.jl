include("src/kharmaResize.jl")
import .kharmaResize as kResize

import JLD2
using Revise

# --------------------------------------------
# --------------------------------------------
test_rhdf = "/home/june/Desktop/Code/Repos/kharmaResize/testDataGhostCells/torus.out1.00048.rhdf"
test_output = "/home/june/Desktop/Code/Repos/kharmaResize/testDataGhostCells/kharma_resampled.rhdf"

BC = [("dirichlet","dirichlet"),("neumann","neumann"),("periodic","periodic")]
cfg = kResize.Config(
    test_rhdf,
    test_output, 
    "linear",# place NN here for nearest neighbor compare
    BC, 
    256, 128, 128)


kResize.main(cfg)
# --------------------------------------------
# --------------------------------------------
# save Bprim
resized = kharmaResize.resize_grid(
        cfg.infile_rhdf,
        cfg.nx1,
        cfg.nx2,
        cfg.nx3,
        method=cfg.resize_method
    )
x1_faces, x2_faces, x3_faces,
x1_centers, x2_centers, x3_centers,
prim_rho, prim_u, prim_uvec, cons_fb = resized


# remove resized B field divergence
cons_fb_prime = kharmaResize.div_cleanup(x1_centers, x2_centers, x3_centers, cons_fb, cfg.BC)
JLD2.save_object("/home/june/Desktop/Code/Repos/kharmaResize/tempSaveResize/" * "cons_fb_prime" * ".jld2", cons_fb_prime)
cons_fb_prime = JLD2.load("/home/june/Desktop/Code/Repos/kharmaResize/tempSaveResize/" * "cons_fb_prime" * ".jld2", "single_stored_object")

dx1 = kharmaResize.kF.hF.get_dx(x1_centers)
dx2 = kharmaResize.kF.hF.get_dx(x2_centers)
dx3 = kharmaResize.kF.hF.get_dx(x3_centers)
divbprime = kharmaResize.kF.kharma_face_div(cons_fb_prime , dx1, dx2, dx3)
    
print("Max Divergence of cleaned Div B: ",maximum(abs.(divbprime)))
# write to new rhdf file
kharmaResize.write_rhdf(
    cfg.infile_rhdf, cfg.outfile_rhdf,                    
    x1_faces, x2_faces, x3_faces,       
    x1_centers, x2_centers, x3_centers, 
    prim_rho, prim_u, prim_uvec, cons_fb,
    cfg.BC                      
)
prim_rho
reshape(prim_rho, size(prim_rho,1),size(prim_rho,2),size(prim_rho,3),1)
# TESTING BELOW
# --------------------------------------------
# --------------------------------------------
# Load in original variables for testing
rhdf_file = kResize.HDF5.h5open(cfg.infile_rhdf, "r" )

cons_fb = kResize.HDF5.read(rhdf_file["cons.fB"])[:,:,:,:,1]
x1_old_faces = kResize.HDF5.read(rhdf_file["Locations"]["x"])[:,1]
x2_old_faces = kResize.HDF5.read(rhdf_file["Locations"]["y"])[:,1]
x3_old_faces = kResize.HDF5.read(rhdf_file["Locations"]["z"])[:,1]
NGhost = kResize.HDF5.attrs(rhdf_file["Info"])["NGhost"]
time = kResize.HDF5.attrs(rhdf_file["Info"])["Time"] 

cons_fb_old = kResize.kF.get_real_cells_vector(kResize.HDF5.read(rhdf_file["cons.fB"])[:,:,:,:,1],NGhost)
prim_rho_old = kResize.kF.get_real_cells(kResize.HDF5.read(rhdf_file["prims.rho"])[:,:,:,1],NGhost)    # cell centered
prim_u_old = kResize.kF.get_real_cells(kResize.HDF5.read(rhdf_file["prims.u"])[:,:,:,1],NGhost)        # cell centered
prim_uvec_old = kResize.kF.get_real_cells_vector(kResize.HDF5.read(rhdf_file["prims.uvec"])[:,:,:,:,1],NGhost) # cell centered

hst_data = kResize.kF.load_kharma_hst("/home/june/Desktop/Code/Repos/kharmaResize/testDataGhostCells/torus.hst")
for key in keys(hst_data)
    println(key)
end
current_hst_index = argmin(abs.(hst_data["time"] .- time))

dx1 = kResize.kF.hF.get_dx(x1_old_faces)
dx2 = kResize.kF.hF.get_dx(x2_old_faces)
dx3 = kResize.kF.hF.get_dx(x3_old_faces)
# --------------------------------------------
# --------------------------------------------
# test Divergence div_cleanup

print("Old max(∇⋅B): ",
    maximum(abs.(
        kResize.kF.kharma_face_div(kResize.kF.get_real_cells_vector(cons_fb,NGhost), dx1, dx2, dx3)
        ))
)
print("Old max(∇⋅B) as reported by Kharma: ",hst_data["MaxDivB"][current_hst_index])

resized = kResize.resize_grid(
        cfg.infile_rhdf,
        cfg.nx1,
        cfg.nx2,
        cfg.nx3,
        method=cfg.resize_method
    )
x1_faces, x2_faces, x3_faces,
x1_centers, x2_centers, x3_centers,
prim_rho, prim_u, prim_uvec, cons_fb = resized

cons_fb_prime = kResize.div_cleanup(x1_centers, x2_centers, x3_centers, cons_fb, cfg.BC)


JLD2.save_object("/home/june/Desktop/Code/Repos/kharmaResize/tempSaveResize/" * "cons_fb_prime" * ".jld2", cons_fb_prime)
JLD2.load("/home/june/Desktop/Code/Repos/kharmaResize/tempSaveResize/" * "cons_fb_prime" * ".jld2", "single_stored_object")

print("Cleaned max(∇⋅B): ",
    maximum(abs.(
        kResize.kF.kharma_face_div(kResize.kF.get_real_cells_vector(cons_fb_prime,NGhost), dx1, dx2, dx3)
        ))
)
# --------------------------------------------
# --------------------------------------------
# Check prim to cons conversion
phdf_file = kResize.HDF5.h5open("/home/june/Desktop/Code/Repos/kharmaResize/testDataGhostCells/torus.out0.00024.phdf", "r" )
cons_rho_old_kharma = kResize.kF.get_real_cells(kResize.HDF5.read(rhdf_file["cons.rho"])[:,:,:,1],NGhost);
cons_u_old_kharma = kResize.kF.get_real_cells(kResize.HDF5.read(rhdf_file["cons.u"])[:,:,:,1],NGhost);
cons_uvec_old_kharma = kResize.kF.get_real_cells_vector(kResize.HDF5.read(rhdf_file["cons.uvec"])[:,:,:,:,1],NGhost);
close(phdf_file)

cons_rho_old_julia, cons_u_old_julia, cons_uvec_old_julia = kResize.calculate_cons_vars(cfg.infile_rhdf,prim_rho_old, prim_u_old, prim_uvec_old, cons_fb_old,"old");

print("maximum difference between cons var rho: ",
    maximum(abs.(
        cons_rho_old_julia .- cons_rho_old_kharma
        ))
)
print("maximum difference between cons var u: ",
    maximum(abs.(
        cons_u_old_julia .- cons_u_old_kharma
        ))
)
print("maximum relative difference between cons var uvec: ",
    maximum(abs.(cons_uvec_old_julia .- cons_uvec_old_kharma))./maximum(cons_uvec_old_julia)
)
kResize.HDF5.read(rhdf_file["cons.rho"])[:,:,:,1]
# --------------------------------------------
# --------------------------------------------
# Inspect created rhdf file
new_rhdf_file = kResize.HDF5.h5open("/home/june/Desktop/Code/Repos/kharmaResize/testDataGhostCells/kharma_resampled.rhdf", "r" )
kResize.HDF5.attrs(rhdf_file["Params"])
size(new_rhdf_file["cons.fB"])

Vector{Int32}([Nx1, Nx2, Nx3])

kk = kResize.HDF5.h5open("/home/june/Desktop/Code/Repos/kharmaResize/testDataGhostCells/torus.out1.00048.rhdf")
loc.lx123
typeof(kResize.HDF5.attrs(rhdf_file["Input"])["File"])
kResize.HDF5.attrs(rhdf_file["Info"])["WallTime"]
MeshBlockSize

192 + 2*4
x1_faces[1],x1_faces[end]


dx1 = kResize.kF.hF.get_dx(x1_faces)
dx2 = kResize.kF.hF.get_dx(x2_faces)
dx3 = kResize.kF.hF.get_dx(x3_faces)

x1_faces[1],x1_faces[end]


NGhost = 4
x1_faces[1] - dx1 * NGhost
x1_faces[end] + dx1 * NGhost


collect(x1_centers[1] - dx1 * NGhost:dx1:x1_centers[end] + dx1 * NGhost)


length(x1_faces) + 2*NGhost
tfaces = x1_old_faces[NGhost+1:end-NGhost]

tfaces
maximum(abs,x1_old_faces - collect(tfaces[1] - kResize.kF.hF.get_dx(tfaces) * NGhost:kResize.kF.hF.get_dx(tfaces):tfaces[end] + kResize.kF.hF.get_dx(tfaces) * NGhost))


zeros(2,2,2,2)[1,:,:]