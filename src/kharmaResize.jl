module kharmaResize

export kharmaResize



include("kharmaFunctions.jl")
include("meshblocks.jl")

import .kharmaFunctions as kF
import .Meshblocks

import HDF5

import LinearAlgebra
import SparseArrays
import IterativeSolvers





struct Config 
    infile_rhdf::String
    outfile_rhdf::String 
    metric_file::String
    resize_method::String
    BC::Array
    # New resolution 
    nx1::Int 
    nx2::Int 
    nx3::Int
    mesh_nx1::Int
    mesh_nx2::Int 
    mesh_nx3::Int

end

function GhostBuster(
    NGhost,BC,
    Nx1, Nx2, Nx3,
    x1_faces,x2_faces,x3_faces,
    x1_centers,x2_centers,x3_centers,
    prim_rho,prim_u,prim_uvec,cons_fb,
    
)   


    dx1 = kF.hF.get_dx(x1_faces)
    dx2 = kF.hF.get_dx(x2_faces)
    dx3 = kF.hF.get_dx(x3_faces)
    # Nx1, Nx2, Nx3 = size(prim_rho)
    x1_faces   = collect(range(x1_faces[1]   - dx1*NGhost, x1_faces[end]   + dx1*NGhost; length = length(x1_faces)   + 2NGhost))
    x2_faces   = collect(range(x2_faces[1]   - dx2*NGhost, x2_faces[end]   + dx2*NGhost; length = length(x2_faces)   + 2NGhost))
    x3_faces   = collect(range(x3_faces[1]   - dx3*NGhost, x3_faces[end]   + dx3*NGhost; length = length(x3_faces)   + 2NGhost))
    x1_centers = collect(range(x1_centers[1] - dx1*NGhost, x1_centers[end] + dx1*NGhost; length = length(x1_centers) + 2NGhost))
    x2_centers = collect(range(x2_centers[1] - dx2*NGhost, x2_centers[end] + dx2*NGhost; length = length(x2_centers) + 2NGhost))
    x3_centers = collect(range(x3_centers[1] - dx3*NGhost, x3_centers[end] + dx3*NGhost; length = length(x3_centers) + 2NGhost))


    prim_rho  = kF.hF.pad_array(prim_rho;Nx1_bc=BC[1],Nx2_bc=BC[2],Nx3_bc=BC[3],NGhost=NGhost) 
    prim_u    = kF.hF.pad_array(prim_u;Nx1_bc=BC[1],Nx2_bc=BC[2],Nx3_bc=BC[3],NGhost=NGhost)  
    prim_uvec1 = kF.hF.pad_array(prim_uvec[:,:,:,1];Nx1_bc=BC[1],Nx2_bc=BC[2],Nx3_bc=BC[3],NGhost=NGhost)        
    prim_uvec2 = kF.hF.pad_array(prim_uvec[:,:,:,2];Nx1_bc=BC[1],Nx2_bc=BC[2],Nx3_bc=BC[3],NGhost=NGhost,transmit_sign=-1)        
    prim_uvec3 = kF.hF.pad_array(prim_uvec[:,:,:,3];Nx1_bc=BC[1],Nx2_bc=BC[2],Nx3_bc=BC[3],NGhost=NGhost,transmit_sign=-1)        
    prim_uvec = cat(prim_uvec1,prim_uvec2,prim_uvec3, dims=4)    

    cons_fb1 = cons_fb[1:Nx1 + 1, 1:Nx2, 1:Nx3, 1]
    cons_fb2 = cons_fb[1:Nx1, 1:Nx2 + 1, 1:Nx3, 2]
    cons_fb3 = cons_fb[1:Nx1, 1:Nx2, 1:Nx3 + 1, 3]
    cons_fb1, cons_fb2, cons_fb3 = kF.hF.pad_face_array(
        cons_fb1, cons_fb2, cons_fb3;
        Nx1_bc=BC[1],Nx2_bc=BC[2],Nx3_bc=BC[3],
        NGhost=NGhost,
    )
    Nx1_face_padded = Nx1+2*NGhost
    Nx2_face_padded = Nx2+2*NGhost
    Nx3_face_padded = Nx3+2*NGhost

    
    cons_fb = zeros(Float64,Nx1_face_padded+1,Nx2_face_padded+1,Nx3_face_padded+1,3)
    cons_fb[1:Nx1_face_padded + 1, 1:Nx2_face_padded, 1:Nx3_face_padded, 1] .= cons_fb1 
    cons_fb[1:Nx1_face_padded, 1:Nx2_face_padded + 1, 1:Nx3_face_padded, 2] .= cons_fb2 
    cons_fb[1:Nx1_face_padded, 1:Nx2_face_padded, 1:Nx3_face_padded + 1, 3] .= cons_fb3

    return (
        x1_faces,x2_faces,x3_faces,
        x1_centers,x2_centers,x3_centers,
        prim_rho,prim_u,prim_uvec,cons_fb,
    )


end

function get_rhdf_params(rhdf_h5_path,BC)
    println( "Reading KHARMA dump:" )
    println( " ",rhdf_h5_path)

    rhdf_file = HDF5.h5open(rhdf_h5_path, "r" )

    numMeshBlocks = HDF5.attrs(rhdf_file["Info"])["NumMeshBlocks"]

    Nx1_old, Nx2_old, Nx3_old = HDF5.attrs(rhdf_file["Info"])["RootGridSize"]
    
    NGhost = HDF5.attrs(rhdf_file["Info"])["NGhost"]
    Nx1_GI = Nx1_old + 2 * NGhost
    Nx2_GI = Nx2_old + 2 * NGhost
    Nx3_GI = Nx3_old + 2 * NGhost

    if numMeshBlocks == 1
    
        # TODO: dont forget to add the removed dims
        x1_old_faces = HDF5.read(rhdf_file["Locations"]["x"])[:,1]
        x2_old_faces = HDF5.read(rhdf_file["Locations"]["y"])[:,1]
        x3_old_faces = HDF5.read(rhdf_file["Locations"]["z"])[:,1]

        x1_old_centers = HDF5.read(rhdf_file["VolumeLocations"]["x"])[:,1]
        x2_old_centers = HDF5.read(rhdf_file["VolumeLocations"]["y"])[:,1]
        x3_old_centers = HDF5.read(rhdf_file["VolumeLocations"]["z"])[:,1]

        prim_rho = HDF5.read(rhdf_file["prims.rho"])[:,:,:,1]    # cell centered
        prim_u = HDF5.read(rhdf_file["prims.u"])[:,:,:,1]        # cell centered
        prim_uvec = HDF5.read(rhdf_file["prims.uvec"])[:,:,:,:,1] # cell centered
        cons_fb = HDF5.read(rhdf_file["cons.fB"])[:,:,:,:,1]      # face centered

        if HDF5.attrs(rhdf_file["Info"])["IncludesGhost"] == 0
            println("Ghosts Cells Missing! Finding...")
            ghostsIncluded = GhostBuster(
                NGhost,BC,
                Nx1_old, Nx2_old, Nx3_old,
                x1_old_faces,x2_old_faces,x3_old_faces,
                x1_old_centers,x2_old_centers,x3_old_centers,
                prim_rho,prim_u,prim_uvec,cons_fb,
                
            )
            x1_old_faces,x2_old_faces,x3_old_faces,
            x1_old_centers,x2_old_centers,x3_old_centers,
            prim_rho,prim_u,prim_uvec,cons_fb, = ghostsIncluded
        end

        x1_range = (x1_old_faces[1 + NGhost], x1_old_faces[end - NGhost])
        x2_range = (x2_old_faces[1 + NGhost], x2_old_faces[end - NGhost])
        x3_range = (x3_old_faces[1 + NGhost], x3_old_faces[end - NGhost])



        
        
    else
        println("Meshblock size > 1, merging...")
        (;x1_faces,x2_faces,x3_faces,
        x1_centers,x2_centers,x3_centers,
        prim_rho,prim_u,prim_uvec,cons_fb) = Meshblocks.merge_meshblocks(rhdf_file)

        x1_old_faces = x1_faces[:,1]
        x2_old_faces = x2_faces[:,1]
        x3_old_faces = x3_faces[:,1]

        x1_old_centers = x1_centers[:,1]
        x2_old_centers = x2_centers[:,1]
        x3_old_centers = x3_centers[:,1]
        
        prim_rho = prim_rho[:,:,:,1]    # cell centered
        prim_u = prim_u[:,:,:,1]        # cell centered
        prim_uvec = prim_uvec[:,:,:,:,1] # cell centered
        cons_fb = cons_fb[:,:,:,:,1]     # face centered

        if HDF5.attrs(rhdf_file["Info"])["IncludesGhost"] == 0
            println("Ghosts Cells Missing! Finding...")
            ghostsIncluded = GhostBuster(
                NGhost,BC,
                Nx1_old, Nx2_old, Nx3_old,
                x1_old_faces,x2_old_faces,x3_old_faces,
                x1_old_centers,x2_old_centers,x3_old_centers,
                prim_rho,prim_u,prim_uvec,cons_fb,
                
            )
            x1_old_faces,x2_old_faces,x3_old_faces,
            x1_old_centers,x2_old_centers,x3_old_centers,
            prim_rho,prim_u,prim_uvec,cons_fb, = ghostsIncluded
        end

        x1_range = (x1_old_faces[1 + NGhost], x1_old_faces[end - NGhost])
        x2_range = (x2_old_faces[1 + NGhost], x2_old_faces[end - NGhost])
        x3_range = (x3_old_faces[1 + NGhost], x3_old_faces[end - NGhost])
        
    


        

    end

    close(rhdf_file)

    return (
        x1_old_faces,x2_old_faces,x3_old_faces,
        x1_old_centers,x2_old_centers,x3_old_centers,
        x1_range,x2_range,x3_range,
        Nx1_GI,Nx2_GI,Nx3_GI,NGhost,
        prim_rho,prim_u,prim_uvec,cons_fb
    )
end

function resize_grid(rhdf_h5_path,Nx1,Nx2,Nx3,BC;method="linear")
    """
        Take in a rhdf KHARMA file and upscale resolution of 
        primatives vars density, energy, velocity vector, and 
        conservative face centered B field

        Nx1, Nx2, Nx3: New resolution

        method: {
                linear = linear interpolation
                NN = Nearest Neighbor
                    }
    """
    println( "Reading KHARMA dump:" )
    println( " ",rhdf_h5_path)
    # rhdf_file = HDF5.h5open(rhdf_h5_path, "r" )
    # # --------------------------------------------
    # # --------------------------------------------
    # # Load In Parameters--------------------------
    # Nx1_old, Nx2_old, Nx3_old =HDF5.attrs(rhdf_file["Info"])["RootGridSize"]

    # NGhost = HDF5.attrs(rhdf_file["Info"])["NGhost"]
    # Nx1_GI = Nx1_old + 2 * NGhost
    # Nx2_GI = Nx2_old + 2 * NGhost
    # Nx3_GI = Nx3_old + 2 * NGhost
    
    # # TODO: dont forget to add the removed dims
    # x1_old_faces = HDF5.read(rhdf_file["Locations"]["x"])[:,1]
    # x2_old_faces = HDF5.read(rhdf_file["Locations"]["y"])[:,1]
    # x3_old_faces = HDF5.read(rhdf_file["Locations"]["z"])[:,1]

    # x1_old_centers = HDF5.read(rhdf_file["VolumeLocations"]["x"])[:,1]
    # x2_old_centers = HDF5.read(rhdf_file["VolumeLocations"]["y"])[:,1]
    # x3_old_centers = HDF5.read(rhdf_file["VolumeLocations"]["z"])[:,1]

    # x1_range = (x1_old_faces[1 + NGhost], x1_old_faces[end - NGhost])
    # x2_range = (x2_old_faces[1 + NGhost], x2_old_faces[end - NGhost])
    # x3_range = (x3_old_faces[1 + NGhost], x3_old_faces[end - NGhost])



    
    # prim_rho = HDF5.read(rhdf_file["prims.rho"])[:,:,:,1]    # cell centered
    # prim_u = HDF5.read(rhdf_file["prims.u"])[:,:,:,1]        # cell centered
    # prim_uvec = HDF5.read(rhdf_file["prims.uvec"])[:,:,:,:,1] # cell centered
    # cons_fb = HDF5.read(rhdf_file["cons.fB"])[:,:,:,:,1]      # face centered
    # # cons_rho =HDF5.read(rhdf_rst["cons.rho"])     # face centered
    # # cons_u =HDF5.read(rhdf_rst["cons.u"])         # face centered
    # # cons_uvec = HDF5.read(rhdf_rst["cons.uvec"])  # face centered

    # close(rhdf_file)
    x1_old_faces,x2_old_faces,x3_old_faces,
    x1_old_centers,x2_old_centers,x3_old_centers,
    x1_range,x2_range,x3_range,
    Nx1_GI,Nx2_GI,Nx3_GI,NGhost,
    prim_rho,prim_u,prim_uvec,cons_fb = get_rhdf_params(rhdf_h5_path, BC)


    # --------------------------------------------
    # --------------------------------------------
    # New Grid Parameters
    x1_new_faces = collect(range(x1_range[1], x1_range[2], length=Nx1 + 1))
    x2_new_faces = collect(range(x2_range[1], x2_range[2], length=Nx2 + 1))
    x3_new_faces = collect(range(x3_range[1], x3_range[2], length=Nx3 + 1))
    
    x1_new_centers = 0.5 * (x1_new_faces[1:end-1] + x1_new_faces[2:end])
    x2_new_centers = 0.5 * (x2_new_faces[1:end-1] + x2_new_faces[2:end])
    x3_new_centers = 0.5 * (x3_new_faces[1:end-1] + x3_new_faces[2:end])

    # --------------------------------------------
    # --------------------------------------------
    # Create easy Interpolations

    itp_prim_rho = kF.hF.create_interpolations(x1_old_centers,x2_old_centers,x3_old_centers,prim_rho,method)
    itp_prim_u = kF.hF.create_interpolations(x1_old_centers,x2_old_centers,x3_old_centers,prim_u,method)
    itp_prim_uvec_x1 = kF.hF.create_interpolations(x1_old_centers,x2_old_centers,x3_old_centers,prim_uvec[:,:,:,1],method)
    itp_prim_uvec_x2 = kF.hF.create_interpolations(x1_old_centers,x2_old_centers,x3_old_centers,prim_uvec[:,:,:,2],method)
    itp_prim_uvec_x3 = kF.hF.create_interpolations(x1_old_centers,x2_old_centers,x3_old_centers,prim_uvec[:,:,:,3],method)
    # itp_gdet = kF.hF.create_interpolations(x1_old_centers,x2_old_centers,x3_old_centers,gdet)

    resized_prim_rho = [itp_prim_rho(i, j, k) for i in x1_new_centers, j in x2_new_centers, k in x3_new_centers]
    resized_prim_u = [itp_prim_u(i, j, k) for i in x1_new_centers, j in x2_new_centers, k in x3_new_centers]
    resized_prim_uvec_x1 = [itp_prim_uvec_x1(i, j, k) for i in x1_new_centers, j in x2_new_centers, k in x3_new_centers]
    resized_prim_uvec_x2 = [itp_prim_uvec_x2(i, j, k) for i in x1_new_centers, j in x2_new_centers, k in x3_new_centers]
    resized_prim_uvec_x3 = [itp_prim_uvec_x3(i, j, k) for i in x1_new_centers, j in x2_new_centers, k in x3_new_centers]
    resized_prim_uvec = cat(resized_prim_uvec_x1,resized_prim_uvec_x2,resized_prim_uvec_x3, dims=4)

    # resized_gdet = [itp_gdet(i, j, k) for i in x1_new_centers, j in x2_new_centers, k in x3_new_centers]
    
    # --------------------------------------------
    # --------------------------------------------
    # Create b Interpolations
    
    # B1: (Nx1_old+1, Nx2_old,   Nx3_old)
    itp_cons_fb_x1 = kF.hF.create_interpolations(
        x1_old_faces, x2_old_centers, x3_old_centers,
        cons_fb[1:Nx1_GI + 1, 1:Nx2_GI, 1:Nx3_GI, 1],method
        
    )
    
    # B2: (Nx1_old,   Nx2_old+1, Nx3_old)
    itp_cons_fb_x2 = kF.hF.create_interpolations(
        x1_old_centers, x2_old_faces, x3_old_centers,
        cons_fb[1:Nx1_GI, 1:Nx2_GI +1, 1:Nx3_GI, 2],method
    )

    # B3: (Nx1_old,   Nx2_old,   Nx3_old+1)
    itp_cons_fb_x3 = kF.hF.create_interpolations(
        x1_old_centers, x2_old_centers, x3_old_faces,
        cons_fb[1:Nx1_GI, 1:Nx2_GI, 1:Nx3_GI +1, 3],method
    )
    # --------------------------------------------
    # --------------------------------------------
    # Use interpolations

    # Active ranges
    # B1: 193 × 96 × 96
    # B2: 192 × 97 × 96
    # B3: 192 × 96 × 97
    resized_cons_fb_x1 = zeros(Nx1 + 1,Nx2 + 1,Nx3 + 1)
    resized_cons_fb_x2 = zeros(Nx1 + 1,Nx2 + 1,Nx3 + 1)
    resized_cons_fb_x3 = zeros(Nx1 + 1,Nx2 + 1,Nx3 + 1)

    
    fb_x1 = [itp_cons_fb_x1(i, j, k) for i in x1_new_faces, j in x2_new_centers, k in x3_new_centers]
    fb_x2 = [itp_cons_fb_x2(i, j, k) for i in x1_new_centers, j in x2_new_faces, k in x3_new_centers]
    fb_x3 = [itp_cons_fb_x3(i, j, k) for i in x1_new_centers, j in x2_new_centers, k in x3_new_faces]


    resized_cons_fb_x1[:,1:end-1,1:end-1] .= fb_x1
    resized_cons_fb_x2[1:end-1,:,1:end-1] .= fb_x2
    resized_cons_fb_x3[1:end-1,1:end-1,:] .= fb_x3
    
    

    resized_cons_fb = cat(resized_cons_fb_x1,resized_cons_fb_x2,resized_cons_fb_x3, dims=4)
    # --------------------------------------------
    # --------------------------------------------
    # Print value of original B field divergence
    dx1 = kF.hF.get_dx(x1_old_faces)
    dx2 = kF.hF.get_dx(x2_old_faces)
    dx3 = kF.hF.get_dx(x3_old_faces)

    # max_div_b = maximum(abs.(divergence_face_cartesian(cons_fb, dx1, dx2, dx3, gdet)))
    cons_fb_real_cells = kF.get_real_cells_vector(cons_fb,NGhost)
    max_div_b  = maximum(abs.(kF.kharma_face_div(cons_fb_real_cells, dx1, dx2, dx3)))
    print("Old max(B divergence): ")
    println(max_div_b)

    # --------------------------------------------
    # --------------------------------------------
    # Print new value of resized B field divergence
    dx1 = kF.hF.get_dx(x1_new_faces)
    dx2 = kF.hF.get_dx(x2_new_faces)
    dx3 = kF.hF.get_dx(x3_new_faces)
    max_div_b  = maximum(abs.(kF.kharma_face_div(resized_cons_fb, dx1, dx2, dx3)))

    print("New max(B divergence): ")
    println(max_div_b)
    # --------------------------------------------
    # --------------------------------------------
    # Create cons values

    # return cc_cons_fb_old
    # cc_cons_fb_new = kF.hF.faceB_to_center(resized_cons_fb, Nx1_old, Nx2_old, Nx3_old; ng=NGhost)
            
    return (
        x1_new_faces, x2_new_faces, x3_new_faces,
        x1_new_centers, x2_new_centers, x3_new_centers,
        resized_prim_rho, resized_prim_u, resized_prim_uvec, 
        resized_cons_fb,
        )
end

function calculate_cons_vars(rhdf_h5_path,prim_rho, prim_u, prim_uvec, cons_fb, grid="new",NGhost=0)
    if NGhost != 0 
        cons_fb = kF.get_real_cells_vector(cons_fb, NGhost)
        prim_uvec = kF.get_real_cells_vector(prim_uvec, NGhost)
        prim_u = kF.get_real_cells_vector(prim_u, NGhost)
        prim_rho = kF.get_real_cells_vector(prim_rho, NGhost)
    end

    # --------------------------------------------
    # --------------------------------------------
    # Read needed vars
    println( "Reading KHARMA dump:" )
    println( " ",rhdf_h5_path)

    rhdf_file = HDF5.h5open(rhdf_h5_path, "r" )

    gamma_ad = HDF5.read(rhdf_file["gamma_ad"])
    gcon_raw = HDF5.read(rhdf_file["gcon_"*grid])
    gcov_raw = HDF5.read(rhdf_file["gcov_"*grid])
    gdet_center_raw = HDF5.read(rhdf_file["gdet_"*grid*"_center"])
    gdet_face_raw = HDF5.read(rhdf_file["gdet_"*grid*"_face"])
    close(rhdf_file)

    
    # --------------------------------------------
    # --------------------------------------------

    gcon,gcov,gdet_center,gdet_face = kF.hF.reorder_dims(gcon_raw,gcov_raw,gdet_center_raw,gdet_face_raw)
    Nx1,Nx2,Nx3 = size(gcon)[1:3]
    cc_cons_fb = kF.hF.faceB_to_center(cons_fb,gdet_face,Nx1,Nx2,Nx3)
    cons_rho, cons_u, cons_uvec = kF.p_to_u_arrays(
        prim_rho, prim_u, prim_uvec, cc_cons_fb, 
        gcon, gcov, gdet_center, 
        gamma_ad)

    return cons_rho, cons_u, cons_uvec

    
end


function div_cleanup(x1_centers, x2_centers, x3_centers, fB, BC)
    # @show(ndims(fB))
    @assert(ndims(x1_centers) == 1)
    @assert(ndims(x2_centers) == 1)
    @assert(ndims(x3_centers) == 1)
    @assert ndims(fB) == 4
    @assert size(fB, 4) == 3
    # init_phi = zeros(Float64,Nx1,Nx2,Nx3)
    dx1 = kF.hF.get_dx(x1_centers)
    dx2 = kF.hF.get_dx(x2_centers)
    dx3 = kF.hF.get_dx(x3_centers)
    phi, residual, residual_norm, history = kF.solve_phi(fB,dx1, dx2, dx3, BC)
    
    # correct x2 boundarys for transmitting
    Nx1, Nx2, Nx3 = size(phi)
    grad_phi = kF.gradient(phi, dx1, dx2, dx3, BC)
    grad_phi[1:Nx1, 1,      1:Nx3, 2] .= fB[1:Nx1, 1,      1:Nx3, 2]
    grad_phi[1:Nx1, Nx2+1, 1:Nx3, 2] .= fB[1:Nx1, Nx2+1, 1:Nx3, 2]

    Bprime = fB .- grad_phi
    # Bprime = fB - kF.gradient(phi, dx1, dx2, dx3, BC)
    print("After Cleaning ∇⋅B|")
    kF.report_max_divB(Bprime,dx1, dx2, dx3,0)
    # divbprime = kF.kharma_face_div(Bprime, dx1, dx2, dx3)
    
    # print("Max Divergence of cleaned Div B: ",maximum(abs.(divbprime)))

    
    return Bprime
    

end


function write_par_file_attribute!(input_group, par_text)
    all(isascii, par_text) ||
        throw(ArgumentError("Parthenon expects ASCII text for Input/File"))

    HDF5.delete_attribute(input_group, "File")

    dtype = HDF5.Datatype(
        HDF5.API.h5t_create(HDF5.API.H5T_STRING, HDF5.API.H5T_VARIABLE)
    )
    HDF5.API.h5t_set_cset(dtype, HDF5.API.H5T_CSET_ASCII)
    space = HDF5.dataspace(par_text)
    attr = HDF5.create_attribute(input_group, "File", dtype, space)

    try
        HDF5.write_attribute(attr, dtype, par_text)
    finally
        close(attr)
        close(space)
        close(dtype)
    end
end

function rewrite_as_vlen_ascii!(parent, name)
    attrs = HDF5.attributes(parent)
    value = read(attrs[name])

    strings = value isa AbstractString ?
        [String(value)] : String.(collect(value))

    all(isascii, strings) ||
        throw(ArgumentError("$name contains non-ASCII characters"))

    # Preserve the original attribute shape: scalar or array.
    space = value isa AbstractString ?
        HDF5.dataspace(value) : HDF5.dataspace(size(value))

    HDF5.delete_attribute(parent, name)

    dtype = HDF5.Datatype(
        HDF5.API.h5t_create(HDF5.API.H5T_STRING, HDF5.API.H5T_VARIABLE)
    )
    HDF5.API.h5t_set_cset(dtype, HDF5.API.H5T_CSET_ASCII)

    attr = HDF5.create_attribute(parent, name, dtype, space)

    # HDF5 variable-length strings are stored as an array of C-string pointers.
    cstrings = Cstring[pointer(s) for s in strings]
    GC.@preserve strings cstrings begin
        HDF5.API.h5a_write(
            attr,
            dtype,
            Ptr{Cvoid}(pointer(cstrings)),
        )
    end

    close(attr)
    close(space)
    close(dtype)
    return nothing
end

function physical_bc(og_rhdf)
    p = HDF5.attrs(og_rhdf["Params"])
    rules = Dict("outflow" => "neumann", "periodic" => "periodic", "transmitting" => "transmitting")
    bc(s) = haskey(rules, s) ? rules[s] : error("no padding rule for KHARMA boundary '$s'")
    [(bc(p["Boundaries/inner_x$d"]), bc(p["Boundaries/outer_x$d"])) for d in 1:3]
end

# TODO
# Treat "Info/dt", "Params/Globals/dt_last", and timestep-control settings such as "GRMHD/dt_start", "dt_min", "max_dt_increase", and "start_dt_light" carefully:
# a finer grid can require a smaller timestep. Don’t blindly reuse the old timestep value; set or recompute it consistently for the resized restart. 
# WallTime and run counters such as "NBDel", "NBNew", and "NCycle" are generally bookkeeping values to reset or regenerate.
function write_rhdf(
    cfg,
    x1_faces, x2_faces, x3_faces,       # face coordinates
    x1_centers, x2_centers, x3_centers, # center coordinates
    prim_rho, prim_u, prim_uvec,        # density, energy, velocity field
    cons_fb;                            # cleaned B-field
    kill_policy = true                  # remove existing outfile
    )
    """
        write new rhdf file from resized values
    """
    println("-----------------------------------------")
    # --------------------------------------------
    # --------------------------------------------
    # 
    mb_nx = (cfg.mesh_nx1, cfg.mesh_nx2, cfg.mesh_nx3)
    infile = cfg.infile_rhdf
    outfile = cfg.outfile_rhdf
    abspath(infile) == abspath(outfile) && throw(ArgumentError("outfile must differ from infile"))
    
    og_rhdf = HDF5.h5open(infile, "r" )
    if isfile(outfile)
        if kill_policy
            println("Warning: Outfile already exist, deleting...")
            rm(outfile)
        else
            throw(ArgumentError("outfile '$(outfile)' already exist and kill policy is false"))

        end
    end

    resized_rhdf = HDF5.h5open(outfile,"w")
    # --------------------------------------------
    # --------------------------------------------
    # get conservative variables
    cons_rho, cons_u, cons_uvec = calculate_cons_vars(cfg.metric_file, prim_rho, prim_u, prim_uvec, cons_fb, "new")
    # --------------------------------------------
    # --------------------------------------------
    # Pad all arrays for NGhost
    BC = physical_bc(og_rhdf)

    NGhost = HDF5.attrs(og_rhdf["Info"])["NGhost"]
    Nx1_old, Nx2_old, Nx3_old = HDF5.attrs(og_rhdf["Info"])["RootGridSize"]

    dx1 = kF.hF.get_dx(x1_faces)
    dx2 = kF.hF.get_dx(x2_faces)
    dx3 = kF.hF.get_dx(x3_faces)
    Nx1, Nx2, Nx3 = size(prim_rho)
    x1_faces   = collect(range(x1_faces[1]   - dx1*NGhost, x1_faces[end]   + dx1*NGhost; length = length(x1_faces)   + 2NGhost))
    x2_faces   = collect(range(x2_faces[1]   - dx2*NGhost, x2_faces[end]   + dx2*NGhost; length = length(x2_faces)   + 2NGhost))
    x3_faces   = collect(range(x3_faces[1]   - dx3*NGhost, x3_faces[end]   + dx3*NGhost; length = length(x3_faces)   + 2NGhost))
    x1_centers = collect(range(x1_centers[1] - dx1*NGhost, x1_centers[end] + dx1*NGhost; length = length(x1_centers) + 2NGhost))
    x2_centers = collect(range(x2_centers[1] - dx2*NGhost, x2_centers[end] + dx2*NGhost; length = length(x2_centers) + 2NGhost))
    x3_centers = collect(range(x3_centers[1] - dx3*NGhost, x3_centers[end] + dx3*NGhost; length = length(x3_centers) + 2NGhost))
    prim_rho  = kF.hF.pad_array(prim_rho;Nx1_bc=BC[1],Nx2_bc=BC[2],Nx3_bc=BC[3],NGhost=NGhost) 
    prim_u    = kF.hF.pad_array(prim_u;Nx1_bc=BC[1],Nx2_bc=BC[2],Nx3_bc=BC[3],NGhost=NGhost)  

    cons_rho  = kF.hF.pad_array(cons_rho;Nx1_bc=BC[1],Nx2_bc=BC[2],Nx3_bc=BC[3],NGhost=NGhost)  
    cons_u    = kF.hF.pad_array(cons_u;Nx1_bc=BC[1],Nx2_bc=BC[2],Nx3_bc=BC[3],NGhost=NGhost)  

    prim_uvec1 = kF.hF.pad_array(prim_uvec[:,:,:,1];Nx1_bc=BC[1],Nx2_bc=BC[2],Nx3_bc=BC[3],NGhost=NGhost)        
    prim_uvec2 = kF.hF.pad_array(prim_uvec[:,:,:,2];Nx1_bc=BC[1],Nx2_bc=BC[2],Nx3_bc=BC[3],NGhost=NGhost,transmit_sign=-1)        
    prim_uvec3 = kF.hF.pad_array(prim_uvec[:,:,:,3];Nx1_bc=BC[1],Nx2_bc=BC[2],Nx3_bc=BC[3],NGhost=NGhost,transmit_sign=-1)        
    prim_uvec = cat(prim_uvec1,prim_uvec2,prim_uvec3, dims=4)

    cons_uvec1 = kF.hF.pad_array(cons_uvec[:,:,:,1];Nx1_bc=BC[1],Nx2_bc=BC[2],Nx3_bc=BC[3],NGhost=NGhost)
    cons_uvec2 = kF.hF.pad_array(cons_uvec[:,:,:,2];Nx1_bc=BC[1],Nx2_bc=BC[2],Nx3_bc=BC[3],NGhost=NGhost,transmit_sign=-1)
    cons_uvec3 = kF.hF.pad_array(cons_uvec[:,:,:,3];Nx1_bc=BC[1],Nx2_bc=BC[2],Nx3_bc=BC[3],NGhost=NGhost,transmit_sign=-1)
    cons_uvec = cat(cons_uvec1,cons_uvec2,cons_uvec3, dims=4)
    
    # Pad face array
    # print("Before NGhosts: ")
    # kF.report_max_divB(cons_fb,dx1, dx2, dx3,0)

    

    cons_fb1 = cons_fb[1:Nx1 + 1, 1:Nx2, 1:Nx3, 1]
    cons_fb2 = cons_fb[1:Nx1, 1:Nx2 + 1, 1:Nx3, 2]
    cons_fb3 = cons_fb[1:Nx1, 1:Nx2, 1:Nx3 + 1, 3]
    # @show maximum(abs, cons_fb2[:, 1, :])
    # @show maximum(abs, cons_fb2[:, end, :])
    cons_fb1, cons_fb2, cons_fb3 = kF.hF.pad_face_array(
        cons_fb1, cons_fb2, cons_fb3;
        Nx1_bc=BC[1],Nx2_bc=BC[2],Nx3_bc=BC[3],
        NGhost=NGhost,
    )
    

    # @show(minimum(prim_rho) > 0)
    # @show(!any(isnan, cons_u))
    # println("Padding occured...")
    # jlo = NGhost + 1
    # jhi = size(cons_fb2, 2) - NGhost

    # @show maximum(abs, cons_fb2[:, jlo, :])
    # @show maximum(abs, cons_fb2[:, jhi, :])
    Nx1_face_padded = Nx1+2*NGhost
    Nx2_face_padded = Nx2+2*NGhost
    Nx3_face_padded = Nx3+2*NGhost

    
    cons_fb = zeros(Float64,Nx1_face_padded+1,Nx2_face_padded+1,Nx3_face_padded+1,3)
    cons_fb[1:Nx1_face_padded + 1, 1:Nx2_face_padded, 1:Nx3_face_padded, 1] .= cons_fb1 
    cons_fb[1:Nx1_face_padded, 1:Nx2_face_padded + 1, 1:Nx3_face_padded, 2] .= cons_fb2 
    cons_fb[1:Nx1_face_padded, 1:Nx2_face_padded, 1:Nx3_face_padded + 1, 3] .= cons_fb3
    # print("After NGhosts: ")
    # kF.report_max_divB(cons_fb,dx1, dx2, dx3,NGhost)
    # reshape

    x1_faces = reshape(x1_faces, length(x1_faces), 1)
    x2_faces = reshape(x2_faces, length(x2_faces), 1)
    x3_faces = reshape(x3_faces, length(x3_faces), 1)
    x1_centers = reshape(x1_centers, length(x1_centers), 1)
    x2_centers = reshape(x2_centers, length(x2_centers), 1)
    x3_centers = reshape(x3_centers, length(x3_centers), 1)
    cons_rho  =reshape(cons_rho, size(cons_rho,1),size(cons_rho,2),size(cons_rho,3),1)
    cons_u    =reshape(cons_u, size(cons_u,1),size(cons_u,2),size(cons_u,3),1)
    cons_uvec =reshape(cons_uvec , size(cons_uvec ,1),size(cons_uvec ,2),size(cons_uvec,3),size(cons_uvec,4),1)

    prim_u_shape = size(prim_u)
    prim_u = reshape(prim_u,prim_u_shape[1],prim_u_shape[2],prim_u_shape[3],1)

    prim_rho_shape = size(prim_rho)
    prim_rho = reshape(prim_rho,prim_rho_shape[1],prim_rho_shape[2],prim_rho_shape[3],1)

    uvec_shape = size(prim_uvec)
    prim_uvec = reshape(prim_uvec,uvec_shape[1],uvec_shape[2],uvec_shape[3],uvec_shape[4],1)

    fb_shape = size(cons_fb)
    cons_fb = reshape(cons_fb,fb_shape[1],fb_shape[2],fb_shape[3],fb_shape[4],1)


    mesh_info=Meshblocks.split_meshblocks(
    x1_faces, x2_faces, x3_faces,
    x1_centers, x2_centers, x3_centers,
    cons_rho, cons_u, cons_uvec,
    prim_rho, prim_u, prim_uvec,
    cons_fb,
    mb_nx, NGhost;
    x2_transmitting = true,
    )
    (;
    # datasets (block index last, as HDF5.read returns them)
    x1_faces,x2_faces,x3_faces,
    x1_centers,x2_centers,x3_centers,
    cons_rho,cons_u,cons_uvec,
    prim_rho,prim_u,prim_uvec,cons_fb,
    # Blocks/
    loc5,lx123,xmin,
    # Info/ attributes
    BlocksPerPE,NumMeshBlocks,RootLevel,MaxLevel,
    Multilevel,RootGridSize,MeshBlockSize,
    # top-level datasets
    Levels,LogicalLocations,
    # for updating <parthenon/meshblock> in Input/File
    meshblock_nx,blocks_per_dim,
    ) = mesh_info


    # --------------------------------------------
    # --------------------------------------------
    # Blocks
    #   loc.level-gid-lid-cnghost-gflag: dataset
    #   loc.lx123:                       dataset
    #   xmin:                            dataset
    resized_rhdf["Blocks/loc.level-gid-lid-cnghost-gflag"] = loc5
    resized_rhdf["Blocks/loc.lx123"] = lx123 
    resized_rhdf["Blocks/xmin"] = xmin



    # --------------------------------------------
    # --------------------------------------------
    # Info
    #   BlocksPerPE:         attributes
    #   BoundaryConditions:  attributes
    #   ComponentNames:      attributes
    #   Coordinates:         attributes
    #   IncludesGhost:       attributes
    #   MaxLevel:            attributes
    #   MeshBlockSize:       attributes
    #   Multilevel:          attributes
    #   NBDel:               attributes
    #   NBNew:               attributes
    #   NCycle:              attributes
    #   NGhost:              attributes
    #   NumComponents:       attributes
    #   NumDims:             attributes
    #   NumMeshBlocks:       attributes
    #   OutputDatasetNames:  attributes
    #   OutputFormatVersion: attributes
    #   Refine:              attributes
    #   RootGridDomain:      attributes
    #   RootGridSize:        attributes
    #   RootLevel:           attributes
    #   Time:                attributes
    #   WallTime:            attributes
    #   dt:                  attributes

    # "BlocksPerPE","MeshBlockSize","MaxLevel","Multilevel","NumMeshBlocks","RootGridDomain","RootGridSize","RootLevel"
    HDF5.create_group(resized_rhdf, "Info")
    for key in keys(HDF5.attrs(og_rhdf["Info"]))
        if !(key in ["BlocksPerPE", "MeshBlockSize", "MaxLevel", "Multilevel", "NumMeshBlocks", "RootGridSize", "RootLevel","dt"])
            HDF5.attributes(resized_rhdf["Info"])[key] = HDF5.attrs(og_rhdf["Info"])[key]
        end
    end 

    # HDF5.attributes(resized_rhdf["Info"])["RootGridDomain"] # TODO Check to make sure this does not change
    HDF5.attributes(resized_rhdf["Info"])["RootGridSize"] = RootGridSize
    HDF5.attributes(resized_rhdf["Info"])["MeshBlockSize"] = MeshBlockSize
    HDF5.attributes(resized_rhdf["Info"])["Multilevel"] = Multilevel
    HDF5.attributes(resized_rhdf["Info"])["NumMeshBlocks"] = NumMeshBlocks
    HDF5.attributes(resized_rhdf["Info"])["RootLevel"] = RootLevel
    HDF5.attributes(resized_rhdf["Info"])["BlocksPerPE"] = BlocksPerPE
    HDF5.attributes(resized_rhdf["Info"])["MaxLevel"] = MaxLevel

    dt_new, dt_last_new = kF.hF.new_dt(Nx1,Nx2,Nx3,Nx1_old,Nx2_old,Nx3_old,HDF5.attrs(og_rhdf["Info"])["dt"],HDF5.attrs(og_rhdf["Params"])["Globals/dt_last"])

    HDF5.attributes(resized_rhdf["Info"])["dt"] = dt_new
    

    # --------------------------------------------
    # --------------------------------------------
    # Input
    #   File: attributes
    #       |input_par_string = HDF5.attrs(og_rhdf["Input"])["File"]
    #       |println(input_par_string)
    input_par_string = HDF5.attrs(og_rhdf["Input"])["File"]
    output_par_string = kF.update_par(
        input_par_string,Nx1,Nx2,Nx3;mesh_Nx1=meshblock_nx[1],mesh_Nx2=meshblock_nx[2],mesh_Nx3=meshblock_nx[3]
        )
    # resized_rhdf["Input/File"] = 
    HDF5.create_group(resized_rhdf, "Input")
    input_group = resized_rhdf["Input"]
    HDF5.attributes(resized_rhdf["Input"])["File"] = output_par_string
    write_par_file_attribute!(input_group, output_par_string)
    


    # --------------------------------------------
    # --------------------------------------------
    # Levels
    resized_rhdf["Levels"] = Levels

    # --------------------------------------------
    # --------------------------------------------
    # Locations
    #   x: dataset
    #   y: dataset
    #   z: dataset

    resized_rhdf["Locations/x"] = x1_faces
    resized_rhdf["Locations/y"] = x2_faces
    resized_rhdf["Locations/z"] = x3_faces
    # --------------------------------------------
    # --------------------------------------------
    # LogicalLocations
    resized_rhdf["LogicalLocations"] = LogicalLocations

    # --------------------------------------------
    # --------------------------------------------
    # Params
    og_params = HDF5.attrs(og_rhdf["Params"])
    HDF5.create_group(resized_rhdf, "Params")
    

    for key in keys(og_params)
        if !(key in ["Globals/dt_last"])
            HDF5.attributes(resized_rhdf["Params"])[key] = og_params[key]
        end
    end

    HDF5.attributes(resized_rhdf["Params"])["Globals/dt_last"] = dt_last_new

    # --------------------------------------------
    # --------------------------------------------
    # VolumeLocations
    #   x: dataset
    #   y: dataset
    #   z: dataset


    resized_rhdf["VolumeLocations/x"] = x1_centers
    resized_rhdf["VolumeLocations/y"] = x2_centers
    resized_rhdf["VolumeLocations/z"] = x3_centers
    # --------------------------------------------
    # --------------------------------------------
    # cons.fB
    
    # kF.report_max_divB(cons_fb,dx1, dx2, dx3,NGhost)
    
    resized_rhdf["cons.fB"] = cons_fb

    # --------------------------------------------
    # --------------------------------------------
    # cons.rho, cons.u,cons.uvec

    resized_rhdf["cons.rho"] = cons_rho
    resized_rhdf["cons.u"] = cons_u
    resized_rhdf["cons.uvec"] = cons_uvec

    # --------------------------------------------
    # --------------------------------------------
    # prims.rho
    
    resized_rhdf["prims.rho"] = prim_rho

    # --------------------------------------------
    # --------------------------------------------
    # prims.u
    resized_rhdf["prims.u"] = prim_u

    # --------------------------------------------
    # --------------------------------------------
    # prims.uvec

    resized_rhdf["prims.uvec"] = prim_uvec

    close(resized_rhdf)
    close(og_rhdf)
    # --------------------------------------------
    # --------------------------------------------
    # correct string types
    HDF5.h5open(outfile, "r+") do h5
        info = h5["Info"]
        params = h5["Params"]

        for name in (
            "BoundaryConditions",
            "ComponentNames",
            "Coordinates",
            "OutputDatasetNames",
        )
            rewrite_as_vlen_ascii!(info, name)
        end

        for name in (
            "B_CT/ct_scheme",
            "Boundaries/inner_x1",
            "Boundaries/inner_x2",
            "Boundaries/inner_x3",
            "Boundaries/outer_x1",
            "Boundaries/outer_x2",
            "Boundaries/outer_x3",
            "Driver/name",
            "Globals/SHA1",
            "Globals/branch",
            "Globals/problem",
            "Globals/version",
        )
            rewrite_as_vlen_ascii!(params, name)
        end
    end
    # --------------------------------------------
    # --------------------------------------------

    

    println("File writed: \n   ", outfile)
    



end

function main(cfg)
    resized = resize_grid(
        cfg.infile_rhdf,
        cfg.nx1,
        cfg.nx2,
        cfg.nx3,
        cfg.BC,
        method=cfg.resize_method
    )
    x1_faces, x2_faces, x3_faces,
    x1_centers, x2_centers, x3_centers,
    prim_rho, prim_u, prim_uvec, cons_fb = resized

    # --------------------------------------------
    # --------------------------------------------
    # remove resized B field divergence
    cons_fb = div_cleanup(x1_centers, x2_centers, x3_centers, cons_fb, cfg.BC)

    # --------------------------------------------
    # --------------------------------------------
    # write to new rhdf file
    write_rhdf(
        cfg,                    
        x1_faces, x2_faces, x3_faces,       
        x1_centers, x2_centers, x3_centers, 
        prim_rho, prim_u, prim_uvec, cons_fb,                     
    )
    
end



end 
