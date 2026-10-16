module helperFunctions

import Interpolations

export helperFunctions



function create_interpolations(x1,x2,x3,F,method="linear")
    """
        create an interpolation function from matrix F[x1,x2,x3]

        method: linear = linear interpolation, NN = Nearest Neighbor
    """

    if method == "linear"
        itp = Interpolations.interpolate( 
            (x1,x2,x3), F, Interpolations.Gridded( Interpolations.Linear() ) 
        )
    elseif method == "NN"
        itp = Interpolations.interpolate( 
            (x1,x2,x3), F, Interpolations.Gridded( Interpolations.Constant() ) 
        )
    else
        throw(ArgumentError("Method '$method' not recognized"))
    end
    
    return itp

end

function are_all_values_equal(vector,tolerance = 1e-12)
    """
        check if all values of a matrix are equal
    """
    vector = vec(vector)
    maximum(vector .- vector[1]) < tolerance
end


function get_dx(x)
    """
        given a uniform 1 dimensional coord vector x,
        return the grid width
    """
    diff_x = diff(x)
    @assert are_all_values_equal(diff_x)
    diff_x[1]


end


function reorder_dims(gcon_raw,gcov_raw,gdet_center_raw,gdet_face_raw)
    """
        Takes in contravariant and covariant metic, cell centered and
        face centered gdet = sqrt(- metric determinet) matrices. 
        Converts them to the following
        gcon: [Nx1,Nx2,Nx3,μ,ν]
        gcov:      ^^^^^
        gdet_center: [Nx1, Nx2, Nx3]
        gdet_face: [Nx1, Nx2, Nx3, face_dim]

    """
    gcon = permutedims(gcon_raw, (3, 2, 1, 5, 4))
    gcov = permutedims(gcov_raw, (3, 2, 1, 5, 4))
    gdet_center = permutedims(gdet_center_raw, (3, 2, 1))
    gdet_face = permutedims(gdet_face_raw, (4, 3, 2, 1))
    return gcon,gcov,gdet_center,gdet_face
end

function faceB_to_center(fB,gdet, nx1, nx2, nx3; ng=0)
    """
        convert conservative B field from face centered to primative 
        cell centered B field

        ng: number of ghost cells

        WARNING: This function assumes thats fB and gdet either both 
                 have ghost cells or both don't
    """
    # Set ng=0 for active-only data; use the actual ghost-zone count otherwise.
    ic = (ng + 1):(ng + nx1)
    jc = (ng + 1):(ng + nx2)
    kc = (ng + 1):(ng + nx3)

    Bc = Array{Float64}(undef, nx1, nx2, nx3, 3)

    # Average the two x1 faces surrounding each cell.
    Bc[:, :, :, 1] .= 0.5 .* (
        fB[ic,     jc, kc, 1]./gdet[ic,     jc, kc, 1] .+
        fB[ic .+ 1, jc, kc, 1]./gdet[ic .+ 1, jc, kc, 1]
    )

    # Average the two x2 faces surrounding each cell.
    Bc[:, :, :, 2] .= 0.5 .* (
        fB[ic, jc,     kc, 2]./gdet[ic, jc,     kc, 2] .+
        fB[ic, jc .+ 1, kc, 2]./gdet[ic, jc .+ 1, kc, 2]
    )

    # Average the two x3 faces surrounding each cell.
    Bc[:, :, :, 3] .= 0.5 .* (
        fB[ic, jc, kc,     3]./gdet[ic, jc, kc,     3] .+ 
        fB[ic, jc, kc .+ 1, 3]./gdet[ic, jc, kc .+ 1, 3]
    )

    return Bc
end


function pad_array(A;Nx1_bc=("periodic","periodic"),Nx2_bc=("periodic","periodic"),Nx3_bc=("periodic","periodic"),NGhost=1)
    """
        create ghost zones of width 1

    """
    isVector = false
    if ndims(A) != 3
        throw(ArgumentError("Only matrices of ndims 3 are allowed"))
    end
    Nx1, Nx2, Nx3 = size(A)
    

    padded_A = zeros(eltype(A), Nx1 + 2 * NGhost, Nx2 + 2* NGhost, Nx3 + 2* NGhost)
    padded_A[
        NGhost + 1:end - NGhost,
        NGhost + 1:end - NGhost,
        NGhost + 1:end - NGhost] .= A

    # --------------------------------------------
    # --------------------------------------------
    # Nx1 boundary Conditions [CORNERS ARE NOT USED]

    # left boundary
    if Nx1_bc[1] == "dirichlet"
        padded_A[1:NGhost,:,:] .= .- padded_A[2*NGhost:-1:NGhost+1, :, :]

    elseif Nx1_bc[1] == "neumann"
        padded_A[1:NGhost,:,:] .= padded_A[2*NGhost:-1:NGhost+1,:,:]

    elseif Nx1_bc[1] == "periodic"
        padded_A[1:NGhost, :, :] .= padded_A[end-2*NGhost+1:end-NGhost, :, :]
    else
        throw(ArgumentError("boundary condition of '$Nx1_bc[1]' is not recognized"))
    end

    # right boundary
    if Nx1_bc[2] == "dirichlet"
        padded_A[end-NGhost+1:end,:,:] .= .- padded_A[end-NGhost:-1:end-2*NGhost+1,:,:]

    elseif Nx1_bc[2] == "neumann"
        padded_A[end-NGhost+1:end,:,:] .= padded_A[end-NGhost:-1:end-2*NGhost+1,:,:]

    elseif Nx1_bc[2] == "periodic"
        padded_A[end-NGhost+1:end,:,:] .= padded_A[NGhost + 1:2*NGhost,:,:]
    else
        throw(ArgumentError("boundary condition of '$Nx1_bc[2]' is not recognized"))
    end

    # --------------------------------------------
    # --------------------------------------------
    # Nx2 boundary Conditions
        # left boundary
    if Nx2_bc[1] == "dirichlet"
        padded_A[:,1:NGhost,:] .= .- padded_A[:,2*NGhost:-1:NGhost+1,:]

    elseif Nx2_bc[1] == "neumann"
        padded_A[:,1:NGhost,:] .= padded_A[:,2*NGhost:-1:NGhost+1,:]

    elseif Nx2_bc[1] == "periodic"
        padded_A[:,1:NGhost,:] .= padded_A[:,end-2*NGhost+1:end-NGhost,:]
    else
        throw(ArgumentError("boundary condition of '$Nx2_bc[1]' is not recognized"))
    end

    # right boundary
    if Nx2_bc[2] == "dirichlet"
        padded_A[:,end-NGhost+1:end,:] .= .- padded_A[:,end-NGhost:-1:end-2*NGhost+1,:]

    elseif Nx2_bc[2] == "neumann"
        padded_A[:,end-NGhost+1:end,:] .= padded_A[:,end-NGhost:-1:end-2*NGhost+1,:]

    elseif Nx2_bc[2] == "periodic"
        padded_A[:,end-NGhost+1:end,:] .= padded_A[:,NGhost + 1:2*NGhost,:]
    else
        throw(ArgumentError("boundary condition of '$Nx2_bc[2]' is not recognized"))
    end

    # --------------------------------------------
    # --------------------------------------------
    # Nx3 boundary Conditions
        # left boundary
    if Nx3_bc[1] == "dirichlet"
        padded_A[:,:,1:NGhost] .= .- padded_A[:,:,2*NGhost:-1:NGhost+1]

    elseif Nx3_bc[1] == "neumann"
        padded_A[:,:,1:NGhost] .= padded_A[:,:,2*NGhost:-1:NGhost+1]

    elseif Nx3_bc[1] == "periodic"
        padded_A[:,:,1:NGhost] .= padded_A[:,:,end-2*NGhost+1:end-NGhost]
    else
        throw(ArgumentError("boundary condition of '$Nx3_bc[1]' is not recognized"))
    end

    # right boundary
    if Nx3_bc[2] == "dirichlet"
        padded_A[:,:,end-NGhost+1:end] .= .- padded_A[:,:,end-NGhost:-1:end-2*NGhost+1]

    elseif Nx3_bc[2] == "neumann"
        padded_A[:,:,end-NGhost+1:end] .= padded_A[:,:,end-NGhost:-1:end-2*NGhost+1]

    elseif Nx3_bc[2] == "periodic"
        padded_A[:,:,end-NGhost+1:end] .= padded_A[:,:,NGhost + 1:2*NGhost]
    else
        throw(ArgumentError("boundary condition of '$Nx3_bc[2]' is not recognized"))
    end
 

    return padded_A
end



end