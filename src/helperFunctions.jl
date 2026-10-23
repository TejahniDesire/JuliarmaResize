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
    maximum(abs,vector .- vector[1]) < tolerance
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

function faceB_to_center(fB, gdet, nx1, nx2, nx3; ng=0)
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
    sdiv(a, g) = iszero(g) ? zero(a) : a / g # replaces 0/0 with 0
    
    # Average the two x1 faces surrounding each cell.
    Bc[:, :, :, 1] .= 0.5 .* (
        sdiv.(fB[ic,     jc, kc, 1], gdet[ic,     jc, kc, 1]) .+ 
        sdiv.(fB[ic .+ 1, jc, kc, 1],gdet[ic .+ 1, jc, kc, 1])
    )

    # Average the two x2 faces surrounding each cell.
    Bc[:, :, :, 2] .= 0.5 .* (
        sdiv.(fB[ic, jc,     kc, 2],gdet[ic, jc,     kc, 2]) .+
        sdiv.(fB[ic, jc .+ 1, kc, 2],gdet[ic, jc .+ 1, kc, 2])
    )

    # Average the two x3 faces surrounding each cell.
    Bc[:, :, :, 3] .= 0.5 .* (
        sdiv.(fB[ic, jc, kc,     3],gdet[ic, jc, kc,     3]) .+ 
        sdiv.(fB[ic, jc, kc .+ 1, 3],gdet[ic, jc, kc .+ 1, 3])
    )

    return Bc
end


function pad_array(A;Nx1_bc=("periodic","periodic"),Nx2_bc=("periodic","periodic"),Nx3_bc=("periodic","periodic"),NGhost=1,transmit_sign=1)
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
    # x2 boundaries
    if Nx2_bc[1] == "transmitting"
        fill_transmit_polar_side!(padded_A, :inner, NGhost, transmit_sign)
    elseif Nx2_bc[1] == "dirichlet"
        padded_A[:,1:NGhost,:] .= .- padded_A[:,2*NGhost:-1:NGhost+1,:]

    elseif Nx2_bc[1] == "neumann"
        padded_A[:,1:NGhost,:] .= padded_A[:,2*NGhost:-1:NGhost+1,:]

    elseif Nx2_bc[1] == "periodic"
        padded_A[:,1:NGhost,:] .= padded_A[:,end-2*NGhost+1:end-NGhost,:]
    else
        throw(ArgumentError("boundary condition of '$Nx2_bc[1]' is not recognized"))
    end

    # right boundary
    if Nx2_bc[2] == "transmitting"
        fill_transmit_polar_side!(padded_A, :outer, NGhost, transmit_sign)
    elseif Nx2_bc[2] == "dirichlet"
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

function fill_transmit_polar_side!(A, side, ng, sign)
    n2, n3 = size(A, 2), size(A, 3)
    n3_active = n3 - 2ng

    iseven(n3_active) ||
        throw(ArgumentError("Transmitting polar BC requires an even number of active x3 zones"))

    first2, last2 = ng + 1, n2 - ng
    first3 = ng + 1
    half_turn = n3_active ÷ 2

    for layer in 1:ng
        if side == :inner
            j = first2 - layer
            jsrc = 2first2 - 1 - j
        elseif side == :outer
            j = last2 + layer
            jsrc = 2last2 + 1 - j
        else
            throw(ArgumentError("side must be :inner or :outer"))
        end

        # Map every x3 index, including x3 ghost indices, back into the
        # active x3 range after a half-turn.
        for k in axes(A, 3)
            ksrc = first3 + mod(k - first3 + half_turn, n3_active)

            # Keep x1 unchanged, as KHARMA does.
            for i in axes(A, 1)
                A[i, j, k] = sign * A[i, jsrc, ksrc]
            end
        end
    end
end



# Fill one axis of a staggered array for periodic, Dirichlet, or Neumann BCs.
function _fill_face_axis!(A, axis, nactive, ng, bc, normal_face)
    first_active = ng + 1
    last_active = first_active + nactive + (normal_face ? 1 : 0) - 1

    for side in 1:2
        kind = bc[side]
        is_inner = side == 1

        if kind == "transmitting"
            throw(ArgumentError("transmitting is supported only on x2"))
        elseif !(kind in ("periodic", "dirichlet", "neumann"))
            throw(ArgumentError("Unsupported boundary condition: $kind"))
        end

        for layer in 1:ng
            target = is_inner ? first_active - layer : last_active + layer

            if kind == "periodic"
                # There are nactive unique positions in a periodic direction.
                source = first_active + mod(target - first_active, nactive)
                sign = 1
            else
                sign = kind == "dirichlet" ? -1 : 1

                if normal_face
                    # Reflect across the boundary face itself.
                    boundary_face = is_inner ? first_active : last_active
                    source = 2 * boundary_face - target
                else
                    # Reflect cell-centered locations across the boundary.
                    source = is_inner ?
                        2 * first_active - 1 - target :
                        2 * last_active + 1 - target
                end
            end

            selectdim(A, axis, target) .=
                sign .* selectdim(A, axis, source)
        end
    end
end


# Apply KHARMA's transmitting polar mapping to a face-centered array.
function _fill_transmitting_x2!(A, n2, n3, ng, side, sign; normal_x2_face=false)
    first2 = ng + 1
    last2 = first2 + n2 + (normal_x2_face ? 1 : 0) - 1
    first3 = ng + 1
    half_turn = n3 ÷ 2

    for layer in 1:ng
        if side == 1
            # Inner x2 boundary
            target_j = first2 - layer
            source_j = normal_x2_face ?
                2 * first2 - target_j :
                2 * first2 - 1 - target_j
        else
            # Outer x2 boundary
            target_j = last2 + layer
            source_j = normal_x2_face ?
                2 * last2 - target_j :
                2 * last2 + 1 - target_j
        end

        for k in axes(A, 3)
            # Shift by half the active x3 domain and wrap periodically.
            source_k = first3 + mod(k - first3 + half_turn, n3)

            for i in axes(A, 1)
                A[i, target_j, k] = sign * A[i, source_j, source_k]
            end
        end
    end

    if normal_x2_face
        # KHARMA sets the x2-normal face on the polar boundary to zero.
        boundary_j = side == 1 ? first2 : last2
        A[:, boundary_j, :] .= 0
    end
end


function pad_face_array(
    A1, A2, A3;
    Nx1_bc=("periodic", "periodic"),
    Nx2_bc=("periodic", "periodic"),
    Nx3_bc=("periodic", "periodic"),
    NGhost=1,
    transmit_sign=(1, -1, -1),
)
    ndims(A1) == ndims(A2) == ndims(A3) == 3 ||
        throw(ArgumentError("A1, A2, and A3 must be 3D arrays"))

    Nx1 = size(A1, 1) - 1
    Nx2 = size(A1, 2)
    Nx3 = size(A1, 3)

    @assert size(A2) == (Nx1, Nx2 + 1, Nx3)
    @assert size(A3) == (Nx1, Nx2, Nx3 + 1)
    @assert NGhost >= 0
    @assert NGhost <= min(Nx1, Nx2, Nx3)

    if Nx2_bc[1] == "transmitting" || Nx2_bc[2] == "transmitting"
        iseven(Nx3) ||
            throw(ArgumentError("Transmitting x2 boundaries require an even active Nx3"))
        transmit_sign isa Tuple && length(transmit_sign) == 3 ||
            throw(ArgumentError("transmit_sign must be a 3-tuple, e.g. (1, -1, -1)"))
    end

    # Allocate padded staggered arrays.
    P1 = zeros(eltype(A1), Nx1 + 1 + 2NGhost, Nx2 + 2NGhost,     Nx3 + 2NGhost)
    P2 = zeros(eltype(A2), Nx1     + 2NGhost, Nx2 + 1 + 2NGhost, Nx3 + 2NGhost)
    P3 = zeros(eltype(A3), Nx1     + 2NGhost, Nx2 + 2NGhost,     Nx3 + 1 + 2NGhost)

    # Copy active data into the padded arrays.
    P1[NGhost+1:NGhost+Nx1+1, NGhost+1:NGhost+Nx2,   NGhost+1:NGhost+Nx3]   .= A1
    P2[NGhost+1:NGhost+Nx1,   NGhost+1:NGhost+Nx2+1, NGhost+1:NGhost+Nx3]   .= A2
    P3[NGhost+1:NGhost+Nx1,   NGhost+1:NGhost+Nx2,   NGhost+1:NGhost+Nx3+1] .= A3

    # Fill x1 boundaries. A1 is face-staggered in x1.
    _fill_face_axis!(P1, 1, Nx1, NGhost, Nx1_bc, true)
    _fill_face_axis!(P2, 1, Nx1, NGhost, Nx1_bc, false)
    _fill_face_axis!(P3, 1, Nx1, NGhost, Nx1_bc, false)

    # Fill x3 boundaries. A3 is face-staggered in x3.
    _fill_face_axis!(P1, 3, Nx3, NGhost, Nx3_bc, false)
    _fill_face_axis!(P2, 3, Nx3, NGhost, Nx3_bc, false)
    _fill_face_axis!(P3, 3, Nx3, NGhost, Nx3_bc, true)

    # Fill x2 boundaries. A2 is face-staggered normal to x2.
    for side in 1:2
        if Nx2_bc[side] == "transmitting"
            _fill_transmitting_x2!(
                P1, Nx2, Nx3, NGhost, side, transmit_sign[1];
                normal_x2_face=false,
            )
            _fill_transmitting_x2!(
                P2, Nx2, Nx3, NGhost, side, transmit_sign[2];
                normal_x2_face=true,
            )
            _fill_transmitting_x2!(
                P3, Nx2, Nx3, NGhost, side, transmit_sign[3];
                normal_x2_face=false,
            )
        else
            bc = side == 1 ?
                (Nx2_bc[1], Nx2_bc[1]) :
                (Nx2_bc[2], Nx2_bc[2])

            _fill_face_axis!(P1, 2, Nx2, NGhost, bc, false)
            _fill_face_axis!(P2, 2, Nx2, NGhost, bc, true)
            _fill_face_axis!(P3, 2, Nx2, NGhost, bc, false)
        end
    end

    return P1, P2, P3
end


function new_dt(Nx1,Nx2,Nx3,Nx1_old,Nx2_old,Nx3_old,dt_old,dt_last_old)
    ratio = minimum((Nx1_old / Nx1, Nx2_old / Nx2, Nx3_old / Nx3))
    safety = 0.9          # optional margin, since the scaling isn't exact

    dt_new = safety * min(ratio, 1.0) * dt_old

    dt_last_new = safety * min(ratio, 1.0) * dt_last_old

    return dt_new, dt_last_new

end

end