module kharmaFunctions



export kharmaFunctions

include("helperFunctions.jl")

import .helperFunctions as hF
import DelimitedFiles
import LinearMaps
import IterativeSolvers
import LinearAlgebra

function load_kharma_hst(filepath::String)
    # 1. Read all lines to parse the header
    lines = readlines(filepath)
    
    # Target the specific line containing the column mapping e.g., "=time"
    header_line = ""
    for line in lines
        if startswith(line, "#") && occursin(r"\[\d+\]=", line)
            header_line = line
            break
        end
    end
    
    if isempty(header_line)
        error("Could not find a valid KHARMA header line matching '[index]=name' format.")
    end
    
    # Use a regex to extract just the column names following the '=' symbol
    headers = [String(m.captures[1]) for m in eachmatch(r"\[\d+\]=([^\s\=\[\]]+)", header_line)]

    # 2. Read the numeric data matrix (omitting ' ' fixes the multiple-space issue)
    data = DelimitedFiles.readdlm(filepath, Float64, comments=true, header=false)
    
    # 3. Construct the dictionary mapping clean headers to column arrays
    hst_dict = Dict{String, Vector{Float64}}()
    for (i, name) in enumerate(headers)
        if i <= size(data, 2)
            hst_dict[name] = data[:, i]
        end
    end
    
    return hst_dict
end

function p_to_u_arrays(rho, uu, uvec, B, gcon, gcov, gdet, gamma_ad)
    """
        Convert primative vars density, energy, velocity, and magnetic field
        to conservative variables using KHARMA's conventions

        gcon: contravariant metric [Nx1,Nx2,Nx3,μ,ν]
        gcov: contravariant metric     ^^^^^

        B is cell centered
    """
    nx1, nx2, nx3 = size(rho)
    @assert size(uu)   == (nx1, nx2, nx3)
    @assert size(uvec) == (nx1, nx2, nx3, 3)
    @assert size(B)    == (nx1, nx2, nx3, 3)
    @assert size(gcon) == (nx1, nx2, nx3, 4, 4)
    @assert size(gcov) == (nx1, nx2, nx3, 4, 4)
    @assert size(gdet) == (nx1, nx2, nx3)

    cons_rho = similar(rho, Float64)
    cons_u = similar(rho, Float64)
    cons_uvec = similar(uvec, Float64)

    for k in 1:nx3, j in 1:nx2, i in 1:nx1
        # Spatial primitive four-velocity and metric γ_ij = g_ij.
        qsq = 0.0
        for a in 1:3, b in 1:3
            qsq += gcov[i,j,k,a+1,b+1] * uvec[i,j,k,a] * uvec[i,j,k,b]
        end
        lorentz = sqrt(1.0 + qsq)

        alpha = 1.0 / sqrt(-gcon[i,j,k,1,1])

        # Contravariant fluid four-velocity.
        ucon = zeros(Float64, 4)
        ucov = zeros(Float64, 4)
        ucon[1] = lorentz / alpha
        for a in 1:3
            ucon[a+1] = uvec[i,j,k,a] -
                         lorentz * alpha * gcon[i,j,k,1,a+1]
        end

        # Lower its index: u_μ = g_μν u^ν.
        for μ in 1:4, ν in 1:4
            ucov[μ] += gcov[i,j,k,μ,ν] * ucon[ν]
        end

        # Gamma-law EOS: p = (gamma_ad - 1) * uu.
        pgas = (gamma_ad - 1.0) * uu[i,j,k]

        # Magnetic four-vector, following KHARMA's GRMHD calc_4vecs.
        bcon = zeros(Float64, 4)
        bcov = zeros(Float64, 4)

        for a in 1:3
            bcon[1] += B[i,j,k,a] * ucov[a+1]
        end
        for a in 1:3
            bcon[a+1] = (B[i,j,k,a] + bcon[1] * ucon[a+1]) / ucon[1]
        end
        for μ in 1:4, ν in 1:4
            bcov[μ] += gcov[i,j,k,μ,ν] * bcon[ν]
        end

        bsq = sum(bcon[μ] * bcov[μ] for μ in 1:4)
        eta = pgas + rho[i,j,k] + uu[i,j,k] + bsq
        ptot = pgas + 0.5 * bsq

        # Time row T^0_μ of the GRMHD stress-energy tensor.
        T0 = zeros(Float64, 4)
        for μ in 1:4
            delta0μ = (μ == 1) ? 1.0 : 0.0
            T0[μ] = eta * ucon[1] * ucov[μ] +
                    ptot * delta0μ -
                    bcon[1] * bcov[μ]
        end

        gd = gdet[i,j,k]
        cons_rho[i,j,k] = rho[i,j,k] * ucon[1] * gd
        cons_u[i,j,k] = T0[1] * gd + cons_rho[i,j,k]
        for a in 1:3
            cons_uvec[i,j,k,a] = T0[a+1] * gd
        end
    end

    return cons_rho, cons_u, cons_uvec
end

function cell_width(d, axis, i, j, k)
    """
        return width of cell in axis direction at coordinated i,k,k
    """
    d isa Number && return d

    if ndims(d) == 1
        return d[axis == 1 ? i : axis == 2 ? j : k]
    elseif ndims(d) == 3
        return d[i, j, k]
    else
        throw(ArgumentError("Widths must be scalars, 1D vectors, or 3D arrays"))
    end
end


function get_real_cells(array,NGhost)
    """
        return only active cells
        assumes size(array) = [Nx1 + 2Ghost cells, Nx2 + 2Ghost cells, Nx3 + 2Ghost cells]
    """
    @assert ndims(array) == 3

    return array[NGhost + 1:end - NGhost,NGhost + 1:end - NGhost,NGhost + 1:end - NGhost]

end

function get_real_cells_vector(array,NGhost)
    """
        return only active cells
        assumes size(array) = [Nx1 + 2Ghost cells, Nx2 + 2Ghost cells, Nx3 + 2Ghost cells, 3]
    """
    @assert ndims(array) == 4

    return array[NGhost + 1:end - NGhost,NGhost + 1:end - NGhost,NGhost + 1:end - NGhost,:]

end


function kharma_face_div(fB, dx1, dx2, dx3)
    # Assumes fB has Julia layout (x1, x2, x3, component),
    # e.g. size (193, 97, 97, 3) for a (192, 96, 96) cell grid.
    @assert ndims(fB) == 4
    @assert size(fB, 4) == 3

    nx, ny, nz = size(fB, 1) - 1, size(fB, 2) - 1, size(fB, 3) - 1
    T = promote_type(
        eltype(fB),
        dx1 isa Number ? typeof(dx1) : eltype(dx1),
        dx2 isa Number ? typeof(dx2) : eltype(dx2),
        dx3 isa Number ? typeof(dx3) : eltype(dx3),
    )
    divB = Array{T}(undef, nx, ny, nz)

    # for k in 1:nz, j in 1:ny, i in 1:nx
    Threads.@threads for idx in CartesianIndices((1:nx, 1:ny, 1:nz))
        i = idx[1]
        j = idx[2]
        k = idx[3]
        divB[i, j, k] =
            (fB[i + 1, j,     k,     1] - fB[i, j,     k,     1]) /
                cell_width(dx1, 1, i, j, k) +
            (fB[i,     j + 1, k,     2] - fB[i,     j,     k,     2]) /
                cell_width(dx2, 2, i, j, k) +
            (fB[i,     j,     k + 1, 3] - fB[i,     j,     k,     3]) /
                cell_width(dx3, 3, i, j, k)
    end

    return divB
end

function report_max_divB(cons_fB,dx1, dx2, dx3,NGhost)
    println("max(∇⋅B): ",
        maximum(abs.(
            kharma_face_div(get_real_cells_vector(cons_fB ,NGhost), dx1, dx2, dx3)
            ))
    )

end


function face_grad(P, direction, i, j, k, dx1, dx2, dx3)
    """
        face_grad(P, direction, i, j, k, dx1, dx2, dx3)

        Backward gradient of cell-centered scalar field `P` along `direction`,
        matching KHARMA's `face_grad`.

        `direction` is `1`, `2`, or `3`. Each spacing can be a scalar, a 1D
        vector of widths along that axis, or a 3D array of local widths.
    """
    if direction == 1
        Δx = cell_width(dx1, 1, i, j, k)
        return (P[i, j, k] - P[i-1, j, k]) / Δx
    elseif direction == 2
        Δx = cell_width(dx2, 2, i, j, k)
        return (P[i, j, k] - P[i, j-1, k]) / Δx
    elseif direction == 3
        Δx = cell_width(dx3, 3, i, j, k)
        return (P[i, j, k] - P[i, j, k-1]) / Δx
    else
        throw(ArgumentError("direction must be 1, 2, or 3"))
    end
end

function gradient(A, dx1, dx2, dx3, BC)
    """
        calculate discrete gradient operator: ∇A
        BC: Boundary Conditions = [
            (lower_x1,uppper_x1),
            (lower_x2,uppper_x2)
            (lower_x2,uppper_x3)
        ]
        
    """
    # assess boundary conditions:
    Nx, Ny, Nz = size(A)
    grad_A = zeros(Float64,Nx+1,Ny+1,Nz+1,3)
    # Active ranges
    # B1: 193 × 96 × 96
    # B2: 192 × 97 × 96
    # B3: 192 × 96 × 97
    A_padded = hF.pad_array(A;Nx1_bc=BC[1],Nx2_bc=BC[2],Nx3_bc=BC[3])
    Threads.@threads for idx in CartesianIndices((1:Nx+1, 1:Ny, 1:Nz))
        # upper and lower ghost cells used for first call A_padded[2,-] - A_padded[1,-] 
        # and last call A_padded[Nx+2,-] - A_padded[Nx+1,-]
        i = idx[1]
        j = idx[2]
        k = idx[3]
        grad_A[i,j,k,1] = face_grad(A_padded, 1, i+1, j+1, k+1, dx1, dx2, dx3)
    end

    Threads.@threads for idx in CartesianIndices((1:Nx, 1:Ny+1, 1:Nz))
        # upper and lower ghost cells used for first call A_padded[2,-] - A_padded[1,-] 
        # and last call A_padded[Nx+2,-] - A_padded[Nx+1,-]
        i = idx[1]
        j = idx[2]
        k = idx[3]
        grad_A[i,j,k,2] = face_grad(A_padded, 2, i+1, j+1, k+1, dx1, dx2, dx3)
    end

    Threads.@threads for idx in CartesianIndices((1:Nx, 1:Ny, 1:Nz+1))
        # upper and lower ghost cells used for first call A_padded[2,-] - A_padded[1,-] 
        # and last call A_padded[Nx+2,-] - A_padded[Nx+1,-]
        i = idx[1]
        j = idx[2]
        k = idx[3]
        grad_A[i,j,k,3] = face_grad(A_padded, 3, i+1, j+1, k+1, dx1, dx2, dx3)
    end


    return grad_A
end


function laplacien(phi,dx1,dx2,dx3,BC)
    """
        calculate discrete laplacien operator: ∇²ϕ
        BC: Boundary Conditions = [
            (lower_x1,uppper_x1),
            (lower_x2,uppper_x2)
            (lower_x2,uppper_x3)
        ]
    """
    @assert ndims(phi) == 3
    Nx1, Nx2, Nx3 = size(phi)
    phi = hF.pad_array(phi;Nx1_bc=BC[1],Nx2_bc=BC[2],Nx3_bc=BC[3])
    T = promote_type(
        eltype(phi),
        dx1 isa Number ? typeof(dx1) : eltype(dx1),
        dx2 isa Number ? typeof(dx2) : eltype(dx2),
        dx3 isa Number ? typeof(dx3) : eltype(dx3),
    )
    laplac_phi = Array{T}(undef, Nx1, Nx2, Nx3)

    # for i in 1:Nx1, j in 1:Nx2, k in 1:Nx3
    Threads.@threads for idx in CartesianIndices((1:Nx1, 1:Nx2, 1:Nx3))
        i = idx[1]
        j = idx[2]
        k = idx[3]
        i_pad = i + 1
        j_pad = j + 1
        k_pad = k + 1
        laplac_phi[i,j,k] = 
            (phi[i_pad+1,j_pad,k_pad] - 2*phi[i_pad,j_pad,k_pad] + phi[i_pad-1,j_pad,k_pad]) / dx1^2 + 
            (phi[i_pad,j_pad+1,k_pad] - 2*phi[i_pad,j_pad,k_pad] + phi[i_pad,j_pad-1,k_pad]) / dx2^2 + 
            (phi[i_pad,j_pad,k_pad+1] - 2*phi[i_pad,j_pad,k_pad] + phi[i_pad,j_pad,k_pad-1]) / dx3^2  
    end

    return laplac_phi

end


# TODO: The below function is chatgpt. Confirm this is valid
# function solve_phi(fB, dx1, dx2, dx3, BC; reltol=1e-10,abstol=1e-12, maxiter=10_000)
#     divB = kharma_face_div(fB, dx1, dx2, dx3)
#     dims = size(divB)
#     n = length(divB)
#     T = promote_type(eltype(divB), Float64)

#     function apply_neg_laplacian!(y, x)
#         phi = reshape(x, dims)
#         y .= -vec(laplacien(phi, dx1, dx2, dx3, BC))
#         return y
#     end

#     A = LinearMaps.LinearMap{T}(
#         apply_neg_laplacian!, n, n;
#         ismutating=true,
#         issymmetric=true,
#         ishermitian=true,
#     )

#     rhs = -T.(vec(divB))
#     println("Solving for ϕ in ∇⋅B=∇²ϕ...")
#     phi_vec, history = IterativeSolvers.cg(
#     A, rhs;
#     reltol=reltol,
#     abstol=abstol,
#     maxiter=maxiter,
#     log=true,
#     )
#     println("ϕ found.")

#     phi = reshape(phi_vec, dims)
#     residual = laplacien(phi, dx1, dx2, dx3, BC) .- divB
#     residual_norm = LinearAlgebra.norm(residual)

#     println("Final Residual Norm: ", residual_norm)

#     return phi, residual, residual_norm, history
# end
function solve_phi(fB, dx1, dx2, dx3, BC;
                   reltol=1e-10, abstol=1e-12, maxiter=10_000)

    divB = kharma_face_div(fB, dx1, dx2, dx3)
    Nx1, Nx2, Nx3 = size(divB)
    dims = size(divB)
    n = length(divB)
    T = promote_type(eltype(divB), Float64)

    # Divergence contribution from prescribing grad(phi) = fB
    # on the lower and upper x2 boundary faces.
    div_boundary = zeros(T, dims)
    div_boundary[:, 1, :] .-= fB[1:Nx1, 1,      1:Nx3, 2] ./ dx2
    div_boundary[:, end, :] .+= fB[1:Nx1, Nx2+1, 1:Nx3, 2] ./ dx2

    function apply_neg_laplacian!(y, x)
        phi = reshape(x, dims)
        grad_phi = gradient(phi, dx1, dx2, dx3, BC)

        # Homogeneous boundary condition for the linear CG operator.
        # Only these valid x2 face locations contribute to face_div.
        grad_phi[1:Nx1, 1,      1:Nx3, 2] .= 0
        grad_phi[1:Nx1, Nx2+1, 1:Nx3, 2] .= 0

        y .= -vec(kharma_face_div(grad_phi, dx1, dx2, dx3))
        return y
    end

    A = LinearMaps.LinearMap{T}(
        apply_neg_laplacian!, n, n;
        ismutating=true,
        issymmetric=true,
        ishermitian=true,
    )

    # Solve -L₀(phi) = -div(B) + div_boundary.
    rhs = -T.(vec(divB)) .+ vec(div_boundary)

    println("Solving for ϕ in ∇⋅B=∇²ϕ with zero corrected x2 boundary flux...")
    phi_vec, history = IterativeSolvers.cg(
        A, rhs;
        reltol=reltol,
        abstol=abstol,
        maxiter=maxiter,
        log=true,
    )

    phi = reshape(phi_vec, dims)

    # Restore the inhomogeneous boundary gradients needed to cancel fB.
    grad_phi = gradient(phi, dx1, dx2, dx3, BC)
    grad_phi[1:Nx1, 1,      1:Nx3, 2] .= fB[1:Nx1, 1,      1:Nx3, 2]
    grad_phi[1:Nx1, Nx2+1, 1:Nx3, 2] .= fB[1:Nx1, Nx2+1, 1:Nx3, 2]

    residual = kharma_face_div(grad_phi, dx1, dx2, dx3) .- divB
    residual_norm = LinearAlgebra.norm(residual)

    println("Final Residual Norm: ", residual_norm)
    return phi, residual, residual_norm, history
end


function update_par(par_string,Nx1, Nx2,Nx3;mesh_Nx1=nothing,mesh_Nx2=nothing,mesh_Nx3=nothing)
    """
        update KHARMA .par file with new Nx1, Nx2, Nx3 values for mesh and meshblock
    """
    if isnothing(mesh_Nx1)
        mesh_Nx1 = Nx1
    end
    if isnothing(mesh_Nx2)
        mesh_Nx2 = Nx2
    end
    if isnothing(mesh_Nx3)
        mesh_Nx3 = Nx3
    end
    meshDict = Dict(
        "<parthenon/mesh>" => Dict("nx1" => Nx1, "nx2" => Nx2, "nx3" => Nx3),
        "<parthenon/meshblock>" => Dict("nx1" => mesh_Nx1, "nx2" => mesh_Nx2, "nx3" => mesh_Nx3)
        )

    new_string = ""
    lines = split(par_string,"\n")
    current_header = nothing
    for line in lines
        if all(isspace, line)
            new_string *= line * "\n"
            continue
            
        end
        if line[1] == '<' 
            current_header = line
            # println(current_header in keys(meshDict))
        end
         
        if (current_header in keys(meshDict))
            m = match(r"nx\d", line)

            if !isnothing(m)
                m2 = match(r"\s+\d+", line)
                line = replace(line, replace(m2.match, r"\s" => "") => meshDict[current_header][m.match])
            end
            
        end
        new_string *= line * "\n"

        # println(line)

    end
    return new_string
end

end