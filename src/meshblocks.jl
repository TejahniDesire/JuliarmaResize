module Meshblocks

import HDF5
export Meshblocks

# WARNING: The code below was written by Claude and should be internally verified.
# I've found that its results are consistent for comparisons to KHARMA calculations,
# that is, it produces the correct meshblock information when increasing the number
# of meshblocks in comparison to a KHARMA simulation who started at the final meshblock count,
# but I have not verified the code line by line. 
"""
    merge_meshblocks(x1_faces, x2_faces, x3_faces,
                     x1_centers, x2_centers, x3_centers,
                     prim_rho, prim_u, prim_uvec, cons_fb,
                     lx123, NGhost)

Stitch a uniform (non-AMR) multi-meshblock Parthenon/KHARMA restart into the
layout it would have with a single meshblock covering the whole mesh.

Inputs are the arrays exactly as `HDF5.read` returns them (block index last):

    x1_faces   (nb1 + 2ng + 1, nblocks)        Locations/x
    x2_faces   (nb2 + 2ng + 1, nblocks)        Locations/y
    x3_faces   (nb3 + 2ng + 1, nblocks)        Locations/z
    x1_centers (nb1 + 2ng,     nblocks)        VolumeLocations/x   (same for y, z)
    prim_rho   (nb1+2ng, nb2+2ng, nb3+2ng, nblocks)
    prim_u     (nb1+2ng, nb2+2ng, nb3+2ng, nblocks)
    prim_uvec  (nb1+2ng, nb2+2ng, nb3+2ng, 3, nblocks)
    cons_fb    (nb1+2ng+1, nb2+2ng+1, nb3+2ng+1, 3, nblocks)
    lx123      (3, nblocks)                    Blocks/loc.lx123
    NGhost     Info/NGhost

Each block's interior cells are copied to their place in the global
array. A block's ghost zones are copied only on sides where it touches the physical
domain boundary; internal ghosts (copies of neighbours) are dropped. Face
values shared by two neighbouring blocks are checked for agreement.

Returns a NamedTuple whose arrays carry a trailing block dimension of 1 (the
same shapes HDF5.read gives for a single-meshblock file), plus the
single-block metadata to write into Blocks/ and Info/.
"""
function merge_meshblocks(
    x1_faces, x2_faces, x3_faces,
    x1_centers, x2_centers, x3_centers,
    prim_rho, prim_u, prim_uvec, cons_fb,
    lx123, NGhost;
    rtol = 1e-12,
)
    ng = Int(NGhost)
    nblocks = size(prim_rho, 4)

    @assert ndims(prim_rho) == 4
    @assert size(prim_u) == size(prim_rho)
    @assert size(prim_uvec) == (size(prim_rho)[1:3]..., 3, nblocks)
    @assert size(cons_fb) == (size(prim_rho, 1) + 1, size(prim_rho, 2) + 1,
                              size(prim_rho, 3) + 1, 3, nblocks)
    @assert size(lx123) == (3, nblocks)

    # Interior cells per block and number of blocks along each direction
    nb  = ntuple(d -> size(prim_rho, d) - 2ng, 3)
    nbl = ntuple(d -> Int(maximum(lx123[d, :])) + 1, 3)
    N   = nb .* nbl                       # global interior cells (RootGridSize)
    Np  = N .+ 2ng                        # global cells incl. ghosts

    @assert prod(nbl) == nblocks "lx123 does not describe a full uniform block grid (AMR?)"

    # Index range (local to the block, ghost-inclusive, 1-based) to copy along
    # direction d. `face = true` means this array is staggered (normal) in d.
    function local_range(d, lx, face)
        lo = lx == 0          ? 1 : ng + 1
        hi = lx == nbl[d] - 1 ? nb[d] + 2ng + face : ng + nb[d] + face
        return lo:hi
    end
    # local index i in block at position lx  ->  global index lx*nb + i
    glob(d, lx, r) = (lx * nb[d]) .+ r

    # ---------------- Coordinates ----------------
    faces_in   = (x1_faces, x2_faces, x3_faces)
    centers_in = (x1_centers, x2_centers, x3_centers)
    faces_out   = ntuple(d -> fill(NaN, Np[d] + 1), 3)
    centers_out = ntuple(d -> fill(NaN, Np[d]), 3)

    for b in 1:nblocks, d in 1:3
        lx = Int(lx123[d, b])
        rf = local_range(d, lx, true)
        rc = local_range(d, lx, false)
        gf = glob(d, lx, rf)
        # shared face with the lower neighbour must agree
        if !isnan(faces_out[d][gf[1]])
            isapprox(faces_out[d][gf[1]], faces_in[d][rf[1], b]; rtol = rtol, atol = 1e-14) ||
                error("Locations mismatch on axis $d between blocks at lx=$(lx-1) and lx=$lx")
        end
        faces_out[d][gf]                 .= faces_in[d][rf, b]
        centers_out[d][glob(d, lx, rc)] .= centers_in[d][rc, b]
    end
    for d in 1:3
        any(isnan, faces_out[d])   && error("unfilled face coordinates on axis $d")
        any(isnan, centers_out[d]) && error("unfilled center coordinates on axis $d")
    end

    # ---------------- Cell-centred fields ----------------
    T = eltype(prim_rho)
    rho_out  = zeros(T, Np..., 1)
    u_out    = zeros(T, Np..., 1)
    uvec_out = zeros(eltype(prim_uvec), Np..., 3, 1)

    for b in 1:nblocks
        lx = ntuple(d -> Int(lx123[d, b]), 3)
        r  = ntuple(d -> local_range(d, lx[d], false), 3)
        g  = ntuple(d -> glob(d, lx[d], r[d]), 3)
        rho_out[g..., 1]     .= prim_rho[r..., b]
        u_out[g..., 1]       .= prim_u[r..., b]
        uvec_out[g..., :, 1] .= prim_uvec[r..., :, b]
    end

    # ---------------- Face-centred B ----------------
    # Component c is staggered only along axis c; the other two axes use the
    # cell ranges. Unused padding slots stay zero, as in a single-block file.
    fb_out = zeros(eltype(cons_fb), (Np .+ 1)..., 3, 1)
    filled = falses((Np .+ 1)..., 3)
    maxdiff = 0.0

    for b in 1:nblocks, c in 1:3
        lx = ntuple(d -> Int(lx123[d, b]), 3)
        r  = ntuple(d -> local_range(d, lx[d], d == c), 3)
        g  = ntuple(d -> glob(d, lx[d], r[d]), 3)
        src = view(cons_fb, r..., c, b)
        dst = view(fb_out, g..., c, 1)
        msk = view(filled, g..., c)
        # Faces already written by a neighbour must agree (shared block faces)
        if any(msk)
            scale = max(maximum(abs, src), eps())
            maxdiff = max(maxdiff, maximum(abs.(dst[msk] .- src[msk])) / scale)
        end
        dst .= src
        msk .= true
    end
    maxdiff > 1e-10 && @warn "Shared block faces of cons.fB disagree" max_rel_diff = maxdiff

    # ---------------- Single-block metadata ----------------
    xmin = reshape([faces_out[d][ng + 1] for d in 1:3], 3, 1)   # interior lower corner

    return (
        x1_faces   = reshape(faces_out[1], :, 1),
        x2_faces   = reshape(faces_out[2], :, 1),
        x3_faces   = reshape(faces_out[3], :, 1),
        x1_centers = reshape(centers_out[1], :, 1),
        x2_centers = reshape(centers_out[2], :, 1),
        x3_centers = reshape(centers_out[3], :, 1),
        prim_rho   = rho_out,
        prim_u     = u_out,
        prim_uvec  = uvec_out,
        cons_fb    = fb_out,
        # Blocks/
        loc5      = zeros(Int32, 5, 1),   # loc.level-gid-lid-cnghost-gflag
        lx123     = zeros(Int64, 3, 1),   # loc.lx123
        xmin      = xmin,                 # Blocks/xmin
        # Info/ attributes
        BlocksPerPE   = Int32(1),
        NumMeshBlocks = Int32(1),
        RootLevel     = Int32(0),
        MaxLevel      = Int32(0),
        Multilevel    = Int32(0),
        RootGridSize  = Int32.(collect(N)),
        MeshBlockSize = Int32.(collect(Np)),
        # top-level datasets
        Levels           = zeros(Int64, 1),
        LogicalLocations = zeros(Int64, 3, 1),
    )
end


"""
    merge_meshblocks(rhdf_file)

Convenience wrapper: read everything straight from an open HDF5 file.
"""
function merge_meshblocks(rhdf_file)
    r(name) = HDF5.read(rhdf_file[name])
    if HDF5.attrs(rhdf_file["Info"])["IncludesGhost"] == 0
        NGhost = 0
    else
        NGhost = HDF5.attrs(rhdf_file["Info"])["NGhost"]
    end

    return merge_meshblocks(
        r("Locations/x"), r("Locations/y"), r("Locations/z"),
        r("VolumeLocations/x"), r("VolumeLocations/y"), r("VolumeLocations/z"),
        r("prims.rho"), r("prims.u"), r("prims.uvec"), r("cons.fB"),
        r("Blocks/loc.lx123"), NGhost,
    )
end


"""
    test_merge_meshblocks(; N=(16,8,8), nbl=(2,2,1), ng=4)

Self-test: build a random single-block field, split it into blocks the way
Parthenon would (ghosts copied from neighbours), merge it back, and check that
the result is bit-identical.
"""
function test_merge_meshblocks(; N = (16, 8, 8), nbl = (2, 2, 1), ng = 4)
    nb = N .÷ nbl
    Np = N .+ 2ng
    nblocks = prod(nbl)
    lx123 = zeros(Int64, 3, nblocks)
    for (b, I) in enumerate(CartesianIndices(nbl))   # x1 fastest, like Parthenon
        lx123[:, b] .= Tuple(I) .- 1
    end

    xf = ntuple(d -> collect(range(-ng, N[d] + ng, length = Np[d] + 1)) ./ N[d], 3)
    xc = ntuple(d -> 0.5 .* (xf[d][1:end-1] .+ xf[d][2:end]), 3)
    rho  = rand(Np...)
    u    = rand(Np...)
    uvec = rand(Np..., 3)
    fb   = zeros((Np .+ 1)..., 3)
    for c in 1:3
        sz = ntuple(d -> Np[d] + (d == c), 3)
        fb[(1:s for s in sz)..., c] .= rand(sz...)
    end

    # split
    blk(d, lx, extra) = (lx * nb[d]) .+ (1:(nb[d] + 2ng + extra))
    bxf = ntuple(d -> zeros(nb[d] + 2ng + 1, nblocks), 3)
    bxc = ntuple(d -> zeros(nb[d] + 2ng, nblocks), 3)
    brho = zeros((nb .+ 2ng)..., nblocks); bu = similar(brho)
    buvec = zeros((nb .+ 2ng)..., 3, nblocks)
    bfb = zeros((nb .+ 2ng .+ 1)..., 3, nblocks)
    for b in 1:nblocks
        lx = lx123[:, b]
        for d in 1:3
            bxf[d][:, b] .= xf[d][blk(d, lx[d], 1)]
            bxc[d][:, b] .= xc[d][blk(d, lx[d], 0)]
        end
        g = ntuple(d -> blk(d, lx[d], 0), 3)
        brho[:, :, :, b] .= rho[g...]
        bu[:, :, :, b]   .= u[g...]
        buvec[:, :, :, :, b] .= uvec[g..., :]
        for c in 1:3
            gf = ntuple(d -> blk(d, lx[d], d == c ? 1 : 0), 3)
            loc = ntuple(d -> 1:length(gf[d]), 3)
            bfb[loc..., c, b] .= fb[gf..., c]
        end
    end

    m = merge_meshblocks(bxf..., bxc..., brho, bu, buvec, bfb, lx123, ng)

    ok = m.x1_faces[:, 1] == xf[1] && m.x2_faces[:, 1] == xf[2] && m.x3_faces[:, 1] == xf[3] &&
         m.x1_centers[:, 1] == xc[1] && m.x2_centers[:, 1] == xc[2] && m.x3_centers[:, 1] == xc[3] &&
         m.prim_rho[:, :, :, 1] == rho && m.prim_u[:, :, :, 1] == u &&
         m.prim_uvec[:, :, :, :, 1] == uvec && m.cons_fb[:, :, :, :, 1] == fb
    println(ok ? "merge_meshblocks round-trip: PASS" : "merge_meshblocks round-trip: FAIL")
    return ok
end

# test_merge_meshblocks()



"""
    morton(l1, l2, l3, nbits)

Z-order (Morton) code of a logical location, x1 the fastest-varying bit.
Parthenon orders the blocks of a uniform mesh (their gid) in this order.
"""
function morton(l1, l2, l3, nbits)
    code = 0
    for bit in 0:max(nbits, 1)-1
        code |= ((l1 >> bit) & 1) << (3bit) |
                ((l2 >> bit) & 1) << (3bit + 1) |
                ((l3 >> bit) & 1) << (3bit + 2)
    end
    return code
end


"""
    split_meshblocks(x1_faces, x2_faces, x3_faces,
                     x1_centers, x2_centers, x3_centers,
                     cons_rho, cons_u, cons_uvec,
                     prim_rho, prim_u, prim_uvec,
                     cons_fb,
                     mb_nx, NGhost; x2_transmitting = true)

Inverse of `merge_meshblocks`: cut single-block (whole-mesh, ghost-inclusive)
arrays into Parthenon meshblocks of interior size `mb_nx = (nx1, nx2, nx3)`,
i.e. the `<parthenon/meshblock>` values.

Inputs may have a trailing block dimension of 1 or not:

    x1_faces   (Nx1 + 2ng + 1[, 1])          x1_centers (Nx1 + 2ng[, 1])   (same for x2, x3)
    prim_rho   (Nx1+2ng, Nx2+2ng, Nx3+2ng[, 1])           (same for prim_u, cons_rho, cons_u)
    prim_uvec  (Nx1+2ng, Nx2+2ng, Nx3+2ng, 3[, 1])        (same for cons_uvec)
    cons_fb    (Nx1+2ng+1, Nx2+2ng+1, Nx3+2ng+1, 3[, 1])

Each block's ghost zones are filled from the neighbouring blocks' interiors,
or from the global ghost zones on a physical boundary, as they would be after
a boundary sync.

Returns a NamedTuple with every array in the shape `HDF5.read` gives for a
multi-block restart (block index last), so each can be written straight to
the dataset of the same name, plus all the Blocks/, Info/ and top-level values
that depend on the block layout.
"""
function split_meshblocks(
    x1_faces, x2_faces, x3_faces,
    x1_centers, x2_centers, x3_centers,
    cons_rho, cons_u, cons_uvec,
    prim_rho, prim_u, prim_uvec,
    cons_fb,
    mb_nx, NGhost;
    x2_transmitting = true,
)
    ng = Int(NGhost)
    nb = ntuple(d -> Int(mb_nx[d]), 3)

    # Accept arrays with or without the trailing block dimension of 1
    function squeeze(A, nd, name)
        ndims(A) == nd && return A
        if ndims(A) == nd + 1 && size(A, nd + 1) == 1
            return dropdims(A; dims = nd + 1)
        end
        throw(ArgumentError("$name: expected $nd dims (optionally + a trailing 1), got size $(size(A))"))
    end

    xf = (squeeze(x1_faces, 1, "x1_faces"), squeeze(x2_faces, 1, "x2_faces"), squeeze(x3_faces, 1, "x3_faces"))
    xc = (squeeze(x1_centers, 1, "x1_centers"), squeeze(x2_centers, 1, "x2_centers"), squeeze(x3_centers, 1, "x3_centers"))
    scal = (
        cons_rho = squeeze(cons_rho, 3, "cons_rho"),
        cons_u   = squeeze(cons_u,   3, "cons_u"),
        prim_rho = squeeze(prim_rho, 3, "prim_rho"),
        prim_u   = squeeze(prim_u,   3, "prim_u"),
    )
    vecs = (
        cons_uvec = squeeze(cons_uvec, 4, "cons_uvec"),
        prim_uvec = squeeze(prim_uvec, 4, "prim_uvec"),
    )
    fb = squeeze(cons_fb, 4, "cons_fb")

    # ---------------- Sizes and checks ----------------
    Np = size(scal.prim_rho)                 # whole mesh incl. ghosts
    N  = Np .- 2ng                           # whole mesh interior (RootGridSize)
    Nb = nb .+ 2ng                           # one block incl. ghosts (MeshBlockSize)

    for A in scal
        size(A) == Np || throw(DimensionMismatch("cell array size $(size(A)) != $Np"))
    end
    for A in vecs
        size(A) == (Np..., 3) || throw(DimensionMismatch("vector array size $(size(A)) != $((Np..., 3))"))
    end
    size(fb) == ((Np .+ 1)..., 3) || throw(DimensionMismatch("cons_fb size $(size(fb)) != $(((Np .+ 1)..., 3))"))
    for d in 1:3
        length(xf[d]) == Np[d] + 1 || throw(DimensionMismatch("x$(d)_faces has length $(length(xf[d])), expected $(Np[d] + 1)"))
        length(xc[d]) == Np[d]     || throw(DimensionMismatch("x$(d)_centers has length $(length(xc[d])), expected $(Np[d])"))
        N[d] % nb[d] == 0 ||
            throw(ArgumentError("mesh nx$d = $(N[d]) is not divisible by meshblock nx$d = $(nb[d])"))
        nb[d] >= ng ||
            throw(ArgumentError("meshblock nx$d = $(nb[d]) is smaller than NGhost = $ng"))
    end
    if x2_transmitting && nb[3] != N[3]
        @warn "Transmitting x2 boundaries need the whole x3 range in one meshblock" meshblock_nx3 = nb[3] mesh_nx3 = N[3]
    end

    nbl = N .÷ nb                            # blocks along each direction
    nblocks = prod(nbl)
    root_level = 0
    while maximum(nbl) > (1 << root_level)
        root_level += 1
    end

    # Block order: Morton / Z-order over the logical locations (Parthenon's gid order)
    locs = vec([(i, j, k) for i in 0:nbl[1]-1, j in 0:nbl[2]-1, k in 0:nbl[3]-1])
    sort!(locs; by = l -> morton(l..., root_level))

    # Global (ghost-inclusive, 1-based) indices covered by the block at lx along d
    grange(d, lx, extra) = (lx * nb[d]) .+ (1:(Nb[d] + extra))

    # ---------------- Allocate outputs ----------------
    bxf = ntuple(d -> zeros(eltype(xf[d]), Nb[d] + 1, nblocks), 3)
    bxc = ntuple(d -> zeros(eltype(xc[d]), Nb[d], nblocks), 3)
    bscal = map(A -> zeros(eltype(A), Nb..., nblocks), scal)
    bvecs = map(A -> zeros(eltype(A), Nb..., 3, nblocks), vecs)
    bfb   = zeros(eltype(fb), (Nb .+ 1)..., 3, nblocks)   # unused slots stay 0

    loc5  = zeros(Int32, 5, nblocks)          # loc.level-gid-lid-cnghost-gflag
    lx123 = zeros(Int64, 3, nblocks)          # loc.lx123
    xmin  = zeros(Float64, 3, nblocks)        # Blocks/xmin (interior lower corner)
    logical_locations = zeros(Int64, 3, nblocks)
    levels = zeros(Int64, nblocks)            # level relative to root level

    # ---------------- Fill blocks ----------------
    for (b, lx) in enumerate(locs)
        gid = b - 1
        r = ntuple(d -> grange(d, lx[d], 0), 3)

        for d in 1:3
            bxf[d][:, b] .= xf[d][grange(d, lx[d], 1)]
            bxc[d][:, b] .= xc[d][r[d]]
            xmin[d, b] = xf[d][lx[d] * nb[d] + ng + 1]
        end

        for (B, A) in zip(bscal, scal)
            B[:, :, :, b] .= A[r...]
        end
        for (B, A) in zip(bvecs, vecs)
            B[:, :, :, :, b] .= A[r..., :]
        end

        # Face component c is staggered (one extra point) along axis c only
        for c in 1:3
            rf = ntuple(d -> grange(d, lx[d], d == c), 3)
            bfb[(1:length(x) for x in rf)..., c, b] .= fb[rf..., c]
        end

        loc5[:, b]  .= Int32[root_level, gid, gid, 0, 0]
        lx123[:, b] .= lx
        logical_locations[:, b] .= lx
    end

    return (
        # datasets (block index last, as HDF5.read returns them)
        x1_faces   = bxf[1], x2_faces   = bxf[2], x3_faces   = bxf[3],   # Locations/x,y,z
        x1_centers = bxc[1], x2_centers = bxc[2], x3_centers = bxc[3],   # VolumeLocations/x,y,z
        cons_rho  = bscal.cons_rho,
        cons_u    = bscal.cons_u,
        cons_uvec = bvecs.cons_uvec,
        prim_rho  = bscal.prim_rho,
        prim_u    = bscal.prim_u,
        prim_uvec = bvecs.prim_uvec,
        cons_fb   = bfb,
        # Blocks/
        loc5  = loc5,
        lx123 = lx123,
        xmin  = xmin,
        # Info/ attributes
        BlocksPerPE   = Int32(nblocks),
        NumMeshBlocks = Int32(nblocks),
        RootLevel     = Int32(root_level),
        MaxLevel      = Int32(0),
        Multilevel    = Int32(0),
        RootGridSize  = Int32.(collect(N)),
        MeshBlockSize = Int32.(collect(Nb)),
        # top-level datasets
        Levels           = levels,
        LogicalLocations = logical_locations,
        # for updating <parthenon/meshblock> in Input/File
        meshblock_nx = nb,
        blocks_per_dim = nbl,
    )
end


"""
    test_split_meshblocks()

1. Checks that a 192x96x96 mesh split into 96x48x96 blocks reproduces the
   loc.level-gid-lid-cnghost-gflag and loc.lx123 values from a real KHARMA restart.
2. Round-trips random data through split_meshblocks -> merge_meshblocks for
   several layouts (needs mergeMeshblocks.jl loaded).
"""
function test_split_meshblocks()
    ok = true

    function fake(N, ng)
        Np = N .+ 2ng
        xf = ntuple(d -> collect(range(-ng, N[d] + ng, length = Np[d] + 1)) ./ N[d], 3)
        xc = ntuple(d -> 0.5 .* (xf[d][1:end-1] .+ xf[d][2:end]), 3)
        fb = zeros((Np .+ 1)..., 3)
        for c in 1:3
            sz = ntuple(d -> Np[d] + (d == c), 3)
            fb[(1:s for s in sz)..., c] .= rand(sz...)
        end
        return xf, xc, rand(Np...), rand(Np...), rand(Np..., 3), rand(Np...), rand(Np...), rand(Np..., 3), fb
    end

    # 1. Metadata against the real 4-block KHARMA restart
    xf, xc, cr, cu, cv, pr, pu, pv, fb = fake((192, 96, 96), 4)
    s = split_meshblocks(xf..., xc..., cr, cu, cv, pr, pu, pv, fb, (96, 48, 96), 4)
    meta_ok = s.loc5 == Int32[1 1 1 1; 0 1 2 3; 0 1 2 3; 0 0 0 0; 0 0 0 0] &&
              s.lx123 == [0 1 0 1; 0 0 1 1; 0 0 0 0] &&
              s.RootLevel == 1 && s.NumMeshBlocks == 4 &&
              size(s.x1_faces) == (105, 4) && size(s.x2_faces) == (57, 4) && size(s.x3_faces) == (105, 4) &&
              size(s.prim_rho) == (104, 56, 104, 4) && size(s.prim_uvec) == (104, 56, 104, 3, 4) &&
              size(s.cons_fb) == (105, 57, 105, 3, 4)
    println("metadata vs. KHARMA 2x2x1 restart: ", meta_ok ? "PASS" : "FAIL")
    ok &= meta_ok

    # 2. Round trip split -> merge
    for (N, nb, ng) in (((192, 96, 96), (96, 48, 96), 4),
                        ((24, 16, 8), (8, 4, 8), 2),
                        ((16, 16, 16), (4, 8, 16), 2))
        xf, xc, cr, cu, cv, pr, pu, pv, fb = fake(N, ng)
        s = split_meshblocks(xf..., xc..., cr, cu, cv, pr, pu, pv, fb, nb, ng; x2_transmitting = false)
        m = merge_meshblocks(s.x1_faces, s.x2_faces, s.x3_faces,
                             s.x1_centers, s.x2_centers, s.x3_centers,
                             s.prim_rho, s.prim_u, s.prim_uvec, s.cons_fb,
                             s.lx123, ng)
        rt = m.x1_faces[:, 1] == xf[1] && m.x2_faces[:, 1] == xf[2] && m.x3_faces[:, 1] == xf[3] &&
             m.x1_centers[:, 1] == xc[1] && m.prim_rho[:, :, :, 1] == pr &&
             m.prim_u[:, :, :, 1] == pu && m.prim_uvec[:, :, :, :, 1] == pv &&
             m.cons_fb[:, :, :, :, 1] == fb
        println("round trip N=$N, meshblock=$nb, ng=$ng: ", rt ? "PASS" : "FAIL")
        ok &= rt
    end
    return ok
end

end