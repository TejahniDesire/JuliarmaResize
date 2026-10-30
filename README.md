# kharmaResize

Resize a [KHARMA](https://github.com/parthenon-hpc-lab/kharma) GRMHD restart file (`.rhdf`) to a new grid resolution and/or meshblock layout, so that a simulation can be continued at a different resolution.

The resizer interpolates the primitive variables and the face-centered magnetic field onto the new grid. It then removes the divergence of B that the interpolation introduces, recomputes the conserved variables with the new metric, and writes a restart file that KHARMA can read directly.

```
torus.out1.00030.rhdf  (192×96×96, 1 meshblock)
        │
        ▼   kharmaResize
256x128x128_resized.rhdf  (256×128×128, 2×2×1 meshblocks of 128×64×128)
```

> **Status:** research code under active development. Parts of this code were written with the help of an AI assistant (marked in the source). Results have been checked against KHARMA in the tests described under [Validation](#validation), but not every routine has been verified line by line. Check your own output before relying on it.

---

## Features

- **Arbitrary resolution changes.** Upscaling or downscaling within the same domain, using linear or nearest-neighbor interpolation.
- **Divergence cleaning.** A projection (Poisson solve with conjugate gradients) makes the resized face-centered B field divergence-free, using the same discrete divergence stencil as KHARMA's face-centered constrained transport.
- **Consistent conserved variables.** `cons.rho`, `cons.u` and `cons.uvec` are recomputed from the interpolated primitives with the metric of the *new* grid, following KHARMA's GRMHD conventions.
- **Exact metric.** The metric on the new grid is evaluated analytically by [pyharm](https://github.com/AFD-Illinois/pyharm), not interpolated.
- **Meshblocks.** Reads restarts with any number of meshblocks and can write the output with a different meshblock layout. The mesh must be uniform; AMR/SMR is not supported.
- **Ghost zones.** Ghost zones are filled with the physical boundary conditions stored in the restart (`outflow`, `periodic`, `transmitting` polar).
- **Restart metadata.** `Blocks/`, `Info/`, `Locations`, the embedded parameter file (`Input/File`) and the timestep are updated for the new grid.

---

## Repository layout

```
kharmaResize/
├── Project.toml
├── src/                      # Julia module
│   ├── kharmaResize.jl       # main pipeline: read, resize, clean, write
│   ├── kharmaFunctions.jl    # primitive→conserved, divergence, Poisson solve, par-file editing
│   ├── helperFunctions.jl    # interpolation, ghost-zone padding, face→center B, dt scaling
│   └── meshblocks.jl         # merge / split Parthenon meshblocks
└── scripts/
    ├── python_wrapper.py     # entry point: set parameters here and run
    ├── pyharmConverter.py    # builds the new-grid metric with pyharm
    └── resize_rst.jl         # Julia command-line front end
```

---

## Requirements

**Julia** (1.10 or newer) with:

- `HDF5`
- `Interpolations`
- `IterativeSolvers`
- `LinearMaps`
- `ArgParse`

The standard libraries `LinearAlgebra`, `SparseArrays` and `DelimitedFiles` are also used.

```julia
julia> ]
pkg> activate .
pkg> instantiate
pkg> add HDF5 Interpolations IterativeSolvers LinearMaps ArgParse
```

**Python** (3.9 or newer) with:

- `numpy`
- `h5py`
- [`pyharm`](https://github.com/AFD-Illinois/pyharm)

**Input files** from the same KHARMA run:

- a restart file (`*.rhdf`) to resize;
- any regular dump (`*.phdf`) from that run, used only to read the coordinate-system parameters and the adiabatic index.

---

## Usage

1. Open `scripts/python_wrapper.py` and set the parameters:

   ```python
   path_to_JULIARMA = "/path/to/kharmaResize/"

   input_rhdf_file  = ".../torus.out1.00030.rhdf"   # restart to resize
   input_phdf_file  = ".../torus.out0.00030.phdf"   # any dump from the same run
   output_rhdf_file = ".../256x128x128_resized.rhdf"

   # new mesh resolution
   Nx1, Nx2, Nx3 = 256, 128, 128

   # new meshblock size; must divide the mesh evenly
   Mesh_Nx1, Mesh_Nx2, Mesh_Nx3 = 128, 64, 128

   # boundary conditions for the divergence-cleaning Poisson solve
   # 1=Dirichlet, 2=Neumann, 3=periodic, 4=transmitting
   boundary_conditions = [(1, 1), (4, 4), (3, 3)]

   method  = 1    # 1 = linear, 2 = nearest neighbor
   nthread = 10   # Julia threads
   ```

2. Run it:

   ```bash
   cd scripts
   python python_wrapper.py
   ```

   The wrapper first writes a temporary metric file (`<output>_metric.h5`), then runs the Julia resizer, then deletes the metric file.

3. Restart KHARMA from the new file in the usual way. The resolution and meshblock sizes in the parameter file embedded in the restart are already updated.

### Running the Julia step directly

If you already have a metric file, you can call the Julia step yourself:

```bash
julia -t 10 scripts/resize_rst.jl input.rhdf output.rhdf \
    --metricfile output_metric.h5 \
    --nx1 256 --nx2 128 --nx3 128 \
    --meshnx1 128 --meshnx2 64 --meshnx3 128 \
    --method 1 \
    --bc1inner 1 --bc1outer 1 --bc2inner 4 --bc2outer 4 --bc3inner 3 --bc3outer 3
```

| Option | Meaning | Default |
|---|---|---|
| `infile`, `outfile` | input and output restart files; they must be different | required |
| `--metricfile` | HDF5 file from `pyharmConverter.resize_rhdf` | required |
| `--nx1/2/3` | new mesh resolution | 256 / 128 / 128 |
| `--meshnx1/2/3` | new meshblock size | equal to the mesh (one block) |
| `--method` | 1 = linear, 2 = nearest neighbor | 1 |
| `--bc{1,2,3}{inner,outer}` | Poisson-solve boundary conditions (see below) | x1 Dirichlet, x2 transmitting, x3 periodic |

---

## How it works

1. **Metric** (`pyharmConverter.py`). The global mesh size and domain are read from the restart's `Info/RootGridSize` and `Info/RootGridDomain`, so the block layout doesn't matter. pyharm builds a grid with the new resolution over the same domain, using the coordinate parameters stored in the `.phdf` (spin, `hslope`, FMKS settings). On that grid it evaluates g_μν, g^μν and √−g at cell centers and at faces. These come from the closed-form Kerr–Schild metric and the coordinate Jacobian, so they are exact up to floating-point precision. The arrays are written to a compressed temporary HDF5 file; because the metric doesn't depend on x3, the file is only a few MB.

2. **Read and merge** (`get_rhdf_params`, `merge_meshblocks`). The restart is read. If it has several meshblocks, they are stitched into one whole-mesh array that includes ghost zones. Ghost zones on internal block faces are dropped, and faces shared by neighboring blocks are checked for agreement.

3. **Interpolate** (`resize_grid`).
   - `prims.rho`, `prims.u` and `prims.uvec` are interpolated in native coordinates from the old cell centers to the new ones.
   - Each component of `cons.fB` (√−g·Bⁱ) is interpolated on its own staggered grid: x1 faces for B¹, and so on.
   - The old ghost zones are included in the interpolation, so new points near the domain edges stay within the data.

4. **Clean the divergence** (`div_cleanup`, `solve_phi`). The resized B field is replaced by B′ = B − ∇φ, where φ solves the discrete Poisson problem ∇·∇φ = ∇·B. The solver is matrix-free conjugate gradients. The divergence and gradient use KHARMA's face-centered stencil, so B′ is divergence-free in KHARMA's discrete sense, to the solver tolerance. B² on the polar faces is kept at zero.

5. **Recompute conserved variables** (`calculate_cons_vars`, `p_to_u_arrays`). Face B is converted to cell-centered primitive B. Then ρu⁰√−g, √−g(T⁰₀ + ρu⁰) and √−g T⁰ᵢ are computed on the new grid with the new metric, following KHARMA's conventions for u^μ, b^μ and the ideal-gas stress-energy tensor.

6. **Fill ghost zones** (`physical_bc`, `pad_array`, `pad_face_array`). Ghost zones are filled with the physical boundary conditions read from the restart's `Params/Boundaries/*`:

   | KHARMA boundary | Padding used |
   |---|---|
   | `outflow` | Neumann (copy) |
   | `periodic` | periodic |
   | `transmitting` | polar transmitting: reflect across the pole, shift by π in x3, flip the sign of the x2 and x3 vector components |

7. **Split and write** (`split_meshblocks`, `write_rhdf`). The arrays are cut into the requested meshblocks, ordered the way Parthenon numbers them (Z-order). The new restart is then written with:
   - updated `Blocks/`, `Info/`, `Locations`, `VolumeLocations`, `Levels` and `LogicalLocations`;
   - the embedded parameter file, with `<parthenon/mesh>` and `<parthenon/meshblock>` updated;
   - a rescaled timestep: `Info/dt` and `Params/Globals/dt_last` are multiplied by `0.9 · min(1, Nx_old/Nx_new)` so the first step after a refinement stays within the CFL limit.

### Two kinds of boundary conditions

- **The `--bc*` options** apply only to the Poisson solve for divergence cleaning. The defaults are Dirichlet φ = 0 at the radial boundaries, a zero-flux (transmitting) condition at the poles, and periodic in x3.
- **The ghost zones written to the restart** always use the physical boundaries recorded in the restart file, regardless of the `--bc*` options.

---

## Limitations

- **Uniform meshes only.** No AMR/SMR (`Multilevel` must be 0), and the domain is kept the same; only the resolution and block layout change.
- **3D only.** 2D runs (`nx3 = 1`) are not supported.
- **Meshblock constraints:**
  - meshblock sizes must divide the mesh evenly;
  - with transmitting x2 boundaries, the meshblock must span the whole x3 direction;
  - Nx3 must be even.
- **Metric parameters must match the run.** The coordinate parameters are read from the `.phdf`, so the `.phdf` and the restart must come from the same run. The script checks that their mesh sizes match. To be safe, they should also be from the same timestep.

---

## Acknowledgements

Built for use with [KHARMA](https://github.com/parthenon-hpc-lab/kharma) (based on [Parthenon](https://github.com/parthenon-hpc-lab/parthenon)), with [pyharm](https://github.com/AFD-Illinois/pyharm) used to evaluate the metric.
