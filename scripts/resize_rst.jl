path_to_JULIARMA = "/home/june/Desktop/Code/Repos/kharmaResize/"

import ArgParse
include(path_to_JULIARMA * "src/kharmaResize.jl")
import .kharmaResize as kResize

import JLD2


function parse_commandline()
    # Initialize settings using the explicit module prefix
    s = ArgParse.ArgParseSettings()

    # Define your arguments using macro syntax with explicit prefixes
    ArgParse.@add_arg_table! s begin
        "infile"
            help = "path to rhdf file"
            arg_type = String
            required = true

        "outfile"
            help = "desired path to rhdf file"
            arg_type = String
            required = true

        "--method", "-m"
            help = "choice for interpolation type (1:linear or 2:nearest neighbor)"
            arg_type = Int
            default = 1
            range_tester = (x -> x == 1 || x == 2)

        "--bc1inner", "-a"
            help = "Inner x1 boundary condition (1=Dirichlet, 2=Neumann, 3=periodic)"
            arg_type = Int
            default = 1
            range_tester = x -> 1 <= x <= 3

        "--bc1outer", "-b"
            help = "Outer x1 boundary condition (1=Dirichlet, 2=Neumann, 3=periodic)"
            arg_type = Int
            default = nothing
            range_tester = x -> 1 <= x <= 3

        "--bc2inner", "-c"
            help = "Inner x2 boundary condition (1=Dirichlet, 2=Neumann, 3=periodic, 4=transmitting)"
            arg_type = Int
            default = 4
            range_tester = x -> 1 <= x <= 4

        "--bc2outer", "-d"
            help = "Outer x2 boundary condition (1=Dirichlet, 2=Neumann, 3=periodic, 4=transmitting)"
            arg_type = Int
            default = nothing
            range_tester = x -> 1 <= x <= 4

        "--bc3inner", "-e"
            help = "Inner x3 boundary condition (1=Dirichlet, 2=Neumann, 3=periodic)"
            arg_type = Int
            default = 3
            range_tester = x -> 1 <= x <= 3

        "--bc3outer", "-f"
            help = "Outer x3 boundary condition (1=Dirichlet, 2=Neumann, 3=periodic)"
            arg_type = Int
            default = nothing
            range_tester = x -> 1 <= x <= 3

        "--nx1", "-x"
            help = "Resized Nx1"
            arg_type = Int
            default = 256

        "--nx2", "-y"
            help = "Resized Nx1"
            arg_type = Int
            default = 128

        "--nx3", "-z"
            help = "Resized Nx1"
            arg_type = Int
            default = 128

        "--meshnx1", "-j"
            help = "Resized meshblocks/Nx1"
            arg_type = Int
            default = 256

        "--meshnx2", "-k"
            help = "Resized meshblocks/Nx2"
            arg_type = Int
            default = 128

        "--meshnx3", "-l"
            help = "Resized meshblocks/Nx3"
            arg_type = Int
            default = 128


    end

    # Parse using the explicit module function call
    return ArgParse.parse_args(s)
end


# Parse arguments into a dictionary
parsed_args = parse_commandline()

parsed_args["bc1outer"] = something(parsed_args["bc1outer"], parsed_args["bc1inner"])
parsed_args["bc2outer"] = something(parsed_args["bc2outer"], parsed_args["bc2inner"])
parsed_args["bc3outer"] = something(parsed_args["bc3outer"], parsed_args["bc3inner"])
# Access data exactly the same way
# println("File: ", parsed_args["filename"])
# println("Key:  ", parsed_args["key"])
# println("Port: ", parsed_args["port"])
# export options

input_str = "Running JuliarmaResizer with the following params: \nFiles: "
input_str *= "   input: " * string(parsed_args["infile"])
input_str *= "   output: " * string(parsed_args["outfile"])

println("Running JuliarmaResizer with the following params: ")
println("---------------------------------------------------")
println("Files: ")
println("   input: ",parsed_args["infile"])
println("   output: ",parsed_args["outfile"])
println("New Resolution: ")

# BC = [(,"dirichlet"),("transmitting","transmitting"),("periodic","periodic")]
BC_input = Dict(
    1 => "dirichlet",
    2 => "neumann",
    3 => "periodic",
    4 => "transmitting"
    )

for key in ["nx1","nx2","nx3"]
    println("   ",key,": ",parsed_args[key])
end

for key in ["meshnx1","meshnx2","meshnx3"]
    println("   ",key,": ",parsed_args[key])
end


println("Boundary Conditions (inner,outer):")
for key in ["1","2","3"]
    println("   nx$(key): (",BC_input[parsed_args["bc" * key * "inner"]], ", ",BC_input[parsed_args["bc" * key * "outer"]],")")
end

method_input = Dict(
    1 => "linear",
    2 => "NN"
)
println("Method: ", method_input[parsed_args["method"]])
# --------------------------------------------
# --------------------------------------------



test_rhdf = "/home/june/Desktop/Code/Repos/kharmaResize/testDataGhostCells/torus.out1.00048.rhdf"
test_output = "/home/june/Desktop/Code/Repos/kharmaResize/testDataGhostCells/kharma_resampled.rhdf"

BC = [
    (BC_input[parsed_args["bc1inner"]],BC_input[parsed_args["bc1outer"]]),
    (BC_input[parsed_args["bc2inner"]],BC_input[parsed_args["bc2outer"]]),
    (BC_input[parsed_args["bc3inner"]],BC_input[parsed_args["bc3outer"]]),
    ]
cfg = kResize.Config(
    parsed_args["infile"],
    parsed_args["outfile"], 
    method_input[parsed_args["method"]],# place NN here for nearest neighbor compare
    BC, 
    parsed_args["nx1"], parsed_args["nx2"], parsed_args["nx3"],
    parsed_args["meshnx1"], parsed_args["meshnx2"], parsed_args["meshnx3"])

println("---------------------------------------------------")
println("Beginning Resizing...")
println()
kResize.main(cfg)