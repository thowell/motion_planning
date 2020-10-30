module DirectMotionPlanning

using LinearAlgebra, ForwardDiff, FiniteDiff, StaticArrays, SparseArrays
using MathOptInterface, Ipopt
using JLD2

include("lqr.jl")
include("unscented.jl")
include("indices.jl")
include("utils.jl")

include("integration.jl")
include("model.jl")

include("problem.jl")

include("objective.jl")
include("objective/quadratic.jl")
include("objective/penalty.jl")

include("constraints.jl")
include("constraints/dynamics.jl")

include("moi.jl")
include("solvers/snopt.jl")

end # module
