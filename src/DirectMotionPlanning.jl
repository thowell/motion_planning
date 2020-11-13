module DirectMotionPlanning

using LinearAlgebra, ForwardDiff, FiniteDiff, StaticArrays, SparseArrays
using MathOptInterface, Ipopt
using Distributions, Interpolations
using JLD2

using Colors
using CoordinateTransformations
using FileIO
using GeometryTypes
using MeshCat, MeshIO, MeshCatMechanisms
using Rotations
using RigidBodyDynamics

include("lqr.jl")
include("unscented.jl")
include("indices.jl")
include("utils.jl")

include("integration.jl")
include("model.jl")

include("problem.jl")

include("objective.jl")
include("objectives/quadratic.jl")
include("objectives/penalty.jl")

include("constraints.jl")
include("constraints/dynamics.jl")

include("moi.jl")
include("solvers/snopt.jl")

end # module
