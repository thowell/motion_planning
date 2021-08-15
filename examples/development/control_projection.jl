contact_control_path = "/home/taylor/Research/ContactControl.jl/src"

using Parameters

# Utilities
include(joinpath(contact_control_path, "utils.jl"))

# Solver
include(joinpath(contact_control_path, "solver/cones.jl"))
include(joinpath(contact_control_path, "solver/interior_point.jl"))
include(joinpath(contact_control_path, "solver/lu.jl"))

n = 3
nz = 3 * 7
nθ = 3 * 3

function residual(z, θ, κ)
    u = z[1:n]
    s1 = z[n .+ (1:n)]
    s2 = z[n + n .+ (1:n)]
    y1 = z[n + n + n .+ (1:n)]
    y2 = z[n + n + n + n .+ (1:n)]
    z1 = z[n + n + n + n + n .+ (1:n)]
    z2 = z[n + n + n + n + n + n .+ (1:n)]

    ū = θ[1:n]
    ul = θ[n .+ (1:n)]
    uu = θ[n + n .+ (1:n)]

    [
     (u - ū) - y1 + y2;
     -y1 - z1;
     -y2 - z2;
     (uu - u) - s1;
     (u - ul) - s2;
     s1 .* z1 .- κ;
     s2 .* z2 .- κ;
    ]
end

@variables z[1:nz], θ[1:nθ], κ[1:1]

r = residual(z, θ, κ)
r .= simplify(r)
r_func = eval(Symbolics.build_function(r, z, θ, κ)[2])

rz = Symbolics.jacobian(r, z)
rz = simplify.(rz)
rz_func = eval(Symbolics.build_function(rz, z, θ)[2])

rθ = Symbolics.jacobian(r, θ)
rθ = simplify.(rθ)
rθ_func = eval(Symbolics.build_function(rθ, z, θ)[2])

idx_ineq = collect([(n .+ (1:(n + n)))..., (n + n + n + n + n .+ (1:(n + n)))...])

opts = InteriorPointOptions(
  κ_init = 0.1,
  κ_tol = 1.0e-5,
  r_tol = 1.0e-8,
  diff_sol = true)

ū = [0.0, 0.0, 0.0]
ul = [-1.0, -1.0, -1.0]
uu = [1.0, 1.0, 1.0]

z0 = 0.1 * ones(nz)
z0[1:n] = copy(ū)
θ0 = [ū; ul; uu]

# solver
ip = interior_point(z0, θ0,
  r! = r_func, rz! = rz_func,
  rz = similar(rz, Float64),
  rθ! = rθ_func,
  rθ = similar(rθ, Float64),
  idx_ineq = idx_ineq,
  opts = opts)

interior_point_solve!(ip)

ip.z[1:n]
