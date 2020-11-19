# Model
include_model("hopper")

# Dimensions
nq = 4 # configuration dimension
nu = 2 # control dimension
nc = 1 # number of contact points
nf = 2 # number of faces for friction cone
nb = nc * nf
ns = nq

# Parameters
g = 9.81 # gravity
μ = 1.0  # coefficient of friction
mb = 10.0 # body mass
ml = 1.0  # leg mass
Jb = 2.5 # body inertia
Jl = 0.25 # leg inertia

n = 2 * nq
m = nu + nc + nb + nc + nb + ns
d = 0

idx_u = (1:nu)
idx_λ = nu .+ (1:nc)
idx_b = nu + nc .+ (1:nb)
idx_ψ = nu + nc + nb .+ (1:nc)
idx_η = nu + nc + nb + nc .+ (1:nb)
idx_s = nu + nc + nb + nc + nb .+ (1:ns)

model = Hopper(n, m, d,
			   mb, ml, Jb, Jl,
			   μ, g,
			   qL, qU,
			   nq,
		       nu,
		       nc,
		       nf,
		       nb,
		   	   ns,
		       idx_u,
		       idx_λ,
		       idx_b,
		       idx_ψ,
		       idx_η,
		       idx_s)

# Free-time model
model_ft = free_time_model(model)

function fd(model::Hopper, x⁺, x, u, w, h, t)
	q3 = view(x⁺, model.nq .+ (1:model.nq))
	q2⁺ = view(x⁺, 1:model.nq)
	q2⁻ = view(x, model.nq .+ (1:model.nq))
	q1 = view(x, 1:model.nq)
	u_ctrl = view(u, model.idx_u)
	λ = view(u, model.idx_λ)
	b = view(u, model.idx_b)
	s = view(u, model.idx_s)
	h = u[end]

	[q2⁺ - q2⁻;
	((1.0 / h) * (M_func(model, q1) * (SVector{4}(q2⁺) - SVector{4}(q1))
	- M_func(model, q2⁺) * (SVector{4}(q3) - SVector{4}(q2⁺)))
	+ transpose(B_func(model, q3)) * SVector{2}(u_ctrl)
	+ transpose(N_func(model, q3)) * SVector{1}(λ)
	+ transpose(P_func(model, q3)) * SVector{2}(b)
	- h * G_func(model, q2⁺)) + s]
end

function maximum_dissipation(model::Hopper, x⁺, u, h)
	q3 = x⁺[model.nq .+ (1:model.nq)]
	q2 = x⁺[1:model.nq]
	ψ = u[model.idx_ψ]
	ψ_stack = ψ[1] * ones(model.nb)
	η = u[model.idx_η]
	h = u[end]
	return P_func(model, q3) * (q3 - q2) / h + ψ_stack - η
end

# Horizon
T = 21

# Time step
tf = 1.0
h = tf / (T - 1)

# Bounds
_uu = Inf * ones(model_ft.m)
_uu[model_ft.idx_u] .= 25.0
_uu[model_ft.idx_s] .= 0.0
_uu[end] = 1.0 * h
_ul = zeros(model_ft.m)
_ul[model_ft.idx_u] .= -25.0
_ul[model_ft.idx_s] .= 0.0
_ul[end] = 0.5 * h
ul, uu = control_bounds(model_ft, T, _ul, _uu)

# Initial and final states
q1 = [0.0, 0.5 , 0.0, 0.5]
q_right = [-0.25, 0.5 + 0.5, pi / 2.0, 0.25]
q_top = [-0.5, 0.5 + 1.0, pi, 0.25]
q_left = [-0.75, 0.5 + 0.5, 3.0 * pi / 2.0, 0.25]
qT = [-1.0, 0.5,  2.0 * pi, 0.5]

xl, xu = state_bounds(model_ft, T,
		[model_ft.qL; model_ft.qL],
		[model_ft.qU; model_ft.qU],
        x1 = [q1; q1],
		xT = [Inf * ones(model_ft.nq); qT])

q_ref = [linear_interp(q1, q_right, 6)...,
         linear_interp(q_right, q_top, 6)[2:end]...,
         linear_interp(q_top, q_left, 6)[2:end]...,
         linear_interp(q_left, qT, 6)[2:end]...]

x_ref = configuration_to_state(q_ref)

# Objective
include_objective(["velocity"])
obj_tracking = quadratic_time_tracking_objective(
    [Diagonal(100.0 * [1.0; 1.0; 0.0; 0.0; 1.0; 1.0; 0.0; 0.0]) for t = 1:T],
    [Diagonal([1.0e-1, 1.0e-1, zeros(model_ft.m - model_ft.nu)...]) for t = 1:T-1],
    [x_ref[t] for t = 1:T],
    [zeros(model_ft.m) for t = 1:T-1],
    1.0)

obj_velocity = velocity_objective([Diagonal(10.0 * ones(model_ft.nq)) for t = 1:T],
	model_ft.nq)
obj = MultiObjective([obj_tracking, obj_velocity])

# Constraints
include_constraints(["contact_al", "free_time"])
con_contact = contact_al_constraints(model_ft, T)
con_free_time = free_time_constraints(T)
con = multiple_constraints([con_free_time, con_contact])

# Problem
prob = trajectory_optimization_problem(model_ft,
               obj,
               T,
               xl = xl,
               xu = xu,
               ul = ul,
               uu = uu,
               con = con
               )

# Trajectory initialization
x0 = deepcopy(x_ref) # linear interpolation on state
u0 = [[1.0e-5 * rand(model_ft.m - 1); h] for t = 1:T-1] # random controls

# Pack trajectories into vector
z0 = pack(x0, u0, prob)

using LinearAlgebra, ForwardDiff, SparseArrays, Optim, LineSearches
include(joinpath(pwd(),"src/solvers/augmented_lagrangian.jl"))
prob = prob
function f(x)
    MOI.eval_objective(prob, x)
end

function g!(G, x)
    MOI.eval_objective_gradient(prob, G, x)
    nothing
end


function c!(c, x)
    MOI.eval_constraint(prob, c, x)
    c .*= -1.0
    nothing
end

spar = sparsity_jacobian(prob)
global jac = zeros(length(spar))
global ii = [s[1] for s in spar]
global jj = [s[2] for s in spar]
function d!(D, x)
    MOI.eval_constraint_jacobian(prob, jac, x)
    D .= sparse(ii, jj, -1.0 .* jac)
end

function f_al(x, al::AugmentedLagrangian)
    # evaluate constraints
    c!(al.c, x)
    bounds!(al, x)
    active_set_update!(al)

    # compute objective
    J = f(x)

    # add augmented Lagrangian terms
    J += al.λ' * al.c + 0.5 * sum(al.as .* al.ρ .* (al.c.^2.0))
    J += al.λl' * al.cl + 0.5 * sum(al.asl .* al.ρl .* (al.cl.^2.0))
    J += al.λu' * al.cu + 0.5 * sum(al.asu .* al.ρu .* (al.cu.^2.0))
end

function g_al!(G, x, al::AugmentedLagrangian)
    # compute objective gradient
    g!(G, x)

    # evaluate constraints
    # ForwardDiff.jacobian!(al.∇c, c!, al.c, x)
    c!(al.c, x)
    d!(al.∇c, x)

    bounds!(al, x)
    active_set_update!(al)

    # add augmented Lagrangian gradient terms
    G .+= al.∇c' * (al.λ + al.as .* al.ρ .* al.c)
    G[al.idx_l] -= (al.λl + al.asl .* al.ρl .* al.cl)
    G[al.idx_u] += (al.λu + al.asu .* al.ρu .* al.cu)
    return nothing
end

function solve(x, al; alg = :LBFGS, max_iter = 5, c_tol = 1.0e-3)
    # reset augmented Lagrangian
    reset!(al)
    println("solving...")
    for i = 1:max_iter

        # update augmented Lagrangian methods
        _f(z) = f_al(z, al)
        _g!(G, z) = g_al!(G, z, al)

        # solve
        sol = optimize(_f, _g!, x, @eval $alg()) # linesearch = LineSearches.BackTracking()

        # evaluate constraints
        x = sol.minimizer
        c_max = constraint_violation(al, x)
        println("iter: $i")
        println("c_max: $c_max")

        # check for convergence -> update augmented Lagrangian
        if c_max < c_tol
            return x, sol
        else
            c!(al.c, x)
            bounds!(al, x)
            update!(al)
        end
    end

    return x, sol
end

n = prob.num_var
m = prob.num_con
xl, xu = prob.primal_bounds
cl, cu = prob.constraint_bounds
idx_ineq = (1:m)[cu .> cl]
sum(isfinite.(xl))
sum(isfinite.(xu))
sum(cu .> cl)

al = augmented_lagrangian(n, m,
    xl = xl, xu = xu, ρ0 = 100.0, s = 10.0,
    idx_ineq = idx_ineq)

@time x_sol_al, sol = solve(copy(z0), al,
    alg = :BFGS, max_iter = 20, c_tol = 1.0e-2)

# Visualize
using Plots
x̄, ū = unpack(x_sol_al, prob)
plot(hcat(x̄...)', width = 2.0)
plot(hcat(ū...)[1:model.nu, :]', width = 2.0, linetype = :steppost)
s̄ = [ū[t][model.idx_s] for t = 1:T-1]
plot(hcat(s̄...)', width = 2.0, linetype = :steppost)

# using Plots
tf, t, h = get_time(ū)
plot(t[1:end-1], hcat(ū...)[1:2,:]', linetype=:steppost,
	xlabel="time (s)", ylabel = "control",
	label = ["angle" "length"],
	width = 2.0, legend = :top)
plot(t[1:end-1], h, linetype=:steppost)
#
include(joinpath(pwd(), "models/visualize.jl"))
vis = Visualizer()
open(vis)
visualize!(vis, model_ft, state_to_configuration(x̄), Δt = h[1])
