include(joinpath(pwd(),"src/direct_policy_optimization.jl"))
include(joinpath(pwd(),"dynamics/double_integrator.jl"))

model = model_analytical
nx = model.nx
nu = model.nu
A, B = get_dynamics(model)

# horizon
T = 51

xl_traj = [zeros(nx) for t = 1:T]
xu_traj = [zeros(nx) for t = 1:T]

ul_traj = [zeros(nu) for t = 1:T]
uu_traj = [zeros(nu) for t = 1:T]

obj = QuadraticTrackingObjective(
	[Diagonal(zeros(nx)) for t = 1:T],
	[Diagonal(zeros(nu)) for t = 1:T-1],
	0.0,
    [zeros(nx) for t=1:T],[zeros(nu) for t=1:T])

# Problem
prob = init_problem(nx,nu,T,model,obj,
                    xl=[zeros(nx) for t = 1:T],
                    xu=[zeros(nx) for t = 1:T],
                    ul=[zeros(nu) for t = 1:T-1],
                    uu=[zeros(nu) for t = 1:T-1],
                    hl=[0.0 for t=1:T-1],
                    hu=[0.0 for t=1:T-1],
                    )

# MathOptInterface problem
prob_moi = init_MOI_Problem(prob)

# Trajectory initialization
X0 = linear_interp(zeros(nx),zeros(nx),T) # linear interpolation on state
U0 = [zeros(nu) for t = 1:T-1] # random controls

# Pack trajectories into vector
Z0 = pack(X0,U0,model.Δt,prob)

# Solve nominal problem
@time Z_nominal = solve(prob_moi,copy(Z0),nlp=:SNOPT7)
X_nom, U_nom, H_nom = unpack(Z_nominal,prob)

# Sample
Q_lqr = [Diagonal(ones(nx)) for t = 1:T]
R_lqr = [Diagonal(ones(nu)) for t = 1:T-1]
H_lqr = [0.0 for t = 1:T-1]

K = TVLQR([A for t=1:T-1],[B for t=1:T-1],[Q_lqr[t] for t=1:T],[R_lqr[t] for t=1:T-1])

α = 1.0
x11 = α*[1.0; 1.0]
x12 = α*[1.0; -1.0]
x13 = α*[-1.0; 1.0]
x14 = α*[-1.0; -1.0]
x1_sample = [x11,x12,x13,x14]

N = 2*nx
models = [model for i = 1:N]
β = 1.0
w = 1.0e-1*ones(nx)
γ = N

xl_traj_sample = [[-Inf*ones(nx) for t = 1:T] for i = 1:N]
xu_traj_sample = [[Inf*ones(nx) for t = 1:T] for i = 1:N]

ul_traj_sample = [[-Inf*ones(nu) for t = 1:T-1] for i = 1:N]
uu_traj_sample = [[Inf*ones(nu) for t = 1:T-1] for i = 1:N]

for i = 1:N
    xl_traj_sample[i][1] = x1_sample[i]
    xu_traj_sample[i][1] = x1_sample[i]
end

prob_sample = init_sample_problem(prob,models,Q_lqr,R_lqr,H_lqr,
    xl=xl_traj_sample,
    xu=xu_traj_sample,
	ul=ul_traj_sample,
    uu=uu_traj_sample,
    β=β,w=w,γ=γ)


prob_sample_moi = init_MOI_Problem(prob_sample)

Z0_sample = ones(prob_sample_moi.n)

# Solve
Z_sample_sol = solve(prob_sample_moi,copy(Z0_sample),nlp=:SNOPT7,
	time_limit=60,tol=1.0e-7,c_tol=1.0e-7)

using JLD
@save joinpath(pwd(),"examples/trajectories/","double_integrator_LQR.jld") Z_sample_sol
# @load joinpath(pwd(),"examples/trajectories/","double_integrator_LQR.jld") Z_sample_sol

# Unpack solutions
X_nom_sample, U_nom_sample, H_nom_sample, X_sample, U_sample, H_sample = unpack(Z_sample_sol,prob_sample)

μ_sample = [sample_mean([X_sample[i][t] for i = 1:N]) for t = 1:T]

μ_sample[1]'Q_lqr
sep_pr = maximum([μ_sample[t]'*Q_lqr[t]*μ_sample[t] for t = 1:T])
plot(hcat(μ_sample...)')

Θ = [reshape(Z_sample_sol[prob_sample.idx_K[t]],nu,nx) for t = 1:T-1]
policy_error = [norm(vec(Θ[t]-K[t]))/norm(vec(K[t])) for t = 1:T-1]
println("Policy solution error (Inf norm): $(norm(policy_error,Inf))")

using Plots
plt = plot(policy_error,xlabel="time step",ylims=(1.0e-16,1.0),yaxis=:log,
    width=2.0,legend=:bottom,ylabel="matrix-norm error",label="")
savefig(plt,joinpath(@__DIR__,"results/LQR_double_integrator.png"))

using PGFPlots
const PGF = PGFPlots

p = PGF.Plots.Linear(range(1,stop=T-1,length=T-1),policy_error,mark="",style="color=black, very thick")

a = Axis(p,
    xmin=1., ymin=1.0e-16, xmax=T-1, ymax=1.0,
    axisEqualImage=false,
    hideAxis=false,
	ylabel="matrix-norm error",
	xlabel="time step",
	ymode="log",
	)

# Save to tikz format
dir = joinpath(@__DIR__,"results")
PGF.save(joinpath(dir,"LQR_double_integrator.tikz"), a, include_preamble=false)
