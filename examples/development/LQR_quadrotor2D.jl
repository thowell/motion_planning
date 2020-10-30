include(joinpath(pwd(),"src/models/quadrotor2D.jl"))
using Plots

nx = model.nx
nu = model.nu

# horizon
T = 51
Δt = 0.1

Δt*(T-1)

# Initial and final states
x1_nom = [0.0; 1.0; 0.0; 0.0; 0.0; 0.0]
xT_nom = [1.0; 1.0; 0.0; 0.0; 0.0; 0.0]

x_nom_ref = linear_interp(x1_nom,xT_nom,T)

Q_nom = [(t < T ? Diagonal([1.0; 1.0; 1.0; 1.0; 1.0; 1.0])
	: Diagonal([10.0; 10.0; 10.0; 10.0; 10.0; 10.0])) for t = 1:T]
R_nom = [Diagonal(1.0e-1*ones(model.nu)) for t = 1:T-1]
c_nom = 0.0

xl_traj = [-Inf*ones(nx) for t = 1:T]
xu_traj = [Inf*ones(nx) for t = 1:T]

xl_traj[1] = x1_nom
xu_traj[1] = x1_nom

xl_traj[T] = xT_nom
xu_traj[T] = xT_nom

ul_traj = [zeros(nu) for t = 1:T]
uu_traj = [Inf*ones(nu) for t = 1:T]

hl = Δt
hh = Δt

obj = QuadraticTrackingObjective(
	Q_nom,
	R_nom,
	c_nom,
    [xT_nom for t=1:T],[zeros(nu) for t=1:T])

# Problem
prob = init_problem(nx,nu,T,model,obj,
                    xl=xl_traj,
                    xu=xu_traj,
                    ul=ul_traj,
                    uu=uu_traj,
                    hl=[hl for t=1:T-1],
                    hu=[hh for t=1:T-1],
                    )

# MathOptInterface problem
prob_moi = init_MOI_Problem(prob)

# Trajectory initialization
X0 = x_nom_ref # linear interpolation on state
U0 = [0.1*ones(nu) for t = 1:T-1] # random controls

# Pack trajectories into vector
Z0 = pack(X0,U0,Δt,prob)

# Solve nominal problem
@time Z_nominal = solve(prob_moi,copy(Z0),nlp=:SNOPT7)
X_nom, U_nom, H_nom = unpack(Z_nominal,prob)

Plots.plot(hcat(X_nom...)')
Plots.plot(hcat(U_nom...)')

obj_fixed = obj = QuadraticTrackingObjective(
	[Diagonal(zeros(nx)) for t = 1:T],
	[Diagonal(zeros(nu)) for t = 1:T-1],
	0.0,
    [zeros(nx) for t=1:T],[zeros(nu) for t=1:T])

xl_traj = [zeros(nx) for t = 1:T]
xu_traj = [zeros(nx) for t = 1:T]

ul_traj = [zeros(nu) for t = 1:T]
uu_traj = [zeros(nu) for t = 1:T]

prob_fixed = init_problem(nx,nu,T,model,obj,
                    xl=xl_traj,
                    xu=xu_traj,
                    ul=ul_traj,
                    uu=uu_traj,
                    hl=[Δt for t=1:T-1],
                    hu=[Δt for t=1:T-1],
                    )
# Sample
Q_lqr = [(t < T ? Diagonal([10.0;10.0;10.0;10.0;10.0;10.0])
	: Diagonal([100.0;100.0;100.0;100.0;100.0;100.0])) for t = 1:T]
R_lqr = [Diagonal(ones(nu)) for t = 1:T-1]
H_lqr = [0.0 for t = 1:T-1]
A_dyn, B_dyn = nominal_jacobians(model,X_nom,U_nom,[Δt for t = 1:T-1])
K = TVLQR_gains(model,X_nom,U_nom,[Δt for t = 1:T-1],Q_lqr,R_lqr)

α = 1.0
x11 = α*[1.0; 0.0; 0.0; 0.0; 0.0; 0.0]
x12 = α*[-1.0; 0.0; 0.0; 0.0; 0.0; 0.0]
x13 = α*[0.0; 1.0; 0.0; 0.0; 0.0; 0.0]
x14 = α*[0.0; -1.0; 0.0; 0.0; 0.0; 0.0]
x15 = α*[0.0; 0.0; 1.0; 0.0; 0.0; 0.0]
x16 = α*[0.0; 0.0; -1.0; 0.0; 0.0; 0.0]
x17 = α*[0.0; 0.0; 0.0; 1.0; 0.0; 0.0]
x18 = α*[0.0; 0.0; 0.0; -1.0; 0.0; 0.0]
x19 = α*[0.0; 0.0; 0.0; 0.0; 1.0; 0.0]
x110 = α*[0.0; 0.0; 0.0; 0.0; -1.0; 0.0]
x111 = α*[0.0; 0.0; 0.0; 0.0; 0.0; 1.0]
x112 = α*[0.0; 0.0; 0.0; 0.0; 0.0; -1.0]

x1_sample = [x11,x12,x13,x14,x15,x16,x17,x18,x19,x110,x111,x112]

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

prob_sample = init_sample_problem(prob_fixed,models,Q_lqr,R_lqr,H_lqr,
    xl=xl_traj_sample,
    xu=xu_traj_sample,
	ul=ul_traj_sample,
    uu=uu_traj_sample,
    β=β,w=w,γ=γ)

prob_sample_moi = init_MOI_Problem(prob_sample)

Z0_sample = ones(prob_sample_moi.n)

# linear dynamics
function discrete_dynamics(model::Quadrotor2D,x⁺,x,u,h,t)
    x⁺ - A_dyn[t]*x - B_dyn[t]*u
end

# Solve
Z_sample_sol = solve(prob_sample_moi,copy(Z0_sample),nlp=:SNOPT7,
	time_limit=60*10,tol=1.0e-7,c_tol=1.0e-7)

using JLD
@save joinpath(pwd(),"examples/trajectories/","planar_quad_LQR.jld") Z_sample_sol
# @load joinpath(pwd(),"examples/trajectories/","planar_quad_LQR.jld") Z_sample_sol

# Unpack solutions
X_nom_sample, U_nom_sample, H_nom_sample, X_sample, U_sample, H_sample = unpack(Z_sample_sol,
	prob_sample)

Θ_linear = [reshape(Z_sample_sol[prob_sample.idx_K[t]],nu,nx) for t = 1:T-1]
policy_error_linear = [norm(vec(Θ_linear[t]-K[t]))/norm(vec(K[t])) for t = 1:T-1]
println("Policy solution error (Inf norm) [linear dynamics]:
    $(norm(policy_error_linear,Inf))")

plt = plot(policy_error_linear,xlabel="time step",ylabel="matrix-norm error",yaxis=:log,
    ylims=(1.0e-16,1.0),width=2.0,legend=:bottom,label="")
savefig(plt,joinpath(@__DIR__,"results/TVLQR_quadrotor2D.png"))

using PGFPlots
const PGF = PGFPlots

px = PGF.Plots.Linear(range(0,stop=Δt*(T-1),length=T),hcat(X_nom...)[1,:],
	mark="",style="color = orange, line width = 2pt",legendentry="x")
pz = PGF.Plots.Linear(range(0,stop=Δt*(T-1),length=T),hcat(X_nom...)[2,:],
	mark="",style="color = purple, line width = 2pt",legendentry="y")
pθ = PGF.Plots.Linear(range(0,stop=Δt*(T-1),length=T),hcat(X_nom...)[3,:],
	mark="",style="color = cyan, line width = 2pt",legendentry="theta")

a = Axis([px;pz;pθ],
    xmin=0.0, ymin=-0.1, xmax=5.0, ymax=1.05,
    axisEqualImage=false,
    hideAxis=false,
	ylabel="configuration",
	xlabel="time",
	legendStyle="{at={(0.01,0.99)},anchor=north west}"
	)

# Save to tikz format
dir = joinpath(@__DIR__,"results")
PGF.save(joinpath(dir,"quadrotor2D_traj.tikz"), a, include_preamble=false)

p = PGF.Plots.Linear(range(1,stop=T-1,length=T-1),policy_error_linear,mark="",style="color=black, line width = 2pt")

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
PGF.save(joinpath(dir,"TVLQR_quadrotor2D.tikz"), a, include_preamble=false)
