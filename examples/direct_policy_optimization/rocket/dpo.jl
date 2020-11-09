include(joinpath(pwd(), "src/direct_policy_optimization/dpo.jl"))
include(joinpath(@__DIR__, "rocket_nominal.jl"))
include(joinpath(@__DIR__, "rocket_slosh.jl"))

# DPO
N = 2 * model_slosh.n
D = 2 * model_slosh.d

β = 1.0
δ = 1.0e-3

# initial samples
x1_sample = resample(x1_slosh, Diagonal(ones(model_slosh.n)), 1.0e-3)

prob_nom = prob_nominal.prob

# mean problem
prob_mean = trajectory_optimization(
				model_slosh,
				EmptyObjective(),
				T,
				dynamics = false,
				con = con_free_time
				)

# sample problems
prob_sample = [trajectory_optimization(
				model_slosh,
				EmptyObjective(),
				T,
				xl = state_bounds(model_slosh, T, x1 = x1_sample[i])[1],
				xu = state_bounds(model_slosh, T, x1 = x1_sample[i])[2],
				ul = ul,
				uu = uu,
				dynamics = false,
				con = con_free_time
				) for i = 1:N]

# sample objective
Q = [(t < T ? Diagonal([1.0; 10.0; 1.0; 1.0; 10.0; 1.0])
	: Diagonal([10.0; 100.0; 10.0; 10.0; 100.0; 10.0])) for t = 1:T]
R = [Diagonal([1.0e-1; 1.0e-1; 1.0]) for t = 1:T-1]

obj_sample = sample_objective(Q, R)
policy = linear_feedback(6, 2,
	idx_input = collect([1, 2, 3, 5, 6, 7]),
	idx_input_nom = (1:6),
	idx_output = (1:2))
dist = disturbances([Diagonal(δ * ones(model_slosh.n)) for t = 1:T-1])
sample = sample_params(β, T)

prob_dpo = dpo_problem(
	prob_nom, prob_mean, prob_sample,
	obj_sample,
	policy,
	dist,
	sample)

# TVLQR policy
X̄_nom, Ū_nom = unpack(Z̄, prob_nominal)
X̄_slosh, Ū_slosh = unpack(Z̄_slosh, prob_slosh)

K = tvlqr(model_nom, X̄_nom, Ū_nom, Q, R, 0.0)

z0_dpo = zeros(prob_dpo.num_var)
z0_dpo[prob_dpo.prob.idx.nom] = pack(X̄_nom, Ū_nom, prob_nom)
z0_dpo[prob_dpo.prob.idx.mean] = pack(X̄_slosh, Ū_slosh, prob_slosh)
for i = 1:N
	z0_dpo[prob_dpo.prob.idx.sample[i]] = pack(X̄_slosh, Ū_slosh, prob_slosh)
end
for j = 1:(N + D)
	z0_dpo[prob_dpo.prob.idx.slack[j]] = vcat(X̄_slosh[2:end]...)
end
for t = 1:T-1
	z0_dpo[prob_dpo.prob.idx.policy[prob_dpo.prob.idx.θ[t]]] = vec(copy(K[t]))
end

include("/home/taylor/.julia/dev/SNOPT7/src/SNOPT7.jl")

# Solve
Z = solve(prob_dpo, copy(z0_dpo),
	nlp = :SNOPT7,
	tol = 1.0e-2, c_tol = 1.0e-2,
	time_limit = 60 * 60 * 12)




























# TVLQR policy
Q_lqr = [(t < T ? Diagonal([100.0*ones(nq_nom); 100.0*ones(nq_nom)])
   : Diagonal([1000.0*ones(nq_nom); 1000.0*ones(nq_nom)])) for t = 1:T]
R_lqr = [Diagonal(1.0*ones(model_nom.nu)) for t = 1:T-1]
H_lqr = [100.0 for t = 1:T-1]

K = TVLQR_gains(model_nom,X_nom,U_nom,[H_nom[1] for t = 1:T-1],Q_lqr,R_lqr)

# Simulate TVLQR (no slosh)
using Distributions

model_sim = model_nom
x1_sim = copy(x1)
T_sim = 10*T

W = Distributions.MvNormal(zeros(model_sim.nx),
	Diagonal(1.0e-5*ones(model_sim.nx)))
w = rand(W,T_sim)

W0 = Distributions.MvNormal(zeros(model_sim.nx),
	Diagonal(1.0e-5*ones(model_sim.nx)))
w0 = rand(W0,1)

z0_sim = vec(copy(x1_sim) + w0)

t_nom = range(0,stop=sum(H_nom),length=T)
t_sim_nom = range(0,stop=sum(H_nom),length=T_sim)

# simulate
z_tvlqr, u_tvlqr, J_tvlqr, Jx_tvlqr, Ju_tvlqr = simulate_linear_controller(K,
    X_nom,U_nom,model_sim,Q_lqr,R_lqr,T_sim,H_nom[1],z0_sim,w,_norm=2,
	ul=ul,uu=uu)

# plot states
plt_x = plot(t_nom,hcat(X_nom...)[1:model_nom.nx,:]',
	legend=:topright,color=:red,
    label="",width=2.0,xlabel="time (s)",
    title="Rocket",ylabel="state")
plt_x = plot!(t_sim_nom,hcat(z_tvlqr...)[1:model_nom.nx,:]',color=:black,label="",
    width=1.0)

# plot COM
plot_traj = plot(hcat(X_nom...)[1,:],hcat(X_nom...)[2,:],
	legend=:topright,color=:red,
    label="",width=2.0,xlabel="y",ylabel="z",
    title="Rocket")
plot_traj = plot!(hcat(z_tvlqr...)[1,:],hcat(z_tvlqr...)[2,:],
	color=:black,
    label="",width=1.0)

plot(t_nom[1:end-1],hcat(U_nom...)',
	linetype=:steppost,color=:red,width=2.0)
plot!(t_sim_nom[1:end-1],hcat(u_tvlqr...)',
	linetype=:steppost,color=:black,width=1.0)

# objective value
J_tvlqr

# state tracking
Jx_tvlqr

# control tracking
Ju_tvlqr


# Fuel-slosh model
x1_slosh = [x1[1:3];0.0;x1[4:6]...;0.0]
xT_slosh = [xT[1:3];0.0;xT[4:6]...;0.0]

# optimize slosh model

# xl <= x <= xl
xl_slosh = -Inf*ones(model_slosh.nx)
xl_slosh[2] = model_slosh.l1
xu_slosh = Inf*ones(model_slosh.nx)

xl_traj_slosh = [copy(xl_slosh) for t = 1:T]
xu_traj_slosh = [copy(xu_slosh) for t = 1:T]

xl_traj_slosh[1] = copy(x1_slosh)
xu_traj_slosh[1] = copy(x1_slosh)

xl_traj_slosh[T] = copy(xT_slosh)
xu_traj_slosh[T] = copy(xT_slosh)

xl_traj_slosh[T][4] = -Inf
xu_traj_slosh[T][4] = Inf

xl_traj_slosh[T][8] = -Inf
xu_traj_slosh[T][8] = Inf

xl_traj_slosh[T][1] = -r_pad
xu_traj_slosh[T][1] = r_pad
xl_traj_slosh[T][2] = xT[2]-0.001
xu_traj_slosh[T][2] = xT[2]+0.001
xl_traj_slosh[T][3] = -1.0*pi/180.0
xu_traj_slosh[T][3] = 1.0*pi/180.0
xl_traj_slosh[T][5] = -0.001
xu_traj_slosh[T][5] = 0.001
xl_traj_slosh[T][6] = -0.001
xu_traj_slosh[T][6] = 0.001
xl_traj_slosh[T][7] = -0.01*pi/180.0
xu_traj_slosh[T][7] = 0.01*pi/180.0

Q_slosh = [(t != T ? Diagonal([1.0;10.0;1.0;0.1;1.0;10.0;1.0;0.1])
    : Diagonal([10.0;100.0;10.0;0.1;10.0;100.0;10.0;0.1])) for t = 1:T]
obj_slosh = QuadraticTrackingObjective(Q_slosh,R,c,
    [xT_slosh for t=1:T],[zeros(model_slosh.nu) for t=1:T])

# Problem
prob_slosh = init_problem(model_slosh.nx,model_slosh.nu,T,model_slosh,obj_slosh,
                    xl=xl_traj_slosh,
                    xu=xu_traj_slosh,
                    ul=[ul for t=1:T-1],
                    uu=[uu for t=1:T-1],
                    hl=[hl for t=1:T-1],
                    hu=[hu for t=1:T-1],
                    )

# MathOptInterface problem
prob_moi_slosh = init_MOI_Problem(prob_slosh)

# Trajectory initialization
X0_slosh = linear_interp(x1_slosh,xT_slosh,T) # linear interpolation on state
U0_slosh = [ones(model_slosh.nu) for t = 1:T-1] # random controls

# Pack trajectories into vector
Z0_slosh = pack(X0_slosh,U0_slosh,h0,prob_slosh)

#NOTE: may need to run examples multiple times to get good trajectories
# Solve nominal problem
@time Z_nominal_slosh = solve(prob_moi_slosh,copy(Z0_slosh),nlp=:SNOPT7,time_limit=300)
X_nom_slosh, U_nom_slosh, H_nom_slosh = unpack(Z_nominal_slosh,prob_slosh)

plot(hcat(U_nom_slosh...)',linetype=:steppost)
sum(H_nom_slosh) # should be 2.73/2.76
sum(H_nom) # should be 2.72

# simulate slosh with TVLQR controller from nominal model
model_sim = model_slosh
W = Distributions.MvNormal(zeros(model_sim.nx),Diagonal(0.0*ones(model_sim.nx)))
w = rand(W,T_sim)

W0 = Distributions.MvNormal(zeros(model_sim.nx),Diagonal(0.0*ones(model_sim.nx)))
w0 = rand(W0,1)

x1_slosh_sim = x1_slosh
z0_sim = vec(copy(x1_slosh_sim) + w0)

t_nom = range(0,stop=sum(H_nom),length=T)
t_sim_nom = range(0,stop=sum(H_nom),length=T_sim)
dt_sim_nom = sum(H_nom)/(T_sim-1)

z_tvlqr, u_tvlqr, J_tvlqr, Jx_tvlqr, Ju_tvlqr = simulate_linear_controller(K,
    X_nom,U_nom,model_sim,Q_lqr,R_lqr,T_sim,H_nom[1],z0_sim,w,_norm=2,
	controller=:policy,ul=ul,uu=uu)

plt_x = plot(t_nom,hcat(X_nom...)[1:model_nom.nx,:]',
	legend=:topright,color=:red,
    label="",width=2.0,xlabel="time (s)",
    title="Rocket",ylabel="state")
plt_x = plot!(t_sim_nom,hcat(z_tvlqr...)[[(1:3)...,(5:7)...],:]',
	color=:black,label="",
    width=1.0)

plot_traj = plot(hcat(X_nom...)[1,:],hcat(X_nom...)[2,:],
	legend=:topright,color=:red,
    label="",width=2.0,xlabel="y",ylabel="z",
    title="Rocket")
plot_traj = plot!(hcat(z_tvlqr...)[1,:],hcat(z_tvlqr...)[2,:],
	color=:black,
    label="",width=1.0)

# objective value
J_tvlqr

# state tracking
Jx_tvlqr

# control tracking
Ju_tvlqr

visualize!(vis,model_slosh,z_tvlqr,Δt=dt_sim_nom)

# DPO
N = 2*model_slosh.nx
models = [model_slosh for i = 1:N]
β = 1.0
w = 1.0e-3*ones(model_slosh.nx)
γ = N

x1_sample = resample([x1_slosh for i = 1:N],β=β,w=w)

xl_traj_sample = [[-Inf*ones(model_slosh.nx) for t = 1:T] for i = 1:N]
xu_traj_sample = [[Inf*ones(model_slosh.nx) for t = 1:T] for i = 1:N]

for i = 1:N
    xl_traj_sample[i][1] = copy(x1_sample[i])
    xu_traj_sample[i][1] = copy(x1_sample[i])

	# xl_traj_sample[i][T] = copy(xT_slosh)
	# xu_traj_sample[i][T] = copy(xT_slosh)

	# xl_traj_sample[i][T][4] = -Inf
	# xu_traj_sample[i][T][4] = Inf
	#
	# xl_traj_sample[i][T][8] = -Inf
	# xu_traj_sample[i][T][8] = Inf
	#
	# xl_traj_sample[i][T][1] = -0.25
	# xu_traj_sample[i][T][1] = 0.25
	# xl_traj_sample[i][T][2] = xT[2]-0.01
	# xu_traj_sample[i][T][2] = xT[2]+0.01
	# xl_traj_sample[i][T][3] = -1.0*pi/180.0
	# xu_traj_sample[i][T][3] = 1.0*pi/180.0
	# xl_traj_sample[i][T][5] = -0.001
	# xu_traj_sample[i][T][5] = 0.001
	# xl_traj_sample[i][T][6] = -0.001
	# xu_traj_sample[i][T][6] = 0.001
	# xl_traj_sample[i][T][7] = -0.01*pi/180.0
	# xu_traj_sample[i][T][7] = 0.01*pi/180.0
end

ul_traj_sample = [[-Inf*ones(model_slosh.nu) for t = 1:T-1] for i = 1:N]
uu_traj_sample = [[Inf*ones(model_slosh.nu) for t = 1:T-1] for i = 1:N]

prob_sample = init_sample_problem(prob_nom,models,Q_lqr,R_lqr,H_lqr,
	β=β,w=w,γ=γ,
    xl=xl_traj_sample,
    xu=xu_traj_sample,
    n_features=model_slosh.nx-2,
    )

prob_sample_moi = init_MOI_Problem(prob_sample)

Z0_sample = zeros(prob_sample_moi.n)

Z0_sample[prob_sample.idx_nom_z] = pack(X_nom,U_nom,H_nom[1],prob_nom)

for t = 1:T
	for i = 1:N
		Z0_sample[prob_sample.idx_sample[i].x[t]] = X_nom_slosh[t]
		t==T && continue
		Z0_sample[prob_sample.idx_sample[i].u[t]] = U_nom_slosh[t]
		Z0_sample[prob_sample.idx_sample[i].h[t]] = H_nom_slosh[1]
		Z0_sample[prob_sample.idx_x_tmp[i].x[t]] = X_nom_slosh[t+1]
	end
end

for t = 1:T-1
	Z0_sample[prob_sample.idx_K[t]] = vec(K[t])
end

# Solve
# Z_sample_sol = solve(prob_sample_moi,copy(Z0_sample),nlp=:SNOPT7,
# 	time_limit=60*60,tol=1.0e-2,c_tol=1.0e-2)
#
using JLD
# @save joinpath(pwd(),"examples/trajectories/","rocket_fuel_slosh.jld") Z_sample_sol
@load joinpath(pwd(),"examples/trajectories/","rocket_fuel_slosh.jld") Z_sample_sol

X_nom_sample, U_nom_sample, H_nom_sample, X_sample, U_sample, H_sample = unpack(Z_sample_sol,prob_sample)
sum(H_nom_sample) # should be 2.89/2.9
sum(H_nom)

# get policy
Θ = [Z_sample_sol[prob_sample.idx_K[t]] for t = 1:T-1]
Θ_mat = [reshape(Θ[t],model_nom.nu,model_nom.nx) for t = 1:T-1]
t_nom_sample = range(0,stop=sum(H_nom_sample),length=T)
t_sim_nom_sample = range(0,stop=sum(H_nom_sample),length=T_sim)

dt_sim_sample = sum(H_nom_sample)/(T_sim-1)

W = Distributions.MvNormal(zeros(model_sim.nx),Diagonal(0.0*ones(model_sim.nx)))
w = rand(W,T_sim)

W0 = Distributions.MvNormal(zeros(model_sim.nx),Diagonal(0.0*ones(model_sim.nx)))
w0 = rand(W0,1)

z0_sim = vec(copy(x1_slosh_sim) + w0)

z_tvlqr, u_tvlqr, J_tvlqr, Jx_tvlqr, Ju_tvlqr = simulate_linear_controller(K,
    X_nom,U_nom,model_sim,Q_lqr,R_lqr,T_sim,H_nom[1],z0_sim,w,_norm=2,
	controller=:policy,ul=ul,uu=uu)

plt_x = plot(t_nom,hcat(X_nom...)[1:model_nom.nx,:]',legend=:topright,color=:red,
    label="",width=2.0,xlabel="time (s)",
    title="Rocket",ylabel="state")
plt_x = plot!(t_sim_nom,hcat(z_tvlqr...)[[(1:3)...,(5:7)...],:]',color=:black,label="",
    width=1.0)

plot_traj = plot(hcat(X_nom...)[1,:],hcat(X_nom...)[2,:],legend=:topright,color=:red,
    label="",width=2.0,xlabel="y",ylabel="z",
    title="Rocket")
plot_traj = plot!(hcat(z_tvlqr...)[1,:],hcat(z_tvlqr...)[2,:],color=:black,
    label="",width=1.0)

z_sample, u_sample, J_sample, Jx_sample, Ju_sample= simulate_linear_controller(Θ,
    X_nom_sample,U_nom_sample,model_sim,Q_lqr,R_lqr,T_sim,H_nom_sample[1],z0_sim,
	w,_norm=2,
	controller=:policy,ul=ul,uu=uu)

plt_x = plot(t_nom_sample,hcat(X_nom_sample...)[1:model_nom.nx,:]',legend=:topright,color=:red,
    label="",width=2.0,xlabel="time (s)",
    title="Rocket",ylabel="state")
plt_x = plot!(t_sim_nom_sample,hcat(z_sample...)[[(1:3)...,(5:7)...],:]',color=:black,label="",
    width=1.0)

plot_traj = plot(hcat(X_nom_sample...)[1,:],hcat(X_nom_sample...)[2,:],legend=:topright,color=:red,
    label="",width=2.0,xlabel="y",ylabel="z",
    title="Rocket")
plot_traj = plot!(hcat(z_sample...)[1,:],hcat(z_sample...)[2,:],color=:black,
    label="",width=1.0)

# objective value
J_sample
J_tvlqr

# state tracking
Jx_sample
Jx_tvlqr

# control tracking
Ju_sample
Ju_tvlqr

plot_traj = plot(hcat(X_nom...)[1,:],hcat(X_nom...)[2,:],legend=:topright,color=:cyan,
    label="",width=2.0,xlabel="y",ylabel="z",
    title="Rocket")

plot_traj = plot!(hcat(X_nom_sample...)[1,:],hcat(X_nom_sample...)[2,:],legend=:topright,color=:orange,
    label="",width=2.0,xlabel="y",ylabel="z",
    title="Rocket")

for i = 1:N
	plot_traj = plot!(hcat(X_sample[i]...)[1,:],hcat(X_sample[i]...)[2,:],label="")
end
display(plot_traj)

# orientation tracking
plt_x = plot(t_nom,hcat(X_nom...)[3,:],legend=:topright,color=:red,
    label="",width=2.0,xlabel="time (s)",
    title="Rocket",ylabel="state")
plt_x = plot!(t_sim_nom,hcat(z_tvlqr...)[3,:],color=:black,label="",
    width=1.0)

plt_x = plot(t_nom_sample,hcat(X_nom_sample...)[3,:],legend=:topright,color=:red,
    label="",width=2.0,xlabel="time (s)",
    title="Rocket",ylabel="state")
plt_x = plot!(t_sim_nom_sample,hcat(z_sample...)[3,:],color=:black,label="",
    width=1.0)

sum(H_nom_sample)
sum(H_nom)

visualize!(vis,model_nom,z_tvlqr,Δt=dt_sim_nom)
visualize!(vis,model_nom,z_sample,Δt=dt_sim_sample)
using PGFPlots
const PGF = PGFPlots

p_traj_nom = PGF.Plots.Linear(hcat(X_nom...)[1,:],hcat(X_nom...)[2,:],
	mark="",style="color=cyan, line width = 2pt",legendentry="TO")
p_traj_sample = PGF.Plots.Linear(hcat(X_nom_sample...)[1,:],hcat(X_nom_sample...)[2,:],
	mark="",style="color=orange, line width = 2pt",legendentry="DPO")

a = Axis([p_traj_nom;p_traj_sample],
    axisEqualImage=false,
    hideAxis=false,
	ylabel="z",
	xlabel="y",
	legendStyle="{at={(0.01,0.99)},anchor=north west}",
	)

# Save to tikz format
dir = joinpath(@__DIR__,"results")
PGF.save(joinpath(dir,"rocket_traj.tikz"), a, include_preamble=false)


# orientation tracking
plt_x = plot(t_nom,hcat(X_nom...)[3,:],legend=:topright,color=:red,
    label="",width=2.0,xlabel="time (s)",
    title="Rocket",ylabel="state")
plt_x = plot!(t_sim_nom,hcat(z_tvlqr...)[3,:],color=:black,label="",
    width=1.0)

plt_x = plot(t_nom_sample,hcat(X_nom_sample...)[3,:],legend=:topright,color=:red,
    label="",width=2.0,xlabel="time (s)",
    title="Rocket",ylabel="state")
plt_x = plot!(t_sim_nom_sample,hcat(z_sample...)[3,:],color=:black,label="",
    width=1.0)

p_nom_orientation = PGF.Plots.Linear(t_nom,hcat(X_nom...)[3,:],
	mark="",style="color=cyan, line width = 2pt",legendentry="TO")
p_nom_sim_orientation = PGF.Plots.Linear(t_sim_nom,hcat(z_tvlqr...)[3,:],
	mark="",style="color=black, line width = 1pt")

p_sample_orientation = PGF.Plots.Linear(t_nom_sample,hcat(X_nom_sample...)[3,:],
	mark="",style="color=orange, line width = 2pt",legendentry="DPO")
p_sample_sim_orientation = PGF.Plots.Linear(t_sim_nom_sample,hcat(z_sample...)[3,:],
	mark="",style="color=black, line width = 1pt")

a = Axis([p_nom_orientation;p_nom_sim_orientation],
    axisEqualImage=false,
    hideAxis=false,
	ylabel="orientation",
	xlabel="time",
	legendStyle="{at={(0.5,0.99)},anchor=north}",
	)

# Save to tikz format
dir = joinpath(@__DIR__,"results")
PGF.save(joinpath(dir,"rocket_tvlqr_orientation.tikz"), a, include_preamble=false)

a = Axis([p_sample_orientation;p_sample_sim_orientation],
    axisEqualImage=false,
    hideAxis=false,
	ylabel="orientation",
	xlabel="time",
	legendStyle="{at={(0.5,0.99)},anchor=north}",
	)

# Save to tikz format
dir = joinpath(@__DIR__,"results")
PGF.save(joinpath(dir,"rocket_dpo_orientation.tikz"), a, include_preamble=false)


# eigen value analysis
C = Diagonal(ones(model_slosh.nx))[1:model_nom.nx,:]
# nominal
A_nom, B_nom = nominal_jacobians(model_nom,X_nom,U_nom,H_nom)
A_nom_cl = [(A_nom[t] - B_nom[t]*K[t]) for t = 1:T-1]
sv_nom = [norm.(eigen(A_nom_cl[t]).values) for t = 1:T-1]
plt_nom = plot(hcat(sv_nom...)',xlabel="time step t",ylabel="eigen value norm",
	title="TVLQR nominal model",linetype=:steppost,
	ylims=(-3,3),labels="")

# slosh nominal
X_nom_slosh = [[copy(X_nom[t]);0.0;0.0] for t = 1:T]
A_nom_slosh, B_nom_slosh = nominal_jacobians(model_slosh,X_nom_slosh,U_nom,H_nom)
A_nom_slosh_cl = [(A_nom_slosh[t] - B_nom_slosh[t]*K[t]*C) for t = 1:T-1]
sv_nom_slosh = [norm.(eigen(A_nom_slosh_cl[t]).values) for t = 1:T-1]
plt_nom_slosh = plot(hcat(sv_nom_slosh...)',xlabel="time step t",ylabel="eigen value norm",
	title="TVLQR slosh model",linetype=:steppost,
	ylims=(-3,3),labels="")

# slosh
A_dpo, B_dpo = nominal_jacobians(model_nom,X_nom_sample,U_nom_sample,H_nom_sample)
A_dpo_cl = [(A_dpo[t] - B_dpo[t]*Θ_mat[t]) for t = 1:T-1]
sv_dpo = [norm.(eigen(A_dpo_cl[t]).values) for t = 1:T-1]
plt_dpo_nom = plot(hcat(sv_dpo...)',xlabel="time step t",ylabel="singular value",
	title="DPO nominal model",linetype=:steppost,
	ylims=(-1,3),labels="")

X_dpo_slosh = [[copy(X_nom_sample[t]);0.0;0.0] for t = 1:T]
A_dpo_slosh, B_dpo_slosh = nominal_jacobians(model_slosh,X_dpo_slosh,U_nom_sample,H_nom_sample)
A_dpo_slosh_cl = [(A_dpo_slosh[t] - B_dpo_slosh[t]*Θ_mat[t]*C) for t = 1:T-1]
sv_dpo_slosh = [norm.(eigen(A_dpo_slosh_cl[t]).values) for t = 1:T-1]
plt_dpo_slosh = plot(hcat(sv_dpo_slosh...)',xlabel="time step t",ylabel="eigen value norm",
	title="DPO slosh model",linetype=:steppost,
	ylims=(-1.0,3.0),labels="")

plot(plt_nom,plt_dpo_nom,layout=(2,1))

plot(plt_nom_slosh,plt_dpo_slosh,layout=(2,1))


plot(t_nom,hcat(X_nom...)[1:6,:]',color=:cyan,label=["y" "z" "ϕ"])
plot!(t_nom_sample, hcat(X_nom_sample...)[1:6,:]',color=:orange,label=["y" "z" "ϕ"])
