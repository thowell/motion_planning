include(joinpath(pwd(), "src/direct_policy_optimization/dpo.jl"))
include(joinpath(@__DIR__, "quadrotor_broken_propeller.jl"))

# Additive noise model
model = additive_noise_model(model)

function fd(model::Quadrotor, x⁺, x, u, w, h, t)
    midpoint_implicit(model, x⁺, x, u, w, h) - w
end

# Nominal solution
X̄, Ū = unpack(Z̄, prob)
prob_nom = prob.prob

# DPO
N = 2 * model.n
D = 2 * model.d

β = 1.0
δ = 1.0e-3

# initial samples
x1_sample = resample(x1, Diagonal(ones(model.n)), 1.0e-3)

# mean problem
prob_mean = trajectory_optimization(
				model,
				EmptyObjective(),
				T,
				ul = control_bounds(model, T, [Inf * ones(4); 0.0], [Inf * ones(4); 0.0])[1],
				uu = control_bounds(model, T, [Inf * ones(4); 0.0], [Inf * ones(4); 0.0])[2],
				dynamics = false
				)

# sample problems
uu_sample = []
for i = 1:6
	push!(uu_sample, [uu1; Inf])
end
for i = 1:6
	push!(uu_sample, [uu2; Inf])
end
for i = 1:6
	push!(uu_sample, [uu3; Inf])
end
for i = 1:6
	push!(uu_sample, [uu4; Inf])
end

prob_sample = [trajectory_optimization(
				model,
				EmptyObjective(),
				T,
				xl = state_bounds(model, T, x1 = x1_sample[i])[1],
				xu = state_bounds(model, T, x1 = x1_sample[i])[2],
				ul = ul,
				uu = control_bounds(model, T, Inf * ones(model.m), uu_sample[i]),
				dynamics = false,
				con = con_free_time
				) for i = 1:N]

# sample objective
Q = [(t < T ? Diagonal([10.0; 10.0; 1.0])
	: Diagonal(100.0 * ones(model.n))) for t = 1:T]
R = [Diagonal(1.0e-1 * ones(model.m)) for t = 1:T-1]

obj_sample = sample_objective(Q, R)
policy = linear_feedback(model.n, model.m-1)
dist = disturbances([Diagonal(δ * [1.0; 1.0; 0.1]) for t = 1:T-1])
sample = sample_params(β, T)

prob_dpo = dpo_problem(
	prob_nom, prob_mean, prob_sample,
	obj_sample,
	policy,
	dist,
	sample)

# TVLQR policy
K = tvlqr(model, X̄, Ū, Q, R, h)

z0_dpo = zeros(prob_dpo.num_var)
z0_dpo[prob_dpo.prob.idx.nom] = pack(X̄, Ū, prob_nom)
z0_dpo[prob_dpo.prob.idx.mean] = pack(X̄, Ū, prob_nom)
for i = 1:N
	z0_dpo[prob_dpo.prob.idx.sample[i]] = pack(X̄, Ū, prob_nom)
end
for j = 1:(N + D)
	z0_dpo[prob_dpo.prob.idx.slack[j]] = vcat(X̄[2:end]...)
end
for t = 1:T-1
	z0_dpo[prob_dpo.prob.idx.policy[prob_dpo.prob.idx.θ[t]]] = vec(copy(K[t]))
end

# Solve
Z = solve(prob_dpo, copy(z0_dpo),
	tol = 1.0e-3, c_tol = 1.0e-3,
	max_iter = 1000)











# Simulate TVLQR

# TVLQR cost
Q_lqr = [t < T ? Diagonal(10.0*ones(model.nx)) : Diagonal(100.0*ones(model.nx)) for t = 1:T]
R_lqr = [Diagonal(1.0*ones(model.nu)) for t = 1:T-1]
H_lqr = [1.0 for t = 1:T-1]

K = TVLQR_gains(model,X_nom,U_nom,[H_nom[1] for t = 1:T-1],Q_lqr,R_lqr)



# Sample
N = 2*model.nx
models = [model for i = 1:N]
β = 1.0
w = 1.0e-3*ones(model.nx)
γ = N
x1_sample = resample([x1 for i = 1:N],β=1.0,w=1.0e-3*ones(model.nx))

xl_traj_sample = [[copy(xl) for t = 1:T] for i = 1:N]
xu_traj_sample = [[copy(xu) for t = 1:T] for i = 1:N]

ul_traj_sample = [[copy(ul_nom) for t = 1:T-1] for i = 1:N]
uu_traj_sample = [[copy(uu_nom) for t = 1:T-1] for i = 1:N]
for t = 1:T-1
	for i = 1:6
		uu_traj_sample[i][t][1] *= 0.5
	end
	for i = 6 .+ (1:6)
		uu_traj_sample[i][t][2] *= 0.5
	end
	for i = 12 .+ (1:6)
		uu_traj_sample[i][t][3] *= 0.5
	end
	for i = 18 .+ (1:6)
		uu_traj_sample[i][t][4] *= 0.5
	end
end

for i = 1:N
    xl_traj_sample[i][1] = copy(x1_sample[i])
    xu_traj_sample[i][1] = copy(x1_sample[i])
end

prob_sample = init_sample_problem(prob,models,Q_lqr,R_lqr,H_lqr,
    xl=xl_traj_sample,
    xu=xu_traj_sample,
	ul=ul_traj_sample,
    uu=uu_traj_sample,
    β=β,w=w,γ=γ)

prob_sample_moi = init_MOI_Problem(prob_sample)

Z0_sample = pack(X_nom,U_nom,H_nom[1],K,prob_sample,r=0.001)


# # Solve
# Z_sample_sol = solve(prob_sample_moi,copy(Z0_sample),nlp=:SNOPT7,
#     time_limit=60*60,tol=1.0e-2,c_tol=1.0e-2)

using JLD
# @save joinpath(pwd(),"quadrotor_prop_fail.jld") Z_sample_sol
@load joinpath(pwd(),"quadrotor_prop_fail.jld") Z_sample_sol

# Z_sample_sol = solve(prob_sample_moi,copy(Z_sample_sol),nlp=:SNOPT7,
#     time_limit=60*60,tol=1.0e-2,c_tol=1.0e-2)

# Unpack solutions
X_nom_sample, U_nom_sample, H_nom_sample, X_sample, U_sample, H_sample = unpack(Z_sample_sol,prob_sample)

K_sample = [reshape(Z_sample_sol[prob_sample.idx_K[t]],model.nu,model.nx) for t = 1:T-1]
# Time trajectories
t_nominal = zeros(T)
t_sample = zeros(T)
for t = 2:T
    t_nominal[t] = t_nominal[t-1] + H_nom[t-1]
    t_sample[t] = t_sample[t-1] + H_nom_sample[t-1]
end

display("time (nominal): $(sum(H_nom))s")
display("time (sample): $(sum(H_nom_sample))s")

# # Plots results
using Plots
# # Position trajectory
x_nom_pos = [X_nom[t][1] for t = 1:T]
y_nom_pos = [X_nom[t][2] for t = 1:T]
# pts = Plots.partialcircle(0,2π,100,r)
# cx,cy = Plots.unzip(pts)
# cx1 = [_cx + xc1 for _cx in cx]
# cy1 = [_cy + yc1 for _cy in cy]
# cx2 = [_cx + xc2 for _cx in cx]
# cy2 = [_cy + yc2 for _cy in cy]
# cx3 = [_cx + xc3 for _cx in cx]
# cy3 = [_cy + yc3 for _cy in cy]
# cx4 = [_cx + xc4 for _cx in cx]
# cy4 = [_cy + yc4 for _cy in cy]
# # cx5 = [_cx + xc5 for _cx in cx]
# # cy5 = [_cy + yc5 for _cy in cy]
#
# plt = plot(Shape(cx1,cy1),color=:red,label="",linecolor=:red)
# # plt = plot!(Shape(cx2,cy2),color=:red,label="",linecolor=:red)
# # plt = plot!(Shape(cx3,cy3),color=:red,label="",linecolor=:red)
# # plt = plot!(Shape(cx4,cy4),color=:red,label="",linecolor=:red)
# # # plt = plot(Shape(cx5,cy5),color=:red,label="",linecolor=:red)
#
# for i = 1:N
#     x_sample_pos = [X_sample[i][t][1] for t = 1:T]
#     y_sample_pos = [X_sample[i][t][2] for t = 1:T]
#     plt = plot!(x_sample_pos,y_sample_pos,aspect_ratio=:equal,
#         color=:cyan,label= i != 1 ? "" : "sample")
# end
# display(plt)
plt = plot()
plt = scatter!(x_nom_pos,y_nom_pos,aspect_ratio=:equal,xlabel="x",ylabel="y",width=4.0,label="TO",color=:purple,legend=:topleft)
x_sample_pos = [X_nom_sample[t][1] for t = 1:T]
y_sample_pos = [X_nom_sample[t][2] for t = 1:T]
plt = plot!(x_sample_pos,y_sample_pos,aspect_ratio=:equal,width=4.0,label="DPO",color=:orange,legend=:bottomright)
#
# savefig(plt,joinpath(@__DIR__,"results/quadrotor_trajectory.png"))
#
# Control
plt = plot(t_nominal[1:T-1],Array(hcat(U_nom...))',color=:purple,width=2.0,
    title="quad",xlabel="time (s)",ylabel="control",
    legend=:bottom,linetype=:steppost)
plt = plot!(t_sample[1:T-1],Array(hcat(U_nom_sample...))',color=:orange,
    width=2.0,linetype=:steppost)
savefig(plt,joinpath(@__DIR__,"results/quadrotor_control.png"))

# # Samples
X_sample
# State samples
plt1 = plot(title="Sample states",legend=:bottom,xlabel="time (s)");
for i = 1:N
    # t_sample = zeros(T)
    # for t = 2:T
    #     t_sample[t] = t_sample[t-1] + H_nom_sample[t-1]
    # end
    plt1 = plot!(hcat(X_sample[i]...)',label="");
end
plt1 = plot!(hcat(X_nom_sample...)',color=:red,width=2.0,
    label=["nominal" "" ""])
display(plt1)
# savefig(plt1,joinpath(@__DIR__,"results/quadrotor_sample_states.png"))
#
# # Control samples
# plt2 = plot(title="Sample controls",xlabel="time (s)",legend=:bottom);
# for i = 1:N
#     t_sample = zeros(T)
#     for t = 2:T
#         t_sample[t] = t_sample[t-1] + H_nom_sample[t-1]
#     end
#     plt2 = plot!(t_sample[1:end-1],hcat(U_sample[i]...)',label="",
#         linetype=:steppost);
# end
# plt2 = plot!(t_sample[1:end-1],hcat(U_nom_sample...)',color=:red,width=2.0,
#     label=["nominal" ""],linetype=:steppost)
# display(plt2)
# savefig(plt2,joinpath(@__DIR__,"results/quadrotor_sample_controls.png"))


using Distributions

model_sim = model
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

t_nom_sample = range(0,stop=sum(H_nom_sample),length=T)
t_sim_nom_sample = range(0,stop=sum(H_nom_sample),length=T_sim)

dt_sim_sample = sum(H_nom_sample)/(T_sim-1)
dt_sim_nom = sum(H_nom)/(T_sim-1)

# simulate
z_tvlqr1, u_tvlqr1, J_tvlqr1, Jx_tvlqr1, Ju_tvlqr1 = simulate_linear_controller(K,
    X_nom,U_nom,model_sim,Q_lqr,R_lqr,T_sim,H_nom[1],z0_sim,w,_norm=2,
	ul=ul_nom,uu=uu1)
z_tvlqr2, u_tvlqr2, J_tvlqr2, Jx_tvlqr2, Ju_tvlqr2 = simulate_linear_controller(K,
    X_nom,U_nom,model_sim,Q_lqr,R_lqr,T_sim,H_nom[1],z0_sim,w,_norm=2,
	ul=ul_nom,uu=uu2)
z_tvlqr3, u_tvlqr3, J_tvlqr3, Jx_tvlqr3, Ju_tvlqr3 = simulate_linear_controller(K,
    X_nom,U_nom,model_sim,Q_lqr,R_lqr,T_sim,H_nom[1],z0_sim,w,_norm=2,
	ul=ul_nom,uu=uu3)
z_tvlqr4, u_tvlqr4, J_tvlqr4, Jx_tvlqr4, Ju_tvlqr4 = simulate_linear_controller(K,
    X_nom,U_nom,model_sim,Q_lqr,R_lqr,T_sim,H_nom[1],z0_sim,w,_norm=2,
	ul=ul_nom,uu=uu4)

z_sample1, u_sample1, J_sample1, Jx_sample1, Ju_sample1 = simulate_linear_controller(K_sample,
    X_nom_sample,U_nom_sample,model_sim,Q_lqr,R_lqr,T_sim,H_nom_sample[1],z0_sim,w,_norm=2,
	ul=ul_nom,uu=uu1)
z_sample2, u_sample2, J_sample2, Jx_sample2, Ju_sample2 = simulate_linear_controller(K_sample,
    X_nom_sample,U_nom_sample,model_sim,Q_lqr,R_lqr,T_sim,H_nom_sample[1],z0_sim,w,_norm=2,
	ul=ul_nom,uu=uu2)
z_sample3, u_sample3, J_sample3, Jx_sample3, Ju_sample3 = simulate_linear_controller(K_sample,
    X_nom_sample,U_nom_sample,model_sim,Q_lqr,R_lqr,T_sim,H_nom_sample[1],z0_sim,w,_norm=2,
	ul=ul_nom,uu=uu3)
z_sample4, u_sample4, J_sample4, Jx_sample4, Ju_sample4 = simulate_linear_controller(K_sample,
    X_nom_sample,U_nom_sample,model_sim,Q_lqr,R_lqr,T_sim,H_nom_sample[1],z0_sim,w,_norm=2,
	ul=ul_nom,uu=uu4)

# plot states
plt_x = plot(t_nom,hcat(X_nom...)[1:model.nx,:]',
	legend=:topright,color=:red,
    label="",width=2.0,xlabel="time (s)",
    title="Quadrotor",ylabel="state")
plt_x = plot!(t_sim_nom,hcat(z_tvlqr1...)[1:model.nx,:]',color=:black,label="",
    width=1.0)
plt_x = plot!(t_sim_nom,hcat(z_tvlqr2...)[1:model.nx,:]',color=:black,label="",
    width=1.0)
plt_x = plot!(t_sim_nom,hcat(z_tvlqr3...)[1:model.nx,:]',color=:black,label="",
    width=1.0)
plt_x = plot!(t_sim_nom,hcat(z_tvlqr4...)[1:model.nx,:]',color=:black,label="",
    width=1.0)

plt_x = plot(t_nom_sample,hcat(X_nom_sample...)[1:model.nx,:]',
	legend=:topright,color=:red,
    label="",width=2.0,xlabel="time (s)",
    title="Quadrotor",ylabel="state")
plt_x = plot!(t_sim_nom_sample,hcat(z_sample1...)[1:model.nx,:]',color=:black,label="",
    width=1.0)
plt_x = plot!(t_sim_nom_sample,hcat(z_sample2...)[1:model.nx,:]',color=:black,label="",
	width=1.0)
plt_x = plot!(t_sim_nom_sample,hcat(z_sample3...)[1:model.nx,:]',color=:black,label="",
    width=1.0)
plt_x = plot!(t_sim_nom_sample,hcat(z_sample4...)[1:model.nx,:]',color=:black,label="",
    width=1.0)

# plot COM
plot_traj = plot(hcat(X_nom...)[1,:],hcat(X_nom...)[2,:],
	legend=:topright,color=:red,
    label="",width=2.0,xlabel="y",ylabel="z",
    title="Quadrotor")
plot_traj = plot!(hcat(z_tvlqr1...)[1,:],hcat(z_tvlqr1...)[2,:],
	color=:black,
    label="",width=1.0)
plot_traj = plot!(hcat(z_tvlqr2...)[1,:],hcat(z_tvlqr2...)[2,:],
	color=:black,
	label="",width=1.0)
plot_traj = plot!(hcat(z_tvlqr3...)[1,:],hcat(z_tvlqr3...)[2,:],
	color=:black,
    label="",width=1.0)
plot_traj = plot!(hcat(z_tvlqr4...)[1,:],hcat(z_tvlqr4...)[2,:],
	color=:black,
    label="",width=1.0)

plot_traj = plot(hcat(X_nom_sample...)[1,:],hcat(X_nom_sample...)[2,:],
	legend=:topright,color=:red,
    label="",width=2.0,xlabel="y",ylabel="z",
    title="Quadrotor")
plot_traj = plot!(hcat(z_sample1...)[1,:],hcat(z_sample1...)[2,:],
	color=:black,
    label="",width=1.0)
plot_traj = plot!(hcat(z_sample2...)[1,:],hcat(z_sample2...)[2,:],
	color=:black,
    label="",width=1.0)
plot_traj = plot!(hcat(z_sample3...)[1,:],hcat(z_sample3...)[2,:],
	color=:black,
    label="",width=1.0)
plot_traj = plot!(hcat(z_sample4...)[1,:],hcat(z_sample4...)[2,:],
	color=:black,
    label="",width=1.0)

plot(t_nom[1:end-1],hcat(U_nom...)',
	linetype=:steppost,width=2.0)
plot!(t_sim_nom[1:end-1],hcat(u_tvlqr1...)',
	linetype=:steppost,color=:black,width=1.0)

plot(t_nom[1:end-1],hcat(U_nom_sample...)',
	linetype=:steppost,width=2.0)
plot!(t_sim_nom[1:end-1],hcat(u_sample1...)',
	linetype=:steppost,color=:black,width=1.0)


# state tracking
(Jx_tvlqr1 + Jx_tvlqr2 + Jx_tvlqr3 + Jx_tvlqr4)/4.0
(Jx_sample1 + Jx_sample2 + Jx_sample3 + Jx_sample4)/4.0

# control tracking
(Ju_tvlqr1 + Ju_tvlqr2 + Ju_tvlqr3 + Ju_tvlqr4)/4.0
(Ju_sample1 + Ju_sample2 + Ju_sample3 + Ju_sample4)/4.0

# objective value
(J_tvlqr1 + J_tvlqr2 + J_tvlqr3 + J_tvlqr4)/4.0
(J_sample1 + J_sample2 + J_sample3 + J_sample4)/4.0

using PGFPlots
const PGF = PGFPlots

# TO trajectory
p_u1_nom = PGF.Plots.Linear(t_nom[1:end-1],hcat(U_nom...)[1,:],
    mark="",style="const plot, color=cyan, line width=2pt, solid")
p_u2_nom = PGF.Plots.Linear(t_nom[1:end-1],hcat(U_nom...)[2,:],
    mark="",style="const plot, color=cyan, line width=2pt, solid")
p_u3_nom = PGF.Plots.Linear(t_nom[1:end-1],hcat(U_nom...)[3,:],
    mark="",style="const plot, color=cyan, line width=2pt, solid")
p_u4_nom = PGF.Plots.Linear(t_nom[1:end-1],hcat(U_nom...)[4,:],
    mark="",style="const plot, color=cyan, line width=2pt, solid")

p_u1_dpo = PGF.Plots.Linear(t_nom_sample[1:end-1],hcat(U_nom_sample...)[1,:],
    mark="",style="const plot, color=orange, line width=2pt, solid")
p_u2_dpo = PGF.Plots.Linear(t_nom_sample[1:end-1],hcat(U_nom_sample...)[2,:],
    mark="",style="const plot, color=orange, line width=2pt, solid")
p_u3_dpo = PGF.Plots.Linear(t_nom_sample[1:end-1],hcat(U_nom_sample...)[3,:],
    mark="",style="const plot, color=orange, line width=2pt, solid")
p_u4_dpo = PGF.Plots.Linear(t_nom_sample[1:end-1],hcat(U_nom_sample...)[4,:],
    mark="",style="const plot, color=orange, line width=2pt, solid")


a1 = Axis([p_u1_nom;p_u1_dpo
    ],
    xmin=0, ymin=0,
    hideAxis=false,
	ylabel="u1",
	xlabel="time",
	)
a2 = Axis([p_u2_nom;p_u2_dpo
    ],
    xmin=0, ymin=0,
    hideAxis=false,
	ylabel="u2",
	xlabel="time",
	)
a3 = Axis([p_u3_nom;p_u3_dpo
    ],
    xmin=0, ymin=0,
    hideAxis=false,
	ylabel="u3",
	xlabel="time",
	)
a4 = Axis([p_u4_nom;p_u4_dpo
    ],
    xmin=0, ymin=0,
    hideAxis=false,
	ylabel="u4",
	xlabel="time",
	)

# Save to tikz format
dir = joinpath(@__DIR__,"results")
PGF.save(joinpath(dir,"quad_prop_u1.tikz"), a1, include_preamble=false)
PGF.save(joinpath(dir,"quad_prop_u2.tikz"), a2, include_preamble=false)
PGF.save(joinpath(dir,"quad_prop_u3.tikz"), a3, include_preamble=false)
PGF.save(joinpath(dir,"quad_prop_u4.tikz"), a4, include_preamble=false)

# #
# # visualize
visualize!(vis,model,z_tvlqr,Δt=dt_sim_nom)
visualize!(vis,model,z_sample4,Δt=dt_sim_sample)
#
for (k,z) in enumerate([z_tvlqr1,z_tvlqr2,z_tvlqr3,z_tvlqr4])
	i = k
	q_to = z
	for t = 1:3:T_sim
		setobject!(vis["traj_to$t$i"], Sphere(Point3f0(0),
			convert(Float32,0.025)),
			MeshPhongMaterial(color=RGBA(128.0/255.0,128.0/255.0,128.0/255.0,1.0)))
		settransform!(vis["traj_to$t$i"], Translation((q_to[t][1],q_to[t][2],q_to[t][3])))
		setvisible!(vis["traj_to$t$i"],false)
	end
end
q_to_nom = X_nom
for t = 1:T
	setobject!(vis["traj_to_nom$t"], Sphere(Point3f0(0),
		convert(Float32,0.05)),
		MeshPhongMaterial(color=RGBA(0.0,255.0/255.0,255.0/255.0,1.0)))
	settransform!(vis["traj_to_nom$t"], Translation((q_to_nom[t][1],q_to_nom[t][2],q_to_nom[t][3])))
	setvisible!(vis["traj_to_nom$t"],false)
end

for (k,z) in enumerate([z_sample1,z_sample2,z_sample3,z_sample4])
	i = k
	q_dpo = z
	for t = 1:3:T_sim
		setobject!(vis["traj_dpo$t$i"], Sphere(Point3f0(0),
			convert(Float32,0.025)),
			MeshPhongMaterial(color=RGBA(128.0/255.0,128.0/255.0,128.0/255.0,1.0)))
		settransform!(vis["traj_dpo$t$i"], Translation((q_dpo[t][1],q_dpo[t][2],q_dpo[t][3])))
		setvisible!(vis["traj_dpo$t$i"],true)
	end
end

q_dpo_nom = X_nom_sample
for t = 1:T
	setobject!(vis["traj_dpo_nom$t"], Sphere(Point3f0(0),
		convert(Float32,0.05)),
		MeshPhongMaterial(color=RGBA(255.0/255.0,127.0/255.0,0.0,1.0)))
	settransform!(vis["traj_dpo_nom$t"], Translation((q_dpo_nom[t][1],q_dpo_nom[t][2],q_dpo_nom[t][3])))
	setvisible!(vis["traj_dpo_nom$t"],true)
end

obj_path = joinpath(pwd(),"/home/taylor/Research/direct_policy_optimization/dynamics/quadrotor/drone.obj")
mtl_path = joinpath(pwd(),"/home/taylor/Research/direct_policy_optimization/dynamics/quadrotor/drone.mtl")

ctm = ModifiedMeshFileObject(obj_path,mtl_path,scale=1.0)
setobject!(vis["drone2"],ctm)
settransform!(vis["drone2"], compose(Translation(X_nom[1][1:3]),LinearMap(MRP(X_nom[1][4:6]...)*RotX(pi/2.0))))
