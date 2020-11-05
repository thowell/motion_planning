include(joinpath(pwd(), "src/direct_policy_optimization/dpo.jl"))
include(joinpath(@__DIR__, "car_obstacles.jl"))

# DPO
N = 2 * model.n
D = 2 * model.d

α = 1.0
β = 1.0 / (N + D)
γ = 0.5
δ = 1.0

x1 = resample(zeros(model.n), Diagonal(ones(model.n)), 1.0)

# mean problem
prob_mean = trajectory_optimization(
				model,
				EmptyObjective(),
				dynamics = false,
				T)

# sample problems
prob_sample = [trajectory_optimization(
				model,
				EmptyObjective(),
				dynamics = false,
				T,
				xl = state_bounds(model, T, x1 = x1[i])[1],
				xu = state_bounds(model, T, x1 = x1[i])[2]
				) for i = 1:N]

# sample objective
Q = [Diagonal(ones(model.n)) for t = 1:T]
R = [Diagonal(ones(model.m)) for t = 1:T-1]

obj_sample = sample_objective(Q, R)
policy = linear_feedback(model.n, model.m)
dist = disturbances([Diagonal(δ * ones(model.d)) for t = 1:T-1])
sample = sample_params(α, β, γ, T)

prob_dpo = dpo_problem(
	prob_nom, prob_mean, prob_sample,
	obj_sample,
	policy,
	dist,
	sample)

z0 = ones(prob_dpo.num_var)

# Solve
z_sol = solve(prob_dpo, copy(z0),
	tol = 1.0e-8, c_tol = 1.0e-8,
	mapl = 0)

# TVLQR policy
A, B = get_dynamics(model)
K = tvlqr(
	[A for t = 1:T-1],
	[B for t = 1:T-1],
	[Q[t] for t = 1:T],
	[R[t] for t = 1:T-1])



# Sample

# LQR cost
Q_lqr = [(t < T ? Diagonal([10.0; 10.0; 1.0])
         : Diagonal(100.0 * ones(model.n))) for t = 1:T]
R_lqr = [Diagonal(1.0e-1 * ones(model.m)) for t = 1:T-1]


N = 2*model.nx
models = [model for i = 1:N]
β = 1.0
w = 1.0e-3*[1.0;1.0;0.1]
γ = 1.0
x1_sample = resample([x1 for i = 1:N],β=0.1,w=[1.0;1.0;0.1])

xl_traj_sample = [[-Inf*ones(model.nx) for t = 1:T] for i = 1:N]
xu_traj_sample = [[Inf*ones(model.nx) for t = 1:T] for i = 1:N]

for i = 1:N
    xl_traj_sample[i][1] = x1_sample[i]
    xu_traj_sample[i][1] = x1_sample[i]
end

K = TVLQR_gains(model,X_nom,U_nom,H_nom,Q_lqr,R_lqr)

prob_sample = init_sample_problem(prob,models,Q_lqr,R_lqr,H_lqr,
    xl=xl_traj_sample,
    xu=xu_traj_sample,
    β=β,w=w,γ=γ)


prob_sample_moi = init_MOI_Problem(prob_sample)

Z0_sample = pack(X_nom,U_nom,H_nom[1],K,prob_sample)

# Solve
Z_sample_sol = solve(prob_sample_moi,copy(Z0_sample),nlp=:SNOPT7,time_limit=60)

using JLD
@save joinpath(pwd(),"examples/trajectories/","car_obstacles.jld") Z_sample_sol
# @load joinpath(pwd(),"examples/trajectories/","car_obstacles.jld") Z_sample_sol

# Unpack solutions
X_nom_sample, U_nom_sample, H_nom_sample, X_sample, U_sample, H_sample = unpack(Z_sample_sol,prob_sample)

# Plots results

# Position trajectory
x_nom_pos = [X_nom[t][1] for t = 1:T]
y_nom_pos = [X_nom[t][2] for t = 1:T]
pts = Plots.partialcircle(0,2π,100,r)
cx,cy = Plots.unzip(pts)
cx1 = [_cx + xc1 for _cx in cx]
cy1 = [_cy + yc1 for _cy in cy]
cx2 = [_cx + xc2 for _cx in cx]
cy2 = [_cy + yc2 for _cy in cy]
cx3 = [_cx + xc3 for _cx in cx]
cy3 = [_cy + yc3 for _cy in cy]
cx4 = [_cx + xc4 for _cx in cx]
cy4 = [_cy + yc4 for _cy in cy]
# cx5 = [_cx + xc5 for _cx in cx]
# cy5 = [_cy + yc5 for _cy in cy]

plt = plot(Shape(cx1,cy1),color=:red,label="",linecolor=:red)
plt = plot!(Shape(cx2,cy2),color=:red,label="",linecolor=:red)
plt = plot!(Shape(cx3,cy3),color=:red,label="",linecolor=:red)
plt = plot!(Shape(cx4,cy4),color=:red,label="",linecolor=:red)
# plt = plot(Shape(cx5,cy5),color=:red,label="",linecolor=:red)

for i = 1:N
    x_sample_pos = [X_sample[i][t][1] for t = 1:T]
    y_sample_pos = [X_sample[i][t][2] for t = 1:T]
    plt = plot!(x_sample_pos,y_sample_pos,aspect_ratio=:equal,
        color=:cyan,label= i != 1 ? "" : "sample")
end
plt = plot!(x_nom_pos,y_nom_pos,aspect_ratio=:equal,xlabel="x",ylabel="y",width=4.0,label="TO",color=:purple,legend=:topleft)
x_sample_pos = [X_nom_sample[t][1] for t = 1:T]
y_sample_pos = [X_nom_sample[t][2] for t = 1:T]
plt = plot!(x_sample_pos,y_sample_pos,aspect_ratio=:equal,width=4.0,label="DPO",color=:orange,legend=:bottomright)

savefig(plt,joinpath(@__DIR__,"results/car_trajectory.png"))

# Control
plt = plot(Array(hcat(U_nom...))',color=:purple,width=2.0,
    title="car",xlabel="time (s)",ylabel="control",label=["v (nominal)" "ω (nominal)"],
    legend=:bottom,linetype=:steppost)
plt = plot!(Array(hcat(U_nom_sample...))',color=:orange,
    width=2.0,label=["v (sample)" "ω (sample)"],linetype=:steppost)
savefig(plt,joinpath(@__DIR__,"results/car_control.png"))

# Samples

# State samples
plt1 = plot(title="Sample states",legend=:bottom,xlabel="time (s)");
for i = 1:N
    plt1 = plot!(hcat(X_sample[i]...)',label="");
end
plt1 = plot!(hcat(X_nom_sample...)',color=:red,width=2.0,
    label=["nominal" "" ""])
display(plt1)
savefig(plt1,joinpath(@__DIR__,"results/car_sample_states.png"))

# Control samples
plt2 = plot(title="Sample controls",xlabel="time (s)",legend=:bottom);
for i = 1:N
    # t_sample = zeros(T)
    # for t = 2:T
    #     t_sample[t] = t_sample[t-1] + H_nom_sample[t-1]
    # end
    plt2 = plot!(hcat(U_sample[i]...)',label="",
        linetype=:steppost);
end
plt2 = plot!(hcat(U_nom_sample...)',color=:red,width=2.0,
    label=["nominal" ""],linetype=:steppost)
display(plt2)
savefig(plt2,joinpath(@__DIR__,"results/car_sample_controls.png"))

using PGFPlots
const PGF = PGFPlots

# TO trajectory
p_nom = PGF.Plots.Linear(hcat(X_nom...)[1,:],hcat(X_nom...)[2,:],
    mark="",style="color=cyan, line width=3pt, solid",legendentry="TO")

# DPO trajectory
p_dpo = PGF.Plots.Linear(hcat(X_nom_sample...)[1,:],hcat(X_nom_sample...)[2,:],
    mark="",style="color=orange, line width=3pt, solid",legendentry="DPO")

# DPO trajectory
p_sample = [PGF.Plots.Linear(hcat(X_sample[i]...)[1,:],hcat(X_sample[i]...)[2,:],
    mark="",style="color=gray, line width=1pt, solid") for i = 1:N]
p_sample[6].legendentry="sample"

# obstacles
p_circle = [PGF.Plots.Circle(circle..., style="color=black,fill=black") for circle in circles]

a = Axis([p_circle;
    p_sample[1];
    p_sample[2];
    p_sample[3];
    p_sample[4];
    p_sample[5];
    p_sample[6];
    p_nom;
    p_dpo
    ],
    xmin=-0.4, ymin=-0.1, xmax=1.4, ymax=1.1,
    axisEqualImage=true,
    hideAxis=false,
	ylabel="y",
	xlabel="x",
	legendStyle="{at={(0.01,0.99)},anchor=north west}",
	)

# Save to tikz format
dir = joinpath(@__DIR__,"results")
PGF.save(joinpath(dir,"car_obstacles.tikz"), a, include_preamble=false)


# Animation
include(joinpath(pwd(),"dynamics/visualize.jl"))

vis = Visualizer()
open(vis)
visualize!(vis,model,[X_nom...,[X_nom[end] for t = 1:T]...],Δt=H_nom_sample[1])
# settransform!(vis["/Cameras/default"], compose(LinearMap(RotY(-pi/2.0)*RotX(pi/2)),Translation(-1, 0,0)))

for i = 1:4
    cyl = Cylinder(Point3f0(xc[i],yc[i],0),Point3f0(xc[i],yc[i],0.1),convert(Float32,0.035))
    setobject!(vis["cyl$i"],cyl,MeshPhongMaterial(color=RGBA(1,0,0,1.0)))
end

q_to = deepcopy(X_nom)
for t = 1:T
	setobject!(vis["traj_to$t"], HyperSphere(Point3f0(0),
		convert(Float32,0.075)),
		MeshPhongMaterial(color=RGBA(0.0,255.0/255.0,255.0/255.0,0.75)))
	settransform!(vis["traj_to$t"], Translation((q_to[t][1],q_to[t][2],-0.1)))
	setvisible!(vis["traj_to$t"],true)
end

q_dpo = deepcopy(X_nom_sample)
for t = 1:T
	setobject!(vis["traj_dpo$t"], HyperSphere(Point3f0(0),
		convert(Float32,0.075)),
		MeshPhongMaterial(color=RGBA(255.0/255.0,127.0/255.0,0.0,0.75)))
	settransform!(vis["traj_dpo$t"], Translation((q_dpo[t][1],q_dpo[t][2],-0.05)))
	setvisible!(vis["traj_dpo$t"],false)
end

q = X_nom_sample
obj_path = joinpath(pwd(),"/home/taylor/Research/direct_policy_optimization/dynamics/cybertruck/cybertruck.obj")
mtl_path = joinpath(pwd(),"/home/taylor/Research/direct_policy_optimization/dynamics/cybertruck/cybertruck.mtl")

ctm = ModifiedMeshFileObject(obj_path,mtl_path,scale=0.05)
t = 1
setobject!(vis["ct1"],ctm)
settransform!(vis["ct1"], compose(Translation([q[t][1];q[t][2];0.0]),LinearMap(RotZ(q[t][3]+pi)*RotX(pi/2.0))))
setvisible!(vis["ct1"],false)
t = 1
setobject!(vis["ct2"],ctm)
settransform!(vis["ct2"], compose(Translation([q[t][1];q[t][2];0.0]),LinearMap(RotZ(q[t][3]+pi)*RotX(pi/2.0))))
setvisible!(vis["ct2"],false)
t = 20#17#20
setobject!(vis["ct3"],ctm)
settransform!(vis["ct3"], compose(Translation([q[t][1];q[t][2];0.0]),LinearMap(RotZ(q[t][3]+pi)*RotX(pi/2.0))))
setvisible!(vis["ct3"],false)

t = 34#36#34
setobject!(vis["ct4"],ctm)
settransform!(vis["ct4"], compose(Translation([q[t][1];q[t][2];0.0]),LinearMap(RotZ(q[t][3]+pi)*RotX(pi/2.0))))
setvisible!(vis["ct4"],false)
t = T
setobject!(vis["ct5"],ctm)
settransform!(vis["ct5"], compose(Translation([q[t][1];q[t][2];0.0]),LinearMap(RotZ(q[t][3]+pi)*RotX(pi/2.0))))
setvisible!(vis["ct5"],false)

# setvisible!(vis["/Background"], true)
# setprop!(vis["/Background"], "top_color", [135,206,250])
# setprop!(vis["/Background"], "bottom_color", [135,206,250])
# vis["/Background"].core
