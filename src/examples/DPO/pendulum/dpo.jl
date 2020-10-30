
# Samples


# TVLQR cost
Q_lqr = [t < T ? Diagonal([10.0;10.0]) : Diagonal([100.0; 100.0]) for t = 1:T]
R_lqr = [Diagonal(0.1*ones(model.nu)) for t = 1:T-1]
H_lqr = [10.0 for t = 1:T-1]

N = 2*model.nx
models = [model for i = 1:N]
β = 1.0
w = 2.0e-3*ones(model.nx)
γ = 1.0

α = 1.0e-3
x11 = α*[1.0; 0.0]
x12 = α*[-1.0; 0.0]
x13 = α*[0.0; 1.0]
x14 = α*[0.0; -1.0]
x1_sample = resample([x11,x12,x13,x14],β=β,w=w)

xl_traj_sample = [[-Inf*ones(model.nx) for t = 1:T] for i = 1:N]
xu_traj_sample = [[Inf*ones(model.nx) for t = 1:T] for i = 1:N]

for i = 1:N
    xl_traj_sample[i][1] = x1_sample[i]
    xu_traj_sample[i][1] = x1_sample[i]
end

K = TVLQR_gains(model,X_nominal,U_nominal,H_nominal,Q_lqr,R_lqr)

prob_sample = init_sample_problem(prob,models,
    Q_lqr,R_lqr,H_lqr,
    β=β,w=w,γ=γ,
    xl=xl_traj_sample,
    xu=xu_traj_sample)

prob_sample_moi = init_MOI_Problem(prob_sample)

Z0_sample = pack(X_nominal,U_nominal,H_nominal[1],K,prob_sample)

# Solve
#NOTE: Ipopt finds different solution compared to SNOPT
Z_sample_sol = solve(prob_sample_moi,Z0_sample,nlp=:SNOPT7,time_limit=30)

# save/load solution
using JLD
@save joinpath(pwd(),"examples/trajectories/","pendulum_min_time.jld") Z_sample_sol
# @load joinpath(pwd(),"examples/trajectories/","pendulum_min_time.jld") Z_sample_sol

# Unpack solution
X_nom_sample, U_nom_sample, H_nom_sample, X_sample, U_sample, H_sample = unpack(Z_sample_sol,prob_sample)

# Plot results

# Time
t_nominal = zeros(T)
t_sample = zeros(T)
for t = 2:T
    t_nominal[t] = t_nominal[t-1] + H_nominal[t-1]
    t_sample[t] = t_sample[t-1] + H_nom_sample[t-1]
end

tf_nom = sum(H_nominal)
tf_sample = sum(H_nom_sample)
display("time (nominal): $(tf_nom)s")
display("time (sample nominal): $(tf_sample)s")

# Control
plt = plot(t_nominal[1:T-1],Array(hcat(U_nominal...))',
    color=:purple,width=2.0,title="Pendulum",xlabel="time (s)",
    ylabel="control",label="nominal",linelegend=:topleft,
    linetype=:steppost)
plt = plot!(t_sample[1:T-1],Array(hcat(U_nom_sample...))',
    color=:orange,width=2.0,label="sample",linetype=:steppost)
savefig(plt,joinpath(@__DIR__,"results/pendulum_control_noise.png"))

# States
plt = plot(t_nominal,hcat(X_nominal...)[1,:],
    color=:purple,width=2.0,xlabel="time (s)",ylabel="state",
    label="θ (nominal)",title="Pendulum",legend=:topleft)
plt = plot!(t_nominal,hcat(X_nominal...)[2,:],color=:purple,width=2.0,label="x (nominal)")
plt = plot!(t_sample,hcat(X_nom_sample...)[1,:],color=:orange,width=2.0,label="θ (sample)")
plt = plot!(t_sample,hcat(X_nom_sample...)[2,:],color=:orange,width=2.0,label="x (sample)")
savefig(plt,joinpath(@__DIR__,"results/pendulum_state_noise.png"))

plot(hcat(X_nominal...)',width=4.0,color=:yellowgreen,
	grid=false,label="")
# State samples
plt1 = plot(t_sample,hcat(X_nom_sample...)[1:2,:]',
	color=:red,width=3.0,title="",
    label="",grid=false);
for i = 1:N
    t_sample = zeros(T)
    for t = 2:T
        t_sample[t] = t_sample[t-1] + H_sample[i][t-1]
    end
    plt1 = plot!(t_sample,hcat(X_sample[i]...)[1:2,:]',label="",
		width=2.0);
end
plt1 = plot!(t_sample,hcat(X_nom_sample...)[1:2,:]',
	color=:red,width=3.0,title="",
    label="",grid=false)
plt1

plt2 = plot(t_sample,hcat(X_nom_sample...)[2,:],color=:red,width=2.0,label="");
for i = 1:N
    t_sample = zeros(T)
    for t = 2:T
        t_sample[t] = t_sample[t-1] + H_sample[i][t-1]
    end
    plt2 = plot!(t_sample,hcat(X_sample[i]...)[2,:],label="");
end
plt12 = plot(plt1,plt2,layout=(2,1),title=["θ" "x"],xlabel="time (s)")
savefig(plt,joinpath(@__DIR__,"results/pendulum_sample_state.png"))

# Control samples
plt3 = plot(t_sample[1:end-1],hcat(U_nom_sample...)[1,:],color=:red,width=3.0,
    title="Control",label="",xlabel="time (s)",
	grid=false);
for i = 1:N
    t_sample = zeros(T)
    for t = 2:T
        t_sample[t] = t_sample[t-1] + H_sample[i][t-1]
    end
    plt3 = plot!(t_sample[1:end-1],hcat(U_sample[i]...)[1,:],label="",
	width=2.0);
end
plt3 = plot!(t_sample[1:end-1],hcat(U_nom_sample...)[1,:],color=:red,width=3.0,
    title="Control",label="",xlabel="time (s)")
display(plt3)
savefig(plt,joinpath(@__DIR__,"results/pendulum_sample_control.png"))

using PGFPlots
const PGF = PGFPlots

# nominal trajectory
psx_nom = PGF.Plots.Linear(t_nominal,hcat(X_nominal...)[1,:],mark="",
	style="color=cyan, line width=3pt",legendentry="pos. (TO)")
psθ_nom = PGF.Plots.Linear(t_nominal,hcat(X_nominal...)[2,:],mark="",
	style="color=cyan, line width=3pt, densely dashed",legendentry="ang. (TO)")

# DPO trajectory
psx_dpo = PGF.Plots.Linear(t_sample,hcat(X_nom_sample...)[1,:],mark="",
	style="color=orange, line width=3pt",legendentry="pos. (DPO)")
psθ_dpo = PGF.Plots.Linear(t_sample,hcat(X_nom_sample...)[2,:],mark="",
	style="color=orange, line width=3pt, densely dashed",legendentry="ang. (DPO)")

a = Axis([psx_nom;psθ_nom;psx_dpo;psθ_dpo],
    xmin=0., ymin=-3, xmax=max(sum(H_nom_sample),sum(H_nominal)), ymax=7.0,
    axisEqualImage=false,
    hideAxis=false,
	ylabel="state",
	xlabel="time",
	legendStyle="{at={(0.01,0.99)},anchor=north west}",
	)

# Save to tikz format
dir = joinpath(@__DIR__,"results")
PGF.save(joinpath(dir,"minimum_time_pendulum_state.tikz"), a, include_preamble=false)

# nominal trajectory
psu_nom = PGF.Plots.Linear(t_nominal[1:end-1],hcat(U_nominal...)[1,:],mark="",
	style="const plot,color=purple, line width=3pt",legendentry="TO")

# DPO trajectory
psu_dpo = PGF.Plots.Linear(t_sample[1:end-1],hcat(U_nom_sample...)[1,:],mark="",
	style="const plot, color=orange, line width=3pt",legendentry="DPO")

a = Axis([psu_nom;psu_dpo],
    xmin=0., ymin=-3.1, xmax=max(sum(H_nom_sample[1:end-1]),sum(H_nominal[1:end-1])), ymax=3.1,
    axisEqualImage=false,
    hideAxis=false,
	ylabel="control",
	xlabel="time",
	legendStyle="{at={(0.01,0.99)},anchor=north west}",
	)

# Save to tikz format
dir = joinpath(@__DIR__,"results")
PGF.save(joinpath(dir,"minimum_time_pendulum_control.tikz"), a, include_preamble=false)

# Animation
function visualize!(vis,model::Pendulum,q;
        color=RGBA(0,0,0,1.0),r=0.1,Δt=0.1)

	i = 1
    l1 = Cylinder(Point3f0(0,0,0),Point3f0(0,0,model.lc),convert(Float32,0.025))
    setobject!(vis["link$i"],l1,MeshPhongMaterial(color=RGBA(0,0,0,1)))

    setobject!(vis["ee$i"], HyperSphere(Point3f0(0),
        convert(Float32,0.05)),
        MeshPhongMaterial(color=color))

    anim = MeshCat.Animation(convert(Int,floor(1/Δt)))

    for t = 1:length(q)

        MeshCat.atframe(anim,t) do
            p_ee = [kinematics(model,q[t])[1], 0.0, kinematics(model,q[t])[2]]
            settransform!(vis["link$i"], cable_transform(zeros(3),p_ee))
            settransform!(vis["ee$i"], Translation(p_ee))
        end
    end
    # settransform!(vis["/Cameras/default"], compose(Translation(-1, -1, 0),LinearMap(RotZ(pi/2))))
    MeshCat.setanimation!(vis,anim)
end

include(joinpath(pwd(),"dynamics/visualize.jl"))

vis = Visualizer()
open(vis)
visualize!(vis,model,[X_nom_sample...,[X_nom_sample[end] for t = 1:T]...],
	Δt=H_nom_sample[1],color=RGBA(255.0/255.0,0,0,1.0))##H_nominal[1])

X_nom_vis = deepcopy(X_nom)
t_cur
