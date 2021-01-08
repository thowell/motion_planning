using Plots
include(joinpath(pwd(), "models/visualize.jl"))
vis = Visualizer()
render(vis)

x̄, ū = unpack(z̄_nom, prob_nominal)
x, u = unpack(z, prob_nominal)

t_nominal = range(0, stop = sum([ū[t][end] for t = 1:T-1]), length = T)
plot(t_nominal[1:end-1], hcat(ū...)[1:end-1, :]', linetype = :steppost)
@show sum([ū[t][end] for t = 1:T-1]) # works when 2.72

# COM traj
xx_nom = [x̄[t][1] for t = 1:T]
zz_nom = [x̄[t][2] for t = 1:T]
x̄_dpo, ū_dpo = unpack(z[prob_dpo.prob.idx.nom], prob_dpo.prob.prob.nom)
t_dpo = range(0, stop = sum([ū_dpo[t][end] for t = 1:T-1]), length = T)

xx = [x̄_dpo[t][1] for t = 1:T]
zz = [x̄_dpo[t][2] for t = 1:T]

plot(xx_nom, zz_nom, color = :cyan)
plot!(xx, zz, color = :orange)

plot(t_nominal, hcat(x̄...)', color = :cyan)
plot!(t_dpo, hcat(x̄_dpo...)', color = :orange)


# orientation tracking
plt_x = plot(t_nominal, hcat(x̄...)[3,:],legend=:topright,color=:red,
    label="",width=2.0,xlabel="time (s)",
    title="Rocket",ylabel="state")
plt_x = plot!(t_sim_nom,hcat(z_lqr...)[3,:],color=:black,label="",
    width=1.0)

plt_x = plot(t_dpo,hcat(x̄_dpo...)[3,:],legend=:topright,color=:red,
    label="",width=2.0,xlabel="time (s)",
    title="Rocket",ylabel="state")
plt_x = plot!(t_sim_dpo,hcat(z_dpo...)[3,:],color=:black,label="",
    width=1.0)

visualize!(vis, model_sl, z_lqr, Δt = dt_sim_nom)
visualize!(vis, model_sl, z_dpo, Δt = dt_sim_dpo)

using PGFPlots
const PGF = PGFPlots

p_traj_nom = PGF.Plots.Linear(hcat(x̄...)[1,:],hcat(x̄...)[2,:],
	mark="none",style="color=cyan, line width = 2pt",legendentry="TO")
p_traj_sample = PGF.Plots.Linear(hcat(x...)[1,:],hcat(x...)[2,:],
	mark="none",style="color=orange, line width = 2pt",legendentry="DPO")

a = Axis([p_traj_nom;p_traj_sample],
    axisEqualImage=false,
    hideAxis=false,
	ylabel="z",
	xlabel="y",
	legendStyle="{at={(0.01,0.99)},anchor=north west}")

# Save to tikz format
PGF.save(joinpath(pwd(),
	"examples/direct_policy_optimization/figures/rocket_traj.tikz"), a, include_preamble=false)

# orientation tracking
p_nom_orientation = PGF.Plots.Linear(t_nom,hcat(x̄...)[3,:],
	mark="none",style="color=cyan, line width = 2pt",legendentry="TO")
p_nom_sim_orientation = PGF.Plots.Linear(t_sim_nom,hcat(z_lqr...)[3,:],
	mark="none",style="color=black, line width = 1pt")

p_sample_orientation = PGF.Plots.Linear(t_dpo,hcat(x...)[3,:],
	mark="none",style="color=orange, line width = 2pt",legendentry="DPO")
p_sample_sim_orientation = PGF.Plots.Linear(t_sim_dpo,hcat(z_dpo...)[3,:],
	mark="none",style="color=black, line width = 1pt")

a = Axis([p_nom_orientation;p_nom_sim_orientation],
    axisEqualImage=false,
    hideAxis=false,
	ylabel="orientation",
	xlabel="time",
	legendStyle="{at={(0.5,0.99)},anchor=north}")

# Save to tikz format
dir = joinpath(pwd(),
	"examples/direct_policy_optimization/figures/rocket_orientation_lqr.tikz")
PGF.save(dir, a, include_preamble=false)

a = Axis([p_sample_orientation;p_sample_sim_orientation],
    axisEqualImage=false,
    hideAxis=false,
	ylabel="orientation",
	xlabel="time",
	legendStyle="{at={(0.5,0.99)},anchor=north}")

# Save to tikz format
dir = joinpath(pwd(),
	"examples/direct_policy_optimization/figures/rocket_orientation_dpo.tikz")
PGF.save(dir, a, include_preamble=false)


# # eigen value analysis
# C = Diagonal(ones(model_sl.nx))[1:model_nom.nx,:]
# # nominal
# A_nom, B_nom = nominal_jacobians(model_nom,X_nom,U_nom,H_nom)
# A_nom_cl = [(A_nom[t] - B_nom[t]*K[t]) for t = 1:T-1]
# sv_nom = [norm.(eigen(A_nom_cl[t]).values) for t = 1:T-1]
# plt_nom = plot(hcat(sv_nom...)',xlabel="time step t",ylabel="eigen value norm",
# 	title="TVLQR nominal model",linetype=:steppost,
# 	ylims=(-3,3),labels="")
#
# # slosh nominal
# X_nom_slosh = [[copy(X_nom[t]);0.0;0.0] for t = 1:T]
# A_nom_slosh, B_nom_slosh = nominal_jacobians(model_sl,X_nom_slosh,U_nom,H_nom)
# A_nom_slosh_cl = [(A_nom_slosh[t] - B_nom_slosh[t]*K[t]*C) for t = 1:T-1]
# sv_nom_slosh = [norm.(eigen(A_nom_slosh_cl[t]).values) for t = 1:T-1]
# plt_nom_slosh = plot(hcat(sv_nom_slosh...)',xlabel="time step t",ylabel="eigen value norm",
# 	title="TVLQR slosh model",linetype=:steppost,
# 	ylims=(-3,3),labels="")
#
# # slosh
# A_dpo, B_dpo = nominal_jacobians(model_nom,X_nom_sample,U_nom_sample,H_nom_sample)
# A_dpo_cl = [(A_dpo[t] - B_dpo[t]*Θ_mat[t]) for t = 1:T-1]
# sv_dpo = [norm.(eigen(A_dpo_cl[t]).values) for t = 1:T-1]
# plt_dpo_nom = plot(hcat(sv_dpo...)',xlabel="time step t",ylabel="singular value",
# 	title="DPO nominal model",linetype=:steppost,
# 	ylims=(-1,3),labels="")
#
# X_dpo_slosh = [[copy(X_nom_sample[t]);0.0;0.0] for t = 1:T]
# A_dpo_slosh, B_dpo_slosh = nominal_jacobians(model_sl,X_dpo_slosh,U_nom_sample,H_nom_sample)
# A_dpo_slosh_cl = [(A_dpo_slosh[t] - B_dpo_slosh[t]*Θ_mat[t]*C) for t = 1:T-1]
# sv_dpo_slosh = [norm.(eigen(A_dpo_slosh_cl[t]).values) for t = 1:T-1]
# plt_dpo_slosh = plot(hcat(sv_dpo_slosh...)',xlabel="time step t",ylabel="eigen value norm",
# 	title="DPO slosh model",linetype=:steppost,
# 	ylims=(-1.0,3.0),labels="")
#
# plot(plt_nom,plt_dpo_nom,layout=(2,1))
#
# plot(plt_nom_slosh,plt_dpo_slosh,layout=(2,1))
#
#
# plot(t_nom,hcat(X_nom...)[1:6,:]',color=:cyan,label=["y" "z" "ϕ"])
# plot!(t_nom_sample, hcat(X_nom_sample...)[1:6,:]',color=:orange,label=["y" "z" "ϕ"])


# Ghost plot
vis = Visualizer()
render(vis)
default_background!(vis)

com_traj_nom = hcat([[-1.0 * x̄[t][1]; 0.26; x̄[t][2]] for t = 1:T]...)
pts_nom = collect(eachcol(com_traj_nom))
material_nom = LineBasicMaterial(color = colorant"cyan", linewidth = 5.0)
setobject!(vis["com_traj_nom"], Object(PointCloud(pts_nom), material_nom, "Line"))

com_traj_dpo = hcat([[-1.0 * x[t][1]; 0.265; x[t][2]] for t = 1:T]...)
pts_dpo = collect(eachcol(com_traj_dpo))
material_dpo = LineBasicMaterial(color = colorant"orange", linewidth = 5.0)
setobject!(vis["com_traj_dpo"], Object(PointCloud(pts_dpo), material_dpo, "Line"))

# settransform!(vis["/Cameras/default"], compose(Translation(0.0, 15.0, -1.0),
# 	LinearMap(RotZ(pi / 2.0))))

body = Cylinder(Point3f0(0.0, 0.0, -1.0 * model_sl.l1),
	Point3f0(0.0, 0.0, 3.0 * model_sl.l1),
	convert(Float32, 0.25))
pad = Cylinder(Point3f0(0.0, 0.0, -0.1),
	Point3f0(0.0, 0.0, 0.1),
	convert(Float32, 0.5))
setobject!(vis["pad"], pad,
	MeshPhongMaterial(color = RGBA(220.0 / 255.0, 220.0 / 255.0, 220.0 / 255.0, 1.0)))
setvisible!(vis["pad"], false)

u_norm = [u[t][1:2] ./ 20.0 for t = 1:T-1]

t = 1
# setobject!(vis["rocket1"], body,
# 	MeshPhongMaterial(color = RGBA(0.0, 0.0, 0.0, 1.0)))
# settransform!(vis["rocket1"],
# 	compose(Translation(-1.0 * x[t][1], 0.0, x[t][2]),
# 	LinearMap(RotY(x[t][3]))))
setobject!(vis["thrust1"],
	Cylinder(Point3f0(0.0, 0.0, 0.0),
	Point3f0(0.0, 0.0, norm(u_norm[t])),
	convert(Float32, 0.15)),
	MeshPhongMaterial(color = RGBA(1.0, 0.0, 0.0, 1.0)))
p = k_thruster(model_sl, x[t])
settransform!(vis["thrust1"], cable_transform([-1.0 * p[1]; 0.0; p[2]],
 	[-1.0 * p[1] + u_norm[t][1]; 0.0; p[2] - u_norm[t][2]]))
t = 7
# setobject!(vis["rocket2"], body,
# 	MeshPhongMaterial(color = RGBA(0.0, 0.0, 0.0, 1.0)))
# settransform!(vis["rocket2"],
# 	compose(Translation(-1.0 * x[t][1], 0.0, x[t][2]),
# 	LinearMap(RotY(x[t][3]))))
setobject!(vis["thrust2"],
	Cylinder(Point3f0(0.0, 0.0, 0.0),
	Point3f0(0.0, 0.0, norm(u_norm[t])),
	convert(Float32, 0.15)),
	MeshPhongMaterial(color = RGBA(1.0, 0.0, 0.0, 1.0)))
p = k_thruster(model_sl, x[t])
settransform!(vis["thrust2"], cable_transform([-1.0 * p[1]; 0.0; p[2]],
 	[-1.0 * p[1] + u_norm[t][1]; 0.0; p[2] - u_norm[t][2]]))
t = 15
# setobject!(vis["rocket3"], body,
# 	MeshPhongMaterial(color = RGBA(0.0, 0.0, 0.0, 1.0)))
# settransform!(vis["rocket3"],
# 	compose(Translation(-1.0 * x[t][1], 0.0, x[t][2]),
# 	LinearMap(RotY(x[t][3]))))
setobject!(vis["thrust3"],
	Cylinder(Point3f0(0.0, 0.0, 0.0),
	Point3f0(0.0, 0.0, norm(u_norm[t])),
	convert(Float32, 0.15)),
	MeshPhongMaterial(color = RGBA(1.0, 0.0, 0.0, 1.0)))
p = k_thruster(model_sl, x[t])
settransform!(vis["thrust3"], cable_transform([-1.0 * p[1]; 0.0; p[2]],
 	[-1.0 * p[1] + u_norm[t][1]; 0.0; p[2] - u_norm[t][2]]))
t = 25
# setobject!(vis["rocket4"], body,
# 	MeshPhongMaterial(color = RGBA(0.0, 0.0, 0.0, 1.0)))
# settransform!(vis["rocket4"],
# 	compose(Translation(-1.0 * x[t][1], 0.0, x[t][2]),
# 	LinearMap(RotY(x[t][3]))))
setobject!(vis["thrust4"],
	Cylinder(Point3f0(0.0, 0.0, 0.0),
	Point3f0(0.0, 0.0, norm(u_norm[t])),
	convert(Float32, 0.15)),
	MeshPhongMaterial(color = RGBA(1.0, 0.0, 0.0, 1.0)))
p = k_thruster(model_sl, x[t])
settransform!(vis["thrust4"], cable_transform([-1.0 * p[1]; 0.0; p[2]],
 	[-1.0 * p[1] + u_norm[t][1]; 0.0; p[2] - u_norm[t][2]]))
t = T
setobject!(vis["rocket5"], body,
	MeshPhongMaterial(color = RGBA(0.0, 0.0, 0.0, 1.0)))
settransform!(vis["rocket5"],
	compose(Translation(-1.0 * x[t][1], 0.0, x[t][2]),
	LinearMap(RotY(x[t][3]))))

##
vis = Visualizer()
render(vis)
default_background!(vis)

settransform!(vis["/Cameras/default"], compose(Translation(0.0, 25.0, -1.0),
	LinearMap(RotZ(pi / 2.0))))

y_shift = 2.35
obj_rocket = joinpath(pwd(), "models/rocket/space_x_booster.obj")
mtl_rocket = joinpath(pwd(), "models/rocket/space_x_booster.mtl")
q = deepcopy(x)
rkt_offset = [3.9,-6.35,0.2]
ctm = ModifiedMeshFileObject(obj_rocket,mtl_rocket,scale=1.0)
t = 1
setobject!(vis["_rocket1"],ctm)
settransform!(vis["_rocket1"],
	compose(Translation(([-q[t][1]-0.5;y_shift;q[t][2] + 2.05] + rkt_offset)...),
	LinearMap(RotY(1.0*q[t][3])*RotZ(pi)*RotX(pi/2.0))))
setvisible!(vis["_rocket1"],true)

t = 7
setobject!(vis["_rocket2"],ctm)
settransform!(vis["_rocket2"],
	compose(Translation(([-q[t][1]+ 0.2;y_shift;q[t][2]-0.6] + rkt_offset)...),
		LinearMap(RotY(1.0*q[t][3])*RotZ(pi)*RotX(pi/2.0))))
setvisible!(vis["_rocket2"],true)
t = 15
setobject!(vis["_rocket3"],ctm)
settransform!(vis["_rocket3"],
	compose(Translation(([-q[t][1]+0.2;y_shift;q[t][2]-0.35] + rkt_offset)...),
	LinearMap(RotY(1.0*q[t][3])*RotZ(pi)*RotX(pi/2.0))))
setvisible!(vis["_rocket3"],true)

t = 29
setobject!(vis["_rocket4"],ctm)
settransform!(vis["_rocket4"],
	compose(Translation(([-q[t][1]-1.0;y_shift;q[t][2]+0.4] + rkt_offset)...),
	LinearMap(RotY(1.0*q[t][3]+ 0.1)*RotZ(pi)*RotX(pi/2.0))))
setvisible!(vis["_rocket4"],true)

t = T
setobject!(vis["_rocket5"],ctm)
settransform!(vis["_rocket5"],
	compose(Translation(([-q[t][1]+0.1;y_shift;q[t][2]-0.7] + rkt_offset)...),
	LinearMap(RotY(1.0*q[t][3])*RotZ(pi)*RotX(pi/2.0))))
setvisible!(vis["_rocket5"],true)

obj_platform = "/home/taylor/Research/direct_policy_optimization/dynamics/rocket/space_x_platform.obj"
mtl_platform = "/home/taylor/Research/direct_policy_optimization/dynamics/rocket/space_x_platform.mtl"

ctm_platform = ModifiedMeshFileObject(obj_platform,mtl_platform,scale=1.0)
setobject!(vis["platform"],ctm_platform)
settransform!(vis["platform"], compose(Translation(0.0,0.0,-0.85),LinearMap(RotZ(pi)*RotX(pi/2))))

setvisible!(vis["rocket1"],true)
setvisible!(vis["rocket2"],true)
setvisible!(vis["rocket3"],true)
setvisible!(vis["rocket4"],true)
setvisible!(vis["rocket5"],true)
