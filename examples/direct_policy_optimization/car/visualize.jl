using Plots
# include(joinpath(@__DIR__, "car_obstacles.jl"))

X̄, Ū = unpack(Z̄, prob)

# Position trajectory
Px = [X̄[t][1] for t = 1:T]
Py = [X̄[t][2] for t = 1:T]

pts = Plots.partialcircle(0.0, 2.0 * π, 100, 0.1)
cx, cy = Plots.unzip(pts)
cx1 = [_cx + circles[1][1] for _cx in cx]
cy1 = [_cy + circles[1][2] for _cy in cy]
cx2 = [_cx + circles[2][1] for _cx in cx]
cy2 = [_cy + circles[2][2] for _cy in cy]
cx3 = [_cx + circles[3][1] for _cx in cx]
cy3 = [_cy + circles[3][2] for _cy in cy]
cx4 = [_cx + circles[4][1] for _cx in cx]
cy4 = [_cy + circles[4][2] for _cy in cy]

plt = plot(Shape(cx1, cy1), color = :red, label = "", linecolor = :red)
plt = plot!(Shape(cx2, cy2), color = :red, label = "", linecolor = :red)
plt = plot!(Shape(cx3, cy3), color = :red, label = "", linecolor = :red)
plt = plot!(Shape(cx4, cy4), color = :red, label = "", linecolor = :red)
plt = plot!(Px, Py, aspect_ratio = :equal, xlabel = "x", ylabel = "y",
    width = 4.0, label = "TO", color = :purple, legend = :topleft)

#DPO
X̄_dpo, Ū_dpo = unpack(Z[prob_dpo.prob.idx.nom], prob_dpo.prob.prob.nom)
X̄_dpo_mean, Ū_dpo_mean = unpack(Z[prob_dpo.prob.idx.mean], prob_dpo.prob.prob.mean)

X̄[end]
X̄_dpo[end]
X̄_dpo_mean[end]
X_sample = []
U_sample = []
for i = 1:N
	x, u = unpack(Z[prob_dpo.prob.idx.sample[i]],
		prob_dpo.prob.prob.sample[i])
	push!(X_sample, x)
	push!(U_sample, u)
end

# Position trajectory
Px_dpo = [X̄_dpo[t][1] for t = 1:T]
Py_dpo = [X̄_dpo[t][2] for t = 1:T]

Px_dpo_mean = [X̄_dpo_mean[t][1] for t = 1:T]
Py_dpo_mean = [X̄_dpo_mean[t][2] for t = 1:T]
plt = plot!(Px_dpo_mean, Py_dpo_mean, width = 2.0, label = "DPO (mean)", color = :green)

Px_sample = []
Py_sample = []
for i = 1:N
	px = [X_sample[i][t][1] for t = 1:T]
	py = [X_sample[i][t][2] for t = 1:T]
	push!(Px_sample, px)
	push!(Py_sample, py)
end

plt = plot!(Px_dpo, Py_dpo, width = 4.0, label = "DPO (nominal)", color = :orange)
for i = 1:N
	plt = plot!(Px_sample[i], Py_sample[i], width = 1.0, label = "", color = :grey)
end
display(plt)

# using PGFPlots
# const PGF = PGFPlots
#
# # TO trajectory
# p_nom = PGF.Plots.Linear(hcat(X̄...)[1,:], hcat(X̄...)[2,:],
#     mark="", style="color=cyan, line width=3pt, solid", legendentry = "TO")
#
# # DPO trajectory
# p_dpo = PGF.Plots.Linear(hcat(X̄_dpo...)[1,:], hcat(X̄_dpo...)[2,:],
#     mark="", style = "color=orange, line width=3pt, solid",legendentry = "DPO")
#
# # Obstacles
# p_circle = [PGF.Plots.Circle(circle..., style = "color=black,fill=black") for circle in circles]
#
# a = Axis([p_circle;
#     p_nom;
#     p_dpo
#     ],
#     xmin = -0.4, ymin = -0.1, xmax = 1.4, ymax = 1.1,
#     axisEqualImage = true,
#     hideAxis = false,
# 	ylabel = "y",
# 	xlabel = "x",
# 	legendStyle = "{at={(0.01,0.99)},anchor=north west}",
# 	)
#
# # Save to tikz format
# dir = joinpath(@__DIR__, "results")
# PGF.save(joinpath(dir, "car_obstacles.tikz"), a, include_preamble = false)

# # Animation
# include(joinpath(pwd(),"dynamics/visualize.jl"))
#
# vis = Visualizer()
# open(vis)
# visualize!(vis,model,[X_nom...,[X_nom[end] for t = 1:T]...],Δt=H_nom_sample[1])
# # settransform!(vis["/Cameras/default"], compose(LinearMap(RotY(-pi/2.0)*RotX(pi/2)),Translation(-1, 0,0)))
#
# for i = 1:4
#     cyl = Cylinder(Point3f0(xc[i],yc[i],0),Point3f0(xc[i],yc[i],0.1),convert(Float32,0.035))
#     setobject!(vis["cyl$i"],cyl,MeshPhongMaterial(color=RGBA(1,0,0,1.0)))
# end
#
# q_to = deepcopy(X_nom)
# for t = 1:T
# 	setobject!(vis["traj_to$t"], Sphere(Point3f0(0),
# 		convert(Float32,0.075)),
# 		MeshPhongMaterial(color=RGBA(0.0,255.0/255.0,255.0/255.0,0.75)))
# 	settransform!(vis["traj_to$t"], Translation((q_to[t][1],q_to[t][2],-0.1)))
# 	setvisible!(vis["traj_to$t"],true)
# end
#
# q_dpo = deepcopy(X_nom_sample)
# for t = 1:T
# 	setobject!(vis["traj_dpo$t"], Sphere(Point3f0(0),
# 		convert(Float32,0.075)),
# 		MeshPhongMaterial(color=RGBA(255.0/255.0,127.0/255.0,0.0,0.75)))
# 	settransform!(vis["traj_dpo$t"], Translation((q_dpo[t][1],q_dpo[t][2],-0.05)))
# 	setvisible!(vis["traj_dpo$t"],false)
# end
#
# q = X_nom_sample
# obj_path = joinpath(pwd(),"/home/taylor/Research/direct_policy_optimization/dynamics/cybertruck/cybertruck.obj")
# mtl_path = joinpath(pwd(),"/home/taylor/Research/direct_policy_optimization/dynamics/cybertruck/cybertruck.mtl")
#
# ctm = ModifiedMeshFileObject(obj_path,mtl_path,scale=0.05)
# t = 1
# setobject!(vis["ct1"],ctm)
# settransform!(vis["ct1"], compose(Translation([q[t][1];q[t][2];0.0]),LinearMap(RotZ(q[t][3]+pi)*RotX(pi/2.0))))
# setvisible!(vis["ct1"],false)
# t = 1
# setobject!(vis["ct2"],ctm)
# settransform!(vis["ct2"], compose(Translation([q[t][1];q[t][2];0.0]),LinearMap(RotZ(q[t][3]+pi)*RotX(pi/2.0))))
# setvisible!(vis["ct2"],false)
# t = 20#17#20
# setobject!(vis["ct3"],ctm)
# settransform!(vis["ct3"], compose(Translation([q[t][1];q[t][2];0.0]),LinearMap(RotZ(q[t][3]+pi)*RotX(pi/2.0))))
# setvisible!(vis["ct3"],false)
#
# t = 34#36#34
# setobject!(vis["ct4"],ctm)
# settransform!(vis["ct4"], compose(Translation([q[t][1];q[t][2];0.0]),LinearMap(RotZ(q[t][3]+pi)*RotX(pi/2.0))))
# setvisible!(vis["ct4"],false)
# t = T
# setobject!(vis["ct5"],ctm)
# settransform!(vis["ct5"], compose(Translation([q[t][1];q[t][2];0.0]),LinearMap(RotZ(q[t][3]+pi)*RotX(pi/2.0))))
# setvisible!(vis["ct5"],false)
#
# # setvisible!(vis["/Background"], true)
# # setprop!(vis["/Background"], "top_color", [135,206,250])
# # setprop!(vis["/Background"], "bottom_color", [135,206,250])
# # vis["/Background"].core
