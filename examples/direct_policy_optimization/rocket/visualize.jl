using Plots
include(joinpath(pwd(), "src/models/visualize.jl"))

X̄, Ū = unpack(Z̄, prob_nominal)

t_nominal = range(0, stop = sum([Ū[t][end] for t = 1:T-1]), length = T)
plot(t_nominal[1:end-1], hcat(Ū...)[1:end-1, :]', linetype=:steppost)
@show sum([Ū[t][end] for t = 1:T-1]) # works when 2.72

vis = Visualizer()
open(vis)
visualize!(vis, model_nom, X̄, Δt = Ū[1][end])

# COM traj
xx_nom = [X̄[t][1] for t = 1:T]
zz_nom = [X̄[t][2] for t = 1:T]

X̄_dpo, Ū_dpo = unpack(Z[prob_dpo.prob.idx.nom], prob_dpo.prob.prob.nom)
t_dpo = range(0, stop = sum([Ū_dpo[t][end] for t = 1:T-1]), length = T)

xx = [X̄_dpo[t][1] for t = 1:T]
zz = [X̄_dpo[t][2] for t = 1:T]

plot(xx_nom, zz_nom, color = :purple)
plot!(xx, zz, color = :orange)

plot(hcat(Ū_dpo...)[3:3,:]')
Ū_dpo[1][end]


plot(t_nominal, hcat(X̄...)', color = :purple)
plot!(t_dpo, hcat(X̄_dpo...)', color = :orange)


# orientation tracking
plt_x = plot(t_nom,hcat(X̄...)[3,:],legend=:topright,color=:red,
    label="",width=2.0,xlabel="time (s)",
    title="Rocket",ylabel="state")
plt_x = plot!(t_sim_nom,hcat(z_tvlqr...)[3,:],color=:black,label="",
    width=1.0)

plt_x = plot(t_dpo,hcat(X̄_dpo...)[3,:],legend=:topright,color=:red,
    label="",width=2.0,xlabel="time (s)",
    title="Rocket",ylabel="state")
plt_x = plot!(t_sim_dpo,hcat(z_dpo...)[3,:],color=:black,label="",
    width=1.0)

visualize!(vis, model_nom, z_tvlqr, Δt = dt_sim_nom)
visualize!(vis, model_nom, z_sample, Δt = dt_sim_sample)

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
C = Diagonal(ones(model_sl.nx))[1:model_nom.nx,:]
# nominal
A_nom, B_nom = nominal_jacobians(model_nom,X_nom,U_nom,H_nom)
A_nom_cl = [(A_nom[t] - B_nom[t]*K[t]) for t = 1:T-1]
sv_nom = [norm.(eigen(A_nom_cl[t]).values) for t = 1:T-1]
plt_nom = plot(hcat(sv_nom...)',xlabel="time step t",ylabel="eigen value norm",
	title="TVLQR nominal model",linetype=:steppost,
	ylims=(-3,3),labels="")

# slosh nominal
X_nom_slosh = [[copy(X_nom[t]);0.0;0.0] for t = 1:T]
A_nom_slosh, B_nom_slosh = nominal_jacobians(model_sl,X_nom_slosh,U_nom,H_nom)
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
A_dpo_slosh, B_dpo_slosh = nominal_jacobians(model_sl,X_dpo_slosh,U_nom_sample,H_nom_sample)
A_dpo_slosh_cl = [(A_dpo_slosh[t] - B_dpo_slosh[t]*Θ_mat[t]*C) for t = 1:T-1]
sv_dpo_slosh = [norm.(eigen(A_dpo_slosh_cl[t]).values) for t = 1:T-1]
plt_dpo_slosh = plot(hcat(sv_dpo_slosh...)',xlabel="time step t",ylabel="eigen value norm",
	title="DPO slosh model",linetype=:steppost,
	ylims=(-1.0,3.0),labels="")

plot(plt_nom,plt_dpo_nom,layout=(2,1))

plot(plt_nom_slosh,plt_dpo_slosh,layout=(2,1))


plot(t_nom,hcat(X_nom...)[1:6,:]',color=:cyan,label=["y" "z" "ϕ"])
plot!(t_nom_sample, hcat(X_nom_sample...)[1:6,:]',color=:orange,label=["y" "z" "ϕ"])
