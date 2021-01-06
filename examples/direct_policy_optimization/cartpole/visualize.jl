using Plots

# Nominal trajectories
x̄_nominal, ū_nominal = unpack(z̄_nominal, prob_nominal)
x̄_friction, ū_friction = unpack(z̄_friction, prob_friction)

# Plots results
s_friction_nominal = [ū_friction[t][7] for t = 1:T-1]
@assert norm(s_friction_nominal, Inf) < 1.0e-4

t_nominal = range(0, stop = h * (T - 1), length = T)

# Control
plt = plot(t_nominal[1:T-1], hcat(ū_nominal...)[1:1, :]',
    color = darkslateblue_color, width=2.0,
    title = "Cartpole",
	xlabel = "time (s)",
	ylabel = "control",
	label = "no friction",
    legend = :topright,
	linetype = :steppost)
plt = plot!(t_nominal[1:T-1], hcat(ū_friction...)[1:1, :]',
	color = goldenrod_color,
    width = 2.0, label = "friction",
	linetype = :steppost)

# States
plt = plot(t_nominal, hcat(x̄_nominal...)[1:4, :]',
    color = darkslateblue_color, width = 2.0, xlabel = "time (s)",
    ylabel = "state", label = "", title = "Cartpole",
	legend = :topright)

plt = plot!(t_nominal, hcat(x̄_friction...)[1:4,:]',
    color = goldenrod_color, width = 2.0, label = "")

include(joinpath(pwd(), "models/visualize.jl"))
vis = Visualizer()
render(vis)
# open(vis)
visualize!(vis, model_nominal, x̄_nominal, Δt = h)

# DPO
x_dpo, u_dpo = unpack(z[prob_dpo.prob.idx.nom], prob_dpo.prob.prob.nom)
s_dpo = [u_dpo[t][7] for t = 1:T-1]
@assert norm(s_dpo, Inf) < 1.0e-4

# Control
plt = plot(t_nominal[1:T-1], hcat(u_dpo...)[1:1, :]',
    color = red_color, width = 2.0,
    title = "Cartpole",
	xlabel = "time (s)", ylabel = "control", label = "dpo",
    legend = :topright, linetype = :steppost)

# States
plt = plot(t_nominal, hcat(x_dpo...)[1:4, :]',
    color = red_color, width = 2.0, xlabel = "time (s)",
    ylabel = "state", label = "dpo", title = "Cartpole",
	legend = :topright)

# Plots
plt_lqr_nom = plot(t_nominal, hcat(x̄_nominal...)[1:2, :]',
	legend = :topleft, color = darkslateblue_color,
    label = ["reference (no friction)" ""],
    width = 2.0, xlabel = "time",
	title = "Cartpole", ylabel = "state", ylims = (-1, 5))
plt_lqr_nom = plot!(t_sim, hcat(z_lqr...)[1:2,:]', color = :black,
    label=["lqr" ""], width = 1.0)

plt_lqr_friction = plot(t_nominal, hcat(x̄_friction...)[1:2,:]',
    color = goldenrod_color, label = ["reference (friction)" ""],
    width=2.0,xlabel="time",title="Cartpole",ylabel="state",legend=:topleft)
plt_tvlqr_friction = plot!(t_sim, hcat(z_lqr_fr...)[1:2,:]',
    color = :black, label = ["lqr" ""], width = 1.0)

plt_dpo = plot(t_nominal, hcat(x_dpo...)[1:2,:]',
	legend=:topleft, color = red_color,
    label=["reference" ""],
    width = 2.0, xlabel = "time", title = "Cartpole", ylabel = "state")
plt_dpo = plot!(t_sim, hcat(z_dpo...)[1:2,:]', color=:black,
    label=["dpo" ""], width = 1.0)
# savefig(plt_sample,joinpath(@__DIR__,"results/cartpole_friction_sample_sim.png"))
#
# # z_sample_c, u_sample_c, J_sample_c, Jx_sample_c, Ju_sample_c = simulate_cartpole_friction(K_sample_coefficients,
# #     X_nom_sample_coefficients,U_nom_sample_coefficients,
# #     model_sim,Q_lqr,R_lqr,T_sim,Δt,X_nom_sample_coefficients[1],w,ul=ul_friction,
# #     uu=uu_friction,friction=true,
# #     μ=μ_sim)
#
# # plt_sample = plot(t_sample_c,hcat(X_nom_sample_coefficients...)[1:2,:]',
# #     legend=:bottom,color=:red,label=["nominal (sample)" ""],
# #     width=2.0,xlabel="time (s)",title="Cartpole",ylabel="state")
# # plt_sample = plot!(t_sim_nominal,hcat(z_sample_c...)[1:2,:]',color=:magenta,
# #     label=["sample" ""],width=2.0,legend=:topleft)
# # savefig(plt_sample,joinpath(@__DIR__,"results/cartpole_friction_sample_sim.png"))
#
#
# using PGFPlots
# const PGF = PGFPlots
#
# # nominal trajectory
# psx_nom = PGF.Plots.Linear(t_nominal,hcat(X_nominal...)[1,:],mark="",
# 	style="color=cyan, line width=3pt")
# psθ_nom = PGF.Plots.Linear(t_nominal,hcat(X_nominal...)[2,:],mark="",
# 	style="color=cyan, line width=3pt, densely dashed")
#
# psx_nom_sim_tvlqr = PGF.Plots.Linear(t_sim_nominal,hcat(z_tvlqr...)[1,:],mark="",
# 	style="color=black, line width=2pt")
# psθ_nom_sim_tvlqr = PGF.Plots.Linear(t_sim_nominal,hcat(z_tvlqr...)[2,:],mark="",
# 	style="color=black, line width=2pt, densely dashed")
#
# a = Axis([psx_nom;psθ_nom;psx_nom_sim_tvlqr;psθ_nom_sim_tvlqr],
#     xmin=0., ymin=-1, xmax=sum(H_nominal), ymax=3.5,
#     axisEqualImage=false,
#     hideAxis=false,
# 	ylabel="state",
# 	xlabel="time",
# 	legendStyle="{at={(0.01,0.99)},anchor=north west}",
# 	)
#
# # Save to tikz format
# dir = joinpath(@__DIR__,"results")
# PGF.save(joinpath(dir,"cartpole_sim_tvlqr_no_friction.tikz"), a, include_preamble=false)
#
# # nominal trajectory
# psx_nom_friction = PGF.Plots.Linear(t_nominal,hcat(X_friction_nominal...)[1,:],mark="",
# 	style="color=purple, line width=3pt")
# psθ_nom_friction = PGF.Plots.Linear(t_sample,hcat(X_friction_nominal...)[2,:],mark="",
# 	style="color=purple, line width=3pt, densely dashed")
#
# psx_nom_sim_tvlqr_friction = PGF.Plots.Linear(t_sim_nominal,hcat(z_tvlqr_friction...)[1,:],mark="",
# 	style="color=black, line width=2pt",legendentry="pos.")
# psθ_nom_sim_tvlqr_friction = PGF.Plots.Linear(t_sim_nominal,hcat(z_tvlqr_friction...)[2,:],mark="",
# 	style="color=black, line width=2pt, densely dashed",legendentry="ang.")
#
# a_tvlqr_friction = Axis([psx_nom_friction;psθ_nom_friction;
#     psx_nom_sim_tvlqr_friction;psθ_nom_sim_tvlqr_friction],
#     xmin=0., ymin=-1, xmax=sum(H_nominal), ymax=3.5,
#     axisEqualImage=false,
#     hideAxis=false,
# 	ylabel="state",
# 	xlabel="time",
# 	legendStyle="{at={(0.01,0.99)},anchor=north west}",
# 	)
#
# # Save to tikz format
# dir = joinpath(@__DIR__,"results")
# PGF.save(joinpath(dir,"cartpole_sim_tvlqr_friction.tikz"), a_tvlqr_friction, include_preamble=false)
#
# # nominal trajectory
# psx_dpo = PGF.Plots.Linear(t_sample,hcat(X_nom_sample...)[1,:],mark="",
# 	style="color=orange, line width=3pt")
# psθ_dpo = PGF.Plots.Linear(t_sample,hcat(X_nom_sample...)[2,:],mark="",
# 	style="color=orange, line width=3pt, densely dashed")
#
# psx_sim_dpo = PGF.Plots.Linear(t_sim_sample,hcat(z_sample...)[1,:],mark="",
# 	style="color=black, line width=2pt",legendentry="pos.")
# psθ_sim_dpo = PGF.Plots.Linear(t_sim_sample,hcat(z_sample...)[2,:],mark="",
# 	style="color=black, line width=2pt, densely dashed",legendentry="ang.")
#
# a_dpo = Axis([psx_dpo;psθ_dpo;psx_sim_dpo;
#     psθ_sim_dpo],
#     xmin=0., ymin=-1, xmax=sum(H_nom_sample), ymax=3.5,
#     axisEqualImage=false,
#     hideAxis=false,
# 	ylabel="state",
# 	xlabel="time",
# 	legendStyle="{at={(0.01,0.99)},anchor=north west}",
# 	)
#
# # Save to tikz format
# dir = joinpath(@__DIR__,"results")
# PGF.save(joinpath(dir,"cartpole_sim_dpo.tikz"), a_dpo, include_preamble=false)
