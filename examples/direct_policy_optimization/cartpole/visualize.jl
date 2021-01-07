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
    color = :magenta, width=2.0,
    title = "Cartpole",
	xlabel = "time (s)",
	ylabel = "control",
	label = "no friction",
    legend = :topright,
	linetype = :steppost)
plt = plot!(t_nominal[1:T-1], hcat(ū_friction...)[1:1, :]',
	color = :cyan,
    width = 2.0, label = "friction",
	linetype = :steppost)

# States
plt = plot(t_nominal, hcat(x̄_nominal...)[1:4, :]',
    color = :magenta, width = 2.0, xlabel = "time (s)",
    ylabel = "state", label = "", title = "Cartpole",
	legend = :topright)

plt = plot!(t_nominal, hcat(x̄_friction...)[1:4,:]',
    color = :cyan, width = 2.0, label = "")

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
    color = :orange, width = 2.0,
    title = "Cartpole",
	xlabel = "time (s)", ylabel = "control", label = "dpo",
    legend = :topright, linetype = :steppost)

# States
plt = plot(t_nominal, hcat(x_dpo...)[1:4, :]',
    color = :orange, width = 2.0, xlabel = "time (s)",
    ylabel = "state", label = "dpo", title = "Cartpole",
	legend = :topright)

# Plots
plt_lqr_nom = plot(t_nominal, hcat(x̄_nominal...)[1:2, :]',
	legend = :topleft, color = :magenta,
    label = ["reference (no friction)" ""],
    width = 2.0, xlabel = "time",
	title = "Cartpole", ylabel = "state", ylims = (-1, 5))
plt_lqr_nom = plot!(t_sim, hcat(z_lqr...)[1:2,:]', color = :black,
    label=["lqr" ""], width = 1.0)

plt_lqr_friction = plot(t_nominal, hcat(x̄_friction...)[1:2,:]',
    color = :cyan, label = ["reference (friction)" ""],
    width=2.0,xlabel="time",title="Cartpole",ylabel="state",legend=:topleft)
plt_tvlqr_friction = plot!(t_sim, hcat(z_lqr_fr...)[1:2,:]',
    color = :black, label = ["lqr" ""], width = 1.0)

plt_dpo = plot(t_nominal, hcat(x_dpo...)[1:2,:]',
	legend=:topleft, color = :orange,
    label=["reference" ""],
    width = 2.0, xlabel = "time", title = "Cartpole", ylabel = "state")
plt_dpo = plot!(t_sim, hcat(z_dpo...)[1:2,:]', color=:black,
    label=["dpo" ""], width = 1.0)

using PGFPlots
const PGF = PGFPlots

# nominal trajectory
psx_nom = PGF.Plots.Linear(t_nominal,hcat(x̄_nominal...)[1,:],mark="none",
	style="color=magenta, line width=2pt")
psθ_nom = PGF.Plots.Linear(t_nominal,hcat(x̄_nominal...)[2,:],mark="none",
	style="color=magenta, line width=2pt")

psx_nom_sim_lqr = PGF.Plots.Linear(t_sim,hcat(z_lqr...)[1,:],mark="none",
	style="color=black, line width=1pt, densely dotted")
psθ_nom_sim_lqr = PGF.Plots.Linear(t_sim,hcat(z_lqr...)[2,:],mark="none",
	style="color=black, line width=1pt, dashed")

a_lqr_no_friction = Axis([psx_nom;psθ_nom;psx_nom_sim_lqr;psθ_nom_sim_lqr],
    xmin=0., ymin=-1, xmax=tf, ymax=3.5,
    axisEqualImage=false,
    hideAxis=false,
	ylabel="state",
	xlabel="time",
	legendStyle="{at={(0.01,0.99)},anchor=north west}")

# Save to tikz format
dir = joinpath(pwd(),"examples/direct_policy_optimization/figures")
PGF.save(joinpath(dir,"cartpole_lqr_no_friction.tikz"), a_lqr_no_friction, include_preamble=false)

# nominal trajectory
psx_nom_friction = PGF.Plots.Linear(t_nominal,hcat(x̄_friction...)[1,:],mark="none",
	style="color=cyan, line width=2pt")
psθ_nom_friction = PGF.Plots.Linear(t_nominal,hcat(x̄_friction...)[2,:],mark="none",
	style="color=cyan, line width=2pt")

psx_nom_sim_lqr_friction = PGF.Plots.Linear(t_sim,hcat(z_lqr_fr...)[1,:],mark="none",
	style="color=black, line width=1pt, densely dotted",legendentry="pos.")
psθ_nom_sim_lqr_friction = PGF.Plots.Linear(t_sim,hcat(z_lqr_fr...)[2,:],mark="none",
	style="color=black, line width=1pt, dashed",legendentry="ang.")

a_lqr_friction = Axis([psx_nom_friction;psθ_nom_friction;
    psx_nom_sim_lqr_friction;psθ_nom_sim_lqr_friction],
    xmin=0., ymin=-1, xmax=tf, ymax=3.5,
    axisEqualImage=false,
    hideAxis=false,
	ylabel="state",
	xlabel="time",
	legendStyle="{at={(0.01,0.99)},anchor=north west}")

# Save to tikz format
dir = joinpath(pwd(),"examples/direct_policy_optimization/figures")
PGF.save(joinpath(dir,"cartpole_lqr_friction.tikz"), a_tvlqr_friction, include_preamble=false)

# nominal trajectory
psx_dpo = PGF.Plots.Linear(t_nominal,hcat(x̄_dpo...)[1,:],mark="none",
	style="color=orange, line width=2pt")
psθ_dpo = PGF.Plots.Linear(t_nominal,hcat(x̄_dpo...)[2,:],mark="none",
	style="color=orange, line width=2pt")

psx_sim_dpo = PGF.Plots.Linear(t_sim,hcat(z_dpo...)[1,:],mark="none",
	style="color=black, line width=1pt, densely dotted",legendentry="pos.")
psθ_sim_dpo = PGF.Plots.Linear(t_sim,hcat(z_dpo...)[2,:],mark="none",
	style="color=black, line width=1pt, dashed",legendentry="ang.")

a_dpo = Axis([psx_dpo;psθ_dpo;psx_sim_dpo;
    psθ_sim_dpo],
    xmin=0., ymin=-1, xmax=tf, ymax=3.5,
    axisEqualImage=false,
    hideAxis=false,
	ylabel="state",
	xlabel="time",
	legendStyle="{at={(0.01,0.99)},anchor=north west}",
	)

# Save to tikz format
dir = joinpath(pwd(),"examples/direct_policy_optimization/figures")
PGF.save(joinpath(dir,"cartpole_dpo.tikz"), a_dpo, include_preamble=false)
