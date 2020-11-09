using Plots
include(joinpath(@__DIR__, "rocket_fuel_slosh.jl"))
include(joinpath(pwd(), "src/models/visualize.jl"))

X̄, Ū = unpack(Z̄, prob)

t_nominal = [0.0, [sum([Ū[i][end] for i = 1:t]) for t = 1:T-1]...]
plot(t_nominal[1:end-1], hcat(Ū...)[1:end-1, :]', linetype=:steppost)
@show sum([Ū[t][end] for t = 1:T-1]) # works when 2.72

vis = Visualizer()
open(vis)
visualize!(vis, model_nominal, X̄, Δt = Ū[1][end])

# COM traj
xx_nom = [X̄[t][1] for t = 1:T]
zz_nom = [X̄[t][2] for t = 1:T]

X̄_dpo, Ū_dpo = unpack(Z[prob_dpo.prob.idx.nom], prob_dpo.prob.prob.nom)
xx = [X̄_dpo[t][1] for t = 1:T]
zz = [X̄_dpo[t][2] for t = 1:T]

using Plots
plot(xx_nom, zz_nom, color = :purple)
plot!(xx, zz, color = :orange)
