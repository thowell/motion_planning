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
