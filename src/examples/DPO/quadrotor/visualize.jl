using Plots
include(joinpath(@__DIR__, "quadrotor_broken_propeller.jl"))
include(joinpath(pwd(), "src/models/visualize.jl"))

X̄, Ū = unpack(Z̄, prob)

@show sum([Ū[t][end] for t = 1:T-1])
t_nominal = [0.0, [sum([Ū[i][end] for i = 1:t]) for t = 1:T-1]...]
plot(t_nominal, hcat(X̄...)[1:3,:]', linetype=:steppost, width = 2.0)
plot(t_nominal[1:end-1], hcat(Ū...)', linetype=:steppost, width = 2.0)

vis = Visualizer()
open(vis)
visualize!(vis, model, X̄, Δt = Ū[1][end])
