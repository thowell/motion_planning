using Plots
include(joinpath(@__DIR__, "pendulum_minimum_time.jl"))

# Unpack solutions
X̄, Ū = unpack(Z̄, prob)

@show sum([Ū[t][end] for t = 1:T-1])
t_nominal = [0.0, [sum([Ū[i][end] for i = 1:t]) for t = 1:T-1]...]
plot(t_nominal, hcat(X̄...)', width = 2.0)
plot(t_nominal[1:end-1], hcat(Ū...)[1:1, :]', linetype = :steppost, width = 2.0)
