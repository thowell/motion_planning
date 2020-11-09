using Plots
include(joinpath(@__DIR__, "pendulum_minimum_time.jl"))

# TO solution
X̄, Ū = unpack(Z̄, prob)

@show sum([Ū[t][end] for t = 1:T-1])
t_nominal = [0.0, [sum([Ū[i][end] for i = 1:t]) for t = 1:T-1]...]
plot(t_nominal, hcat(X̄...)', width = 2.0)
plot(t_nominal[1:end-1], hcat(Ū...)[1:1, :]',
	linetype = :steppost, width = 2.0, color = :purple)

# DPO solution
X, U = unpack(Z, prob_dpo.prob.prob.nom)

@show sum([U[t][end] for t = 1:T-1])
t_dpo = [0.0, [sum([U[i][end] for i = 1:t]) for t = 1:T-1]...]
plot(t_dpo, hcat(X...)', width = 2.0)
plot(t_dpo[1:end-1], hcat(U...)[1:1, :]',
	linetype = :steppost, width = 2.0, color = :orange)
