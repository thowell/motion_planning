using Plots

# TO solution
x̄, ū = unpack(z̄, prob)
@show tf_nom = sum([ū[t][end] for t = 1:T-1]) # total time
t_nominal = range(0, stop = tf_nom, length = T)
plot(t_nominal, hcat(x̄...)', width = 2.0)
plot(t_nominal[1:end-1], hcat(ū...)[1:1, :]',
	linetype = :steppost, width = 2.0, color = :purple)

# DPO solution
x, u = unpack(z, prob_dpo.prob.prob.nom)
@show tf_dpo = sum([u[t][end] for t = 1:T-1])
t_dpo = range(0, stop = tf_dpo, length = T)
plot(t_dpo, hcat(x...)', width = 2.0)
plot(t_dpo[1:end-1], hcat(u...)[1:1, :]',
	linetype = :steppost, width = 2.0, color = :orange)
