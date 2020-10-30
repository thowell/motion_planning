using Plots
include(joinpath(@__DIR__, "cartpole_friction.jl"))

# Unpack solutions
X̄_nominal, Ū_nominal = unpack(Z̄_nominal, prob_nominal)
X̄_friction, Ū_friction = unpack(Z̄_friction, prob_friction)

# Plots results
S_friction_nominal = [Ū_friction[t][7] for t = 1:T-1]
@assert norm(S_friction_nominal, Inf) < 1.0e-4
b_friction_nominal = [(Ū_friction[t][2] - Ū_friction[t][3]) for t = 1:T-1]

t_nominal = range(0, stop = h * (T - 1), length = T)

# Control
plt = plot(t_nominal[1:T-1], hcat(Ū_nominal...)[1:1, :]',
    color = :purple, width=2.0,
    title = "Cartpole", xlabel = "time (s)", ylabel = "control", label = "nominal",
    legend = :topright, linetype = :steppost)
plt = plot!(t_nominal[1:T-1], hcat(Ū_friction...)[1:1, :]', color = :orange,
    width = 2.0, label = "nominal (friction)", linetype = :steppost)

# States
plt = plot(t_nominal, hcat(X̄_nominal...)[1:4, :]',
    color = :purple, width = 2.0, xlabel = "time (s)",
    ylabel = "state", label = "", title = "Cartpole", legend = :topright)

plt = plot!(t_nominal, hcat(X̄_friction...)[1:4,:]',
    color = :orange, width = 2.0, label = "")

B̄_nominal = [Ū_nominal[t][2:3] for t = 1:T-1]
B̄_friction = [Ū_friction[t][2:3] for t = 1:T-1]

plot(t_nominal[1:end-1], hcat(B̄_nominal...)', linetype = :steppost, width = 2.0)
plot(t_nominal[1:end-1], hcat(B̄_friction...)', linetype = :steppost, width = 2.0)
