using Plots

x̄, ū = unpack(z̄, prob)

# Position trajectory
Px = [x̄[t][1] for t = 1:T]
Py = [x̄[t][2] for t = 1:T]

pts = Plots.partialcircle(0.0, 2.0 * π, 100, 0.1)
cx, cy = Plots.unzip(pts)
cx1 = [_cx + circles[1][1] for _cx in cx]
cy1 = [_cy + circles[1][2] for _cy in cy]
cx2 = [_cx + circles[2][1] for _cx in cx]
cy2 = [_cy + circles[2][2] for _cy in cy]
cx3 = [_cx + circles[3][1] for _cx in cx]
cy3 = [_cy + circles[3][2] for _cy in cy]
cx4 = [_cx + circles[4][1] for _cx in cx]
cy4 = [_cy + circles[4][2] for _cy in cy]

plt = plot(Shape(cx1, cy1), color = :red, label = "", linecolor = :red)
plt = plot!(Shape(cx2, cy2), color = :red, label = "", linecolor = :red)
plt = plot!(Shape(cx3, cy3), color = :red, label = "", linecolor = :red)
plt = plot!(Shape(cx4, cy4), color = :red, label = "", linecolor = :red)
plt = plot!(Px, Py, aspect_ratio = :equal, xlabel = "x", ylabel = "y",
    width = 4.0, label = "TO", color = :purple, legend = :topleft)

# DPO
x, u = unpack(z, prob_dpo.prob.prob.nom)

# Position trajectory
Px_dpo = [x[t][1] for t = 1:T]
Py_dpo = [x[t][2] for t = 1:T]
plt = plot!(Px_dpo, Py_dpo, width = 4.0, label = "DPO", color = :orange)
