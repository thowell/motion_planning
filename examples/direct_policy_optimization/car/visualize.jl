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

plt = plot(Shape(cx1, cy1), color = :black, label = "", linecolor = :black)
plt = plot!(Shape(cx2, cy2), color = :black, label = "", linecolor = :black)
plt = plot!(Shape(cx3, cy3), color = :black, label = "", linecolor = :black)
plt = plot!(Shape(cx4, cy4), color = :black, label = "", linecolor = :black)
plt = plot!(Px, Py, aspect_ratio = :equal, xlabel = "x", ylabel = "y",
    width = 4.0, label = "TO", color = goldenrod_color, legend = :topleft)

# DPO
x, u = unpack(z, prob_dpo.prob.prob.nom)

# Position trajectory
Px_dpo = [x[t][1] for t = 1:T]
Py_dpo = [x[t][2] for t = 1:T]
plt = plot!(Px_dpo, Py_dpo, width = 4.0, label = "DPO", color = red_color)

# PGFPlots for paper
using PGFPlots
const PGF = PGFPlots

# TO trajectory
p_nom = PGF.Plots.Linear(hcat(x̄...)[1,:], hcat(x̄...)[2,:],
    mark = "",
	style = "color=cyan, line width=2pt, solid",
	legendentry = "TO")

# DPO trajectory
p_dpo = PGF.Plots.Linear(hcat(x...)[1,:], hcat(x...)[2,:],
    mark = "",
	style = "color=orange, line width=3pt, solid",
	legendentry = "DPO")

# obstacles
p_circle = [PGF.Plots.Circle(circle...,
	style = "color=black,fill=black") for circle in circles]

a = Axis([p_circle;
    p_sample[1];
    p_sample[2];
    p_sample[3];
    p_sample[4];
    p_sample[5];
    p_sample[6];
    p_nom;
    p_dpo],
    xmin = -0.4, ymin = -0.1, xmax = 1.4, ymax = 1.1,
    axisEqualImage = true,
    hideAxis = false,
	ylabel = "y",
	xlabel = "x",
	legendStyle = "{at={(0.01,0.99)}, anchor=north west}")

# Save to tikz format
dir = joinpath(@__DIR__, "figures")
PGF.save(joinpath(dir, "car_obstacles.tikz"), a, include_preamble = false)
