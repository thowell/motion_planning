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
    mark = "none",
	style = "color=cyan, line width=2pt, solid",
	legendentry = "TO")

# DPO trajectory
p_dpo = PGF.Plots.Linear(hcat(x...)[1,:], hcat(x...)[2,:],
    mark = "none",
	style = "color=orange, line width=2pt, solid",
	legendentry = "DPO")

# obstacles
pc1 = PGF.Plots.Circle(circles[1]...,
	style = "color=black, fill=black")
pc2 = PGF.Plots.Circle(circles[2]...,
	style = "color=black, fill=black")
pc3 = PGF.Plots.Circle(circles[3]...,
	style = "color=black, fill=black")
pc4 = PGF.Plots.Circle(circles[4]...,
	style = "color=black, fill=black")

a = Axis([
    p_nom;
    p_dpo;
	pc1;
	pc2;
	pc3;
	pc4],
    xmin = -0.4, ymin = -0.1, xmax = 1.4, ymax = 1.1,
    axisEqualImage = true,
    hideAxis = true,
	# ylabel = "y",
	# xlabel = "x",
	legendStyle = "{at={(0.01, 0.99)}, anchor = north west}")

# Save to tikz format
PGF.save(joinpath(@__DIR__, "car_obstacles.tikz"), a, include_preamble = false)
