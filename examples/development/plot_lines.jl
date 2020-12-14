using MeshCat, GeometryBasics, Colors

vis = Visualizer()
render(vis) # open(vis)

# circle
θ = range(0, stop = 2 * π, length = 100)
posy = 0.2 * cos.(θ)
posz = 0.8 .+ 0.2 .* sin.(θ)

ee_positions = hcat([[0.0; posy[t]; posz[t]] for t = 1:100]...)
points = collect(eachcol(ee_positions))
material = LineBasicMaterial(color=colorant"orange", linewidth=6.0)
setobject!(vis["ee path"], Object(PointCloud(points), material, "Line"))
