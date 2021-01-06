"""
	Cart-pole
"""

struct Cartpole{I, T} <: Model{I, T}
	n::Int
    m::Int
	d::Int

    mc     # mass of the cart in kg
    mp     # mass of the pole (point mass at the end) in kg
    l      # length of the pole in m
    g      # gravity m/s^2
end

function f(model::Cartpole, x, u, w)
    H = @SMatrix [model.mc + model.mp model.mp * model.l * cos(x[2]);
				  model.mp * model.l * cos(x[2]) model.mp * model.l^2.0]
    C = @SMatrix [0.0 -1.0 * model.mp * x[4] * model.l * sin(x[2]);
	 			  0.0 0.0]
    G = @SVector [0.0,
				  model.mp * model.g * model.l * sin(x[2])]
    B = @SVector [1.0,
				  0.0]
    qdd = SVector{2}(-H \ (C * view(x, 3:4) + G - B * u[1]))

    return @SVector [x[3],
					 x[4],
					 qdd[1],
					 qdd[2]]
end

n, m, d = 4, 1, 4
model = Cartpole{Midpoint, FixedTime}(n, m, d, 1.0, 0.2, 0.5, 9.81)

"""
	Cart-pole with Coulomb friction
"""
struct CartpoleFriction{I, T} <: Model{I, T}
	n::Int
    m::Int
	d::Int

    mc     # mass of the cart in kg
    mp     # mass of the pole (point mass at the end) in kg
    l      # length of the pole in m
    g      # gravity m/s^2
	μ      # coefficient of friction
end

function f(model::CartpoleFriction, x, u, w)
    H = @SMatrix [model.mc + model.mp model.mp * model.l * cos(x[2]);
				  model.mp * model.l * cos(x[2]) model.mp * model.l^2.0]
    C = @SMatrix [0.0 -1.0 * model.mp * x[4] * model.l * sin(x[2]);
	 			  0.0 0.0]
    G = @SVector [0.0,
				  model.mp * model.g * model.l * sin(x[2])]
    B = @SVector [1.0,
				  0.0]
    qdd = SVector{2}(-H \ (C * view(x, 3:4) + G - B * (u[1] + u[2] - u[3])))

    return @SVector [x[3],
					 x[4],
					 qdd[1],
					 qdd[2]]
end

n, m, d = 4, 7, 4
model_friction = CartpoleFriction{Midpoint, FixedTime}(n, m, d, 1.0, 0.2, 0.5, 9.81, 0.1)

function visualize!(vis, model, q;
       Δt = 0.1,
	   color = RGBA(1,0,0,1.0))

	default_background!(vis)

	l2 = Cylinder(Point3f0(-model.l * 2.0, 0.0, 0.0),
		Point3f0(model.l * 2.0, 0.0, 0.0),
		convert(Float32, 0.025))

	setobject!(vis["slider"], l2, MeshPhongMaterial(color = RGBA(0.0, 0.0, 0.0, 1.0)))

    l1 = Cylinder(Point3f0(0.0, 0.0, 0.0),
		Point3f0(0.0, 0.0, model.l),
		convert(Float32, 0.025))

    setobject!(vis["arm"], l1,
		MeshPhongMaterial(color = RGBA(0.0, 0.0, 0.0, 1.0)))

    setobject!(vis["base"], HyperSphere(Point3f0(0.0),
        convert(Float32, 0.1)),
        MeshPhongMaterial(color = color))

    setobject!(vis["ee"], HyperSphere(Point3f0(0.0),
        convert(Float32, 0.05)),
        MeshPhongMaterial(color = color))

	anim = MeshCat.Animation(convert(Int, floor(1.0 / Δt)))

	for t = 1:length(q)
	    MeshCat.atframe(anim,t) do
			x = q[t]
			px = x[1] + model.l * sin(x[2])
			pz = -model.l * cos(x[2])
            settransform!(vis["arm"], cable_transform([x[1]; 0;0], [px; 0.0; pz]))
            settransform!(vis["base"], Translation([x[1]; 0.0; 0.0]))
            settransform!(vis["ee"], Translation([px; 0.0; pz]))
	    end
	end

	settransform!(vis["/Cameras/default"],
		compose(Translation(0.0, 0.0, -1.0), LinearMap(RotZ(- pi / 2))))

	MeshCat.setanimation!(vis,anim)
end
