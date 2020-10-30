"""
	Cart-pole
"""

struct Cartpole{T} <: Model
	n::Int
    m::Int
	d::Int

    mc::T     # mass of the cart in kg
    mp::T     # mass of the pole (point mass at the end) in kg
    l::T      # length of the pole in m
    g::T      # gravity m/s^2
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

n, m, d = 4, 1, 0
model = Cartpole(n, m, d, 1.0, 0.2, 0.5, 9.81)

"""
	Cart-pole with Coulomb friction
"""
struct CartpoleFriction{T} <: Model
	n::Int
    m::Int
	d::Int

    mc::T     # mass of the cart in kg
    mp::T     # mass of the pole (point mass at the end) in kg
    l::T      # length of the pole in m
    g::T      # gravity m/s^2
	μ::T      # coefficient of friction
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

n, m, d = 4, 7, 0
model_friction = CartpoleFriction(n, m, d, 1.0, 0.2, 0.5, 9.81, 0.1)
