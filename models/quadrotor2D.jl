"""
    Planar quadrotor
"""

mutable struct Quadrotor2D{I, T} <: Model{I, T}
    n::Int
    m::Int
    d::Int

    L    # length
    mass # mass
    J    # inertia
    g    # gravity
end

function f(model::Quadrotor2D, x, u, w)
    qdd1 = -1.0 * sin(x[3]) / model.mass * (u[1] + u[2])
    qdd2 = -1.0 * model.g + cos(x[3]) / model.mass * (u[1] + u[2])
    qdd3 = model.L / model.J * (-1.0 * u[1] + u[2])

    @SVector [x[4],
              x[5],
              x[6],
              qdd1,
              qdd2,
              qdd3]
end

n, m, d = 6, 2, 0
model = Quadrotor2D{Midpoint, FixedTime}(n, m, d, 1.0, 1.0, 1.0, 9.81)
