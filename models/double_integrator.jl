"""
    Double integrator
"""

struct DoubleIntegrator{I, T} <: Model{I, T}
    n::Int
    m::Int
    d::Int
end

function fd(model::DoubleIntegrator{Discrete, FixedTime}, x⁺, x, u, w, h, t)
    x⁺ - [x[1] + x[2] + w[1]; x[2] + u[1] + w[2]]
end

function get_dynamics(model::DoubleIntegrator)
    A = @SMatrix [1.0 1.0; 0.0 1.0]
    B = @SMatrix [0.0; 1.0]

    return A, B
end

n, m, d = 2, 1, 2
model = DoubleIntegrator{Discrete, FixedTime}(n, m, d)
