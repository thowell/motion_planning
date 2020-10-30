"""
    Double integrator
"""

struct DoubleIntegrator
    n::Int
    m::Int
    d::Int
end

function f(model::DoubleIntegrator, x, u, w)
    @SVector [x[2],
              u[1]]
end

n, m, d = 2, 1, 0
model = DoubleIntegrator(n, m, d)
