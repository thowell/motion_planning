"""
    Car

    Unicycle model (http://lavalle.pl/planning/)
"""

struct Car{I, T} <: Model{I, T}
    n::Int
    m::Int
    d::Int
end

function f(model::Car, x, u, w)
    @SVector [u[1] * cos(x[3]),
              u[1] * sin(x[3]),
              u[2]]
end

n, m, d = 3, 2, 0
model = Car{Midpoint, FixedTime}(n, m, d)
