struct Pendulum{T} <: Model
    n::Int
    m::Int
    d::Int

    mass::T    # mass
    b::T    # friction
    lc::T   # length to center of mass
    I::T    # inertia
    g::T    # gravity

end

function f(model::Pendulum, x, u, w)
    @SVector [x[2],
              (u[1] / ((model.mass * model.lc * model.lc))
                - model.g * sin(x[1]) / model.lc
                - model.b * x[2] / (model.mass * model.lc * model.lc))]
end

function k(model::Pendulum, x)
    @SVector [model.lc * sin(x[1]),
              -1.0 * model.lc * cos(x[1])]
end

n, m, d = 2, 1, 0
model = Pendulum(n, m, d, 1.0, 0.1, 0.5, 0.25, 9.81)
