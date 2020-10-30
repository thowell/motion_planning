"""
    explicit midpoint integrator
"""
function midpoint(model, x, u, w, h)
    x + h * f(model, x + 0.5 * h * f(model, x, u, w), u, w)
end

"""
    implicit midpoint integrator
"""
function midpoint_implicit(model, x⁺, x, u, w, h)
    x⁺ - (x + h * f(model, 0.5 * (x + x⁺), u, w))
end

"""
    explicit RK3 integrator
"""
function rk3(model, z, u, w, h)
    k1 = k2 = k3 = zero(z)
    k1 = h * f(model, z, u, w)
    k2 = h * f(model, z + 0.5 * k1, u, w)
    k3 = h * f(model, z - k1 + 2.0 * k2, u, w)
    z + (k1 + 4.0 * k2 + k3) / 6.0
end

"""
    discrete explicit dynamics
"""
function fd(model, x, u, w, h, t)
    midpoint(model, x, u, w, h)
end

"""
    discrete implicit dynamics
"""
function fd(model, x⁺, x, u, w, h, t)
    midpoint_implicit(model, x⁺, x, u, w, h)
end
