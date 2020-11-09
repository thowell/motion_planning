abstract type Model end

struct TemplateModel <: Model
	n::Int # state dimension
	m::Int # control dimension
	d::Int # disturbance dimension

	# additional model parameters ...
end

function f(model::TemplateModel, x, u, w)
	# continuous-time dynamics
	nothing
end

function k(model::TemplateModel, x)
	# kinematics
	nothing
end

model = TemplateModel(0, 0, 0)

state_output(model, x) = x
state_output_idx(model, idx) = idx
control_output(model, u) = u

"""
	propagate dynamics with implicit integrator
"""
function propagate_dynamics(model, x, u, w, h, t)
    d(z) = fd(model, z, x, u, w, h, t)

    y = copy(x)
    r = d(y)

    iter = 0

    while norm(r, 2) > 1.0e-8 && iter < 10
        ∇r = ForwardDiff.jacobian(d, y)
        Δy = -1.0 * ∇r \ r

        α = 1.0

        while α > 1.0e-8
            ŷ = y + α * Δy
            r̂ = d(ŷ)
            if norm(r̂) < norm(r)
                y = ŷ
                r = r̂
                break
            else
                α *= 0.5
            end
        end

        iter += 1
    end

    # @show norm(r)
    # @show iter
    return y
end

function propagate_dynamics_jacobian(model, x, u, w, h, t)
	y = propagate_dynamics(model, x, u, w, h, t)

    dy(z) = fd(model, z, x, u, w, h, t)
	dx(z) = fd(model, y, z, u, w, h, t)
	du(z) = fd(model, y, x, z, w, h, t)

	Dy = ForwardDiff.jacobian(dy, y)
	A = -1.0 * Dy \ ForwardDiff.jacobian(dx, x)
	B = -1.0 * Dy \ ForwardDiff.jacobian(du, u)

	return y, A, B
end
