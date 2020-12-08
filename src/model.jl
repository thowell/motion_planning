abstract type Integration end

abstract type Model{I <: Integration, T <: Time} end

struct TemplateModel{I, T} <: Model{I, T}
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

model = TemplateModel{Integration, Time}(0, 0, 0)

state_output(model, x) = x
control_output(model, u) = u

include_model(str::String) = include(joinpath(pwd(), "models", str * ".jl"))

"""
	propagate dynamics with implicit integrator
	- LM
"""

function propagate_dynamics(model, x, u, w, h, t)
	# L-M
	α = 1.0
	reg = 1.0e-8
	y = copy(x)

	res(z) = fd(model, z, x, u, w, h, t)
	merit(z) = res(z)' * res(z)

	iter = 0

	while iter < 100
		me = merit(y)
		r = res(y)
		∇r = ForwardDiff.jacobian(res, y)

		_H = ∇r' * ∇r
		Is = Diagonal(diag(_H))
		H = (_H + reg * Is)

		pd_iter = 0
		while !isposdef(Hermitian(Array(H)))
			reg *= 2.0
			H = (_H + reg * Is)
			pd_iter += 1

			if pd_iter > 100 || reg > 1.0e12
				@error "regularization failure"
			end
		end

		Δy = -1.0 * H \ (∇r' * r)

		ls_iter = 0
		while merit(y + α * Δy) > me + 1.0e-4 * r' * (α * Δy)
			α *= 0.5
			reg = reg
			ls_iter += 1

			if ls_iter > 100 || reg > 1.0e12
				@error "line search failure"
			end
		end

		y .+= α * Δy
		α = min(1.2 * α, 1.0)
		reg = 0.5 * reg

		iter += 1

		norm(α * Δy, Inf) < 1.0e-6 && (return y)
		norm(r, Inf) < 1.0e-8 && (return y)
	end

	@warn "gauss-newton failed"
	return y
end

# function propagate_dynamics(model, x, u, w, h, t)
#     _fd(z) = fd(model, z, x, u, w, h, t)
#
#     y = copy(x)
#     r = _fd(y)
#
#     iter = 0
#
#     while norm(r, 2) > 1.0e-8 && iter < 10
#         ∇r = ForwardDiff.jacobian(_fd, y)
#
#         Δy = -1.0 * ∇r \ r
#
#         α = 1.0
#
# 		iter_ls = 0
#         while α > 1.0e-8 && iter_ls < 10
#             ŷ = y + α * Δy
#             r̂ = _fd(ŷ)
#
#             if norm(r̂) < norm(r)
#                 y = ŷ
#                 r = r̂
#                 break
#             else
#                 α *= 0.5
# 				iter_ls += 1
#             end
#
# 			if iter_ls == 10
# 				@warn "line search failed"
# 				# print("y: $y")
# 				# print("x: $x")
# 				# print("u: $u")
# 				# print("w: $w")
# 				# println("l2: $(norm(r))")
# 				# println("linf: $(norm(r, Inf))")
# 			end
#         end
#
#         iter += 1
#     end
#
# 	if iter == 10
# 		@warn "newton failed"
# 		# print("y: $y")
# 		# print("x: $x")
# 		# print("u: $u")
# 		# print("w: $w")
# 		# println("l2: $(norm(r))")
# 		# println("linf: $(norm(r, Inf))")
# 		y = fd(model, x, u, w, h, t)
# 	end
#     # @show norm(r)
#     # @show iter
#     return y
# end

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

"""
	get gravity compensating torques
"""
function initial_torque(model, q1, h)
	x = [q1; q1]
    _fd(z) = fd(model, x, x, z, zeros(model.d), h, 0)

    y = ones(model.m)
    r = _fd(y)

    iter = 0

    while norm(r, 2) > 1.0e-8 && iter < 10
        ∇r = ForwardDiff.jacobian(_fd, y)

        Δy = -1.0 * ∇r \ r

        α = 1.0

		iter_ls = 0
        while α > 1.0e-8 && iter_ls < 10
            ŷ = y + α * Δy
            r̂ = _fd(ŷ)

            if norm(r̂) < norm(r)
                y = ŷ
                r = r̂
                break
            else
                α *= 0.5
				iter_ls += 1
            end

			if iter_ls == 10
				@warn "line search failed"
				# print("y: $y")
				# print("x: $x")
				# print("u: $u")
				# print("w: $w")
				# println("l2: $(norm(r))")
				# println("linf: $(norm(r, Inf))")
			end
        end

        iter += 1
    end

	if iter == 10
		@warn "newton failed"
	end
    # @show norm(r)
    # @show iter
    return y
end

# Model conversions

"""
	free final time model
"""
function free_time_model(model::Model{I, FixedTime}) where I <: Integration
	model_ft = typeof(model).name.wrapper{I, FreeTime}([f == :m ? getfield(model,f) + 1 : getfield(model,f)
	 	for f in fieldnames(typeof(model))]...)
	return model_ft
end

function free_time_model(model::Model{I, FreeTime}) where I <: Integration
	@warn "Model is already free time"
	return model
end

"""
	no slip model
"""
function no_slip_model(model)
	# modify parameters
	m = model.nu + model.nc + model.nb + model.ns
	idx_ψ = (1:0)
	idx_η = (1:0)
	idx_s = model.nu + model.nc + model.nb .+ (1:model.ns)

	# assemble update parameters
	params = []
	for f in fieldnames(typeof(model))
		if f == :m
			push!(params, m)
		elseif f == :idx_ψ
			push!(params, idx_ψ)
		elseif f == :idx_η
			push!(params, idx_η)
		elseif f == :idx_s
			push!(params, idx_s)
		else
			push!(params, getfield(model,f))
		end
	end

	return typeof(model)(params...)
end
