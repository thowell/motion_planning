using LinearAlgebra, ForwardDiff
using Convex, SCS, ECOS

"""
	projection onto second-order cone
"""
function Πsoc(v, s)
	if norm(v) <= -s
		# @warn "below cone"
		return zero(v), 0.0
	elseif norm(v) <= s
		# @warn "in cone"
		return v, s
	elseif norm(v) > abs(s)
		# @warn "outside cone"
		a = 0.5 * (1.0 + s / norm(v))
		return a * v, a * norm(v)
	else
		@warn "soc projection error"
		return zero(v), 0.0
	end
end

function projection_difference(v, s)
	vcat(Πsoc(v, s)...) - [v; s]
end

# projection_difference(ones(2), 1.0)

"""
 min v'b
 st norm(b) <= y
"""

# parameters
v = [1.0; 1.0e-1]
ψ = norm(v)
y = 1.0

"Convex.jl"
b = Variable(2)
prob = minimize(v' * b)
prob.constraints += norm(b) <= y
@time solve!(prob, ECOS.Optimizer)

@show prob.status
@show b.value
@show prob.constraints[1].dual
prob.optval

function lagrangian(v, y, b, ψ)
	v' * b + ψ' * projection_difference(b, y) + 0.5 * projection_difference(b, y)' * projection_difference(b, y)
end

function r(z, θ)
	b = z[1:2]
	ψ = z[3:5]
	y = θ[1]
	v = θ[2:3]

	lb(w) = lagrangian(v, y, w, ψ)
	# lψ(w) = lagrangian(v, y, b, w)

	# return ForwardDiff.gradient(lb, b)
	return [ForwardDiff.gradient(lb, b);
			projection_difference(b, y)]

end

θ = [y; v]
rz(x) = r(x, θ)

sol = levenberg_marquardt(rz, ones(5))
rθ(x) = r(sol, x)

drdz = ForwardDiff.jacobian(rz, sol)
drdθ = ForwardDiff.jacobian(rθ, θ)

# norm(drdz - drz(z, θ))
# norm(drdθ - drθ(z, θ))
rank(drdz)
# eigen(drdz).values
#
# rank(drz(z, θ))
# eigen(drz(z, θ)).values

x1 = (-drdz \ drdθ)[1:2,:]
x3 = (-(drdz' * drdz) \ (drdz' * drdθ))[1:2,:]

ρ = 1.0e-8
x2 = (-(drdz' * drdz + ρ * I) \ (drdz' * drdθ))[1:2,:]
x4 = (-drdz' * ((drdz * drdz' + ρ * I) \ drdθ))[1:2,:]

# x2 = (-drz(z,θ) \ drθ(z,θ))[1:2,:]
# x4 = (-(drz(z,θ)' * drz(z,θ)) \ (drz(z, θ)' * drθ(z, θ)))[1:2, :]
#
