using LinearAlgebra, ForwardDiff
using Convex, SCS, ECOS

"""
 min v'b
 st norm(b) <= y
"""

# parameters
v = [100.0; 100.0]
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



function r(z,θ)
	b = z[1:2]
	ψ = z[3]
	y = θ[1]
	v = θ[2:3]

	return [norm(b) * v + ψ * b;
		    ψ * (y - norm(b))]
end

# function drz(z,θ)
# 	b = z[1:2]
# 	ψ = z[3]
# 	y = θ[1]
# 	v = θ[2:3]
#
# 	return [ψ*d_vec_norm(b) vec_norm(b);
# 		    -ψ*vec_norm(b)' (y - _norm(b))]
# end
#
# function drθ(z,θ)
# 	return Array([zeros(2) Diagonal(ones(2));
# 			ψ zeros(1,2)])
# end

# if y == 0.0
# 	z = [zeros(2); ψ]
# else
# 	z = [b.value; ψ]
# end
z = [b.value; prob.constraints[1].dual]
θ = [y; v]
rz(x) = r(x, θ)
rθ(x) = r(z, x)

drdz = ForwardDiff.jacobian(rz, z)
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
