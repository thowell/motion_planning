using Convex, ECOS

"""
	contact constraints
		-differentiate through second-order cone program to get friction
"""
function friction_socp(model, x⁺, u, h) # version for particle
	q3 = view(x⁺, model.nq .+ (1:model.nq))
	q2 = view(x⁺, 1:model.nq)
	λ = view(u, model.idx_λ)

	v = (q3 - q2)[1:2] ./ h #
	y = model.μ * λ

	"Convex.jl"
	b = Variable(2)
	prob = minimize(v' * b)
	prob.constraints += norm(b) <= y
	solve!(prob, ECOS.Optimizer(verbose = false));

	# @show prob.status
	# @show b.value
	# @show prob.constraints[1].dual
	# prob.optval

	# # norms
	# _norm(x) = sqrt(x' * x)
	function vec_norm(x)
		if norm(x) == 0.0
			return ones(length(x)) ./ norm(ones(length(x)))
		else
			x ./ norm(x)
		end
	end
	# function d_vec_norm(x)
	# 	if norm(x) == 0.0
	# 		y = 1.0 * ones(length(x))#./norm(ones(length(x)))
	# 		return (I - y * y' / (y' * y)) /_norm(y)
	# 	else
	# 		(I - x * x' / (x' * x)) /_norm(x)
	# 	end
	# end
	function r(z,θ)
		b = z[1:2]
		ψ = z[3]
		y = θ[1]
		v = θ[2:3]

		return [norm(b) * v + ψ * b;
			    ψ * (y - norm(b))]
	end

	function drz(z,θ)
		b = z[1:2]
		ψ = z[3]
		y = θ[1]
		v = θ[2:3]

		return [v * vec_norm(b)' + ψ * I b;
			    -ψ * vec_norm(b)' (y - norm(b))]
	end

	function drθ(z,θ)
		b = z[1:2]
		ψ = z[3]
		y = θ[1]
		v = θ[2:3]
		return Array([zeros(2) Diagonal(norm(b) * ones(2));
					  ψ zeros(1, 2)])
	end

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
	# drz(z, θ)
	# drθ(z, θ)
	# norm(drdz - drz(z, θ))
	# norm(drdθ - drθ(z, θ))
	# rank(drdz)
	# eigen(drdz).values
	#
	# rank(drz(z, θ))
	# eigen(drz(z, θ)).values

	# x1 = (-drdz \ drdθ)[1:2,:]
	# x3 = (-(drdz' * drdz) \ (drdz' * drdθ))[1:2,:]

	ρ = 1.0e-5
	sol = (-(drdz' * drdz + ρ * I) \ (drdz' * drdθ))[1:2,:]
	# x4 = (-drdz' * ((drdz * drdz' + ρ * I) \ drdθ))[1:2,:]
	dλ = sol[:, 1] .* model.μ
	dq2 = -1.0 * sol[:, 2:3] ./ h
	dq3 = sol[:, 2:3] ./ h
	# print(b)
	return b, dλ, dq2, dq3
end

struct ContactDiffConstraints <: Constraints
	n
	ineq
end

function contact_diff_constraints(model, T)
	n = model.nc * (T + 1) + model.nb * (T - 1) + 1 * (T - 1)
	ineq = con_ineq_contact(model, T)

	return ContactDiffConstraints(n, ineq)
end

function constraints!(c, Z, con::ContactDiffConstraints, model, idx, h, T)
	shift = 0

	# signed-distance function
	for t = 1:T
		x = view(Z, idx.x[t])

		if t == 1
			q = view(x, 1:model.nq)
			c[1:model.nc] = ϕ_func(model, q)
		end

		q⁺ = view(x, model.nq .+ (1:model.nq))
		c[model.nc + (t-1) * model.nc .+ (1:model.nc)] = ϕ_func(model, q⁺)
	end

	shift += model.nc * (T + 1)

	# maximum dissipation
	for t = 1:T-1
		x⁺ = view(Z, idx.x[t+1])
		u = view(Z, idx.u[t])
		b = view(u, model.idx_b)

		b_sol, dλ, dq2, dq3 = friction_socp(model, x⁺, u, h)
		c[shift + (t - 1) * model.nb .+ (1:model.nb)] = b_sol - b
	end

	shift += model.nb * (T - 1)

	# impact complementarity
	for t = 1:T-1
		x⁺ = view(Z, idx.x[t + 1])
		u = view(Z, idx.u[t])

		q = view(x⁺, model.nq .+ (1:model.nq))
		s = view(u, model.idx_s)
		λ = view(u, model.idx_λ)

		c[shift + (t-1) * 1 + 1] = s[1] - (λ' * ϕ_func(model, q))[1]
	end

	shift += (T - 1)

    return nothing
end

function constraints_jacobian!(∇c, Z, con::ContactDiffConstraints, model, idx, h, T)
	shift = 0
	c_shift = 0

	# signed-distance function
	ϕ(y) = ϕ_func(model, y)

	for t = 1:T
		x = view(Z, idx.x[t])

		if t == 1
			q = view(x, 1:model.nq)
			r_idx = 1:model.nc
			c_idx = idx.x[t][1:model.nq]
			len = length(r_idx) * length(c_idx)
			∇c[shift .+ (1:len)] = vec(ForwardDiff.jacobian(ϕ, q))
			shift += len
		end

		q⁺ = view(x, model.nq .+ (1:model.nq))
		r_idx = model.nc + (t-1) * model.nc .+ (1:model.nc)
		c_idx = idx.x[t][model.nq .+ (1:model.nq)]
		len = length(r_idx) * length(c_idx)
		∇c[shift .+ (1:len)] = vec(ForwardDiff.jacobian(ϕ, q⁺))
		shift += len
	end

	c_shift += model.nc * (T + 1)

	# friction socp
	for t = 1:T-1
		x⁺ = view(Z, idx.x[t+1])
		u = view(Z, idx.u[t])

		b_sol, dλ, dq2, dq3 = friction_socp(model, x⁺, u, h)

		r_idx = c_shift + (t - 1) * model.nb .+ (1:model.nb)
		c_idx = idx.x[t + 1][collect([(1:2)..., (4:5)...])]
		len = length(r_idx) * length(c_idx)
		∇c[shift .+ (1:len)] = vec([dq2; dq3])
		shift += len

		c_idx = idx.u[t][model.idx_λ]
		len = length(r_idx) * length(c_idx)
		∇c[shift .+ (1:len)] = vec(dλ)
		shift += len

		c_idx = idx.u[t][model.idx_b]
		len = length(r_idx) * length(c_idx)
		∇c[shift .+ (1:len)] = vec(Diagonal(-1.0 * ones(model.nb)))
		shift += len
	end

	c_shift += model.nb * (T - 1)

	# impact complementarity
	for t = 1:T-1
		x⁺ = view(Z, idx.x[t + 1])
		u = view(Z, idx.u[t])

		q = view(x⁺, model.nq .+ (1:model.nq))
		s = view(u, model.idx_s)
		λ = view(u, model.idx_λ)

		cq(y) = s - [λ' * ϕ_func(model, y)]
		cu(y) = y[model.idx_s] - [y[model.idx_λ]' * ϕ_func(model, q)]

		r_idx = c_shift + (t-1) * 1 + 1
		c_idx = idx.x[t + 1][model.nq .+ (1:model.nq)]
		len = length(r_idx) * length(c_idx)
		∇c[shift .+ (1:len)] = vec(ForwardDiff.jacobian(cq, q))
		shift += len

		c_idx = idx.u[t]
		len = length(r_idx) * length(c_idx)
		∇c[shift .+ (1:len)] = vec(ForwardDiff.jacobian(cu, u))
		shift += len
	end

	c_shift += (T - 1)

	return nothing
end

function constraints_sparsity(con::ContactDiffConstraints, model, idx, T;
	shift_row = 0, shift_col = 0)

	row = []
    col = []

	con_shift = 0

	# signed-distance function
	for t = 1:T

		if t == 1
			r_idx = shift_row .+ (1:model.nc)
			c_idx = shift_col .+ idx.x[t][1:model.nq]
			row_col!(row, col, r_idx, c_idx)
		end

		r_idx = shift_row + model.nc + (t-1) * model.nc .+ (1:model.nc)
		c_idx = shift_col .+ idx.x[t][model.nq .+ (1:model.nq)]
		row_col!(row, col, r_idx, c_idx)
	end

	con_shift += model.nc * (T + 1)

	# friction socp
	for t = 1:T-1
		r_idx = shift_row + con_shift + (t - 1) * model.nb .+ (1:model.nb)
		c_idx = shift_col .+ idx.x[t + 1][collect([(1:2)..., (4:5)...])]
		row_col!(row, col, r_idx, c_idx)

		c_idx = shift_col .+ idx.u[t][model.idx_λ]
		row_col!(row, col, r_idx, c_idx)

		c_idx = idx.u[t][model.idx_b]
		row_col!(row, col, r_idx, c_idx)
	end

	con_shift += model.nb * (T - 1)

	# impact complementarity
	for t = 1:T-1
		r_idx = shift_row + con_shift + (t-1) * 1 + 1
		c_idx = shift_col .+ idx.x[t + 1][model.nq .+ (1:model.nq)]
		row_col!(row, col, r_idx, c_idx)

		c_idx = shift_col .+ idx.u[t]
		row_col!(row, col, r_idx, c_idx)
	end

	con_shift += (T - 1)

    return collect(zip(row, col))
end

function con_ineq_contact(model, T)

	con_ineq = []
	shift = 0

	# signed-distance function
	push!(con_ineq, [i for i = 1:model.nc * (T + 1)])
	shift += model.nc * (T + 1)

	# maximum dissipation
	# push!(con_ineq, [i for i = shift .+ (1:model.nb * (T - 1))])
	shift += model.nb * (T - 1)

	# impact complementarity
	push!(con_ineq, [i for i = shift .+ (1:(T - 1))])
	shift += (T - 1)

    return vcat(con_ineq...)
end
