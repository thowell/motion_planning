"""
	contact (no slip) constraints
"""

struct ContactNoSlip <: Constraints
	n
	ineq
end

function contact_no_slip_constraints(model, T)
	n = model.nc * (T + 1) + model.nc * (T - 1) + 2 * (T - 1)
	ineq = con_ineq_contact(model, T)

	return ContactNoSlip(n, ineq)
end

function constraints!(c, Z, con::ContactNoSlip, model, idx, h, T)
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

	# friction cone
	for t = 1:T-1
		u = view(Z, idx.u[t])
		c[shift + (t - 1) * model.nc .+ (1:model.nc)] = friction_cone(model, u)
	end

	shift += model.nc * (T - 1)

	# no slip
	for t = 1:T-1
		x⁺ = view(Z, idx.x[t+1])
		u = view(Z, idx.u[t])
		c[shift + (t - 1) + 1] = no_slip(model, x⁺, u, h)
	end

	shift += (T - 1)

    return nothing
end

function constraints_jacobian!(∇c, Z, con::ContactNoSlip, model, idx, h, T)
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

	# impact complementarity
	for t = 1:T-1
		x⁺ = view(Z, idx.x[t + 1])
		u = view(Z, idx.u[t])

		q = view(x⁺, model.nq .+ (1:model.nq))
		s = view(u, model.idx_s)
		λ = view(u, model.idx_λ)

		cq(y) = s - [λ' * ϕ_func(model, y)]
		cu(y) = y[model.idx_s] - [y[model.idx_λ]' * ϕ_func(model, q)]

		print()
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

	# friction cone
	fc(y) = friction_cone(model, y)
	for t = 1:T-1
		u = view(Z, idx.u[t])
		r_idx = c_shift + (t - 1) * model.nc .+ (1:model.nc)
		c_idx = idx.u[t]
		len = length(r_idx) * length(c_idx)
		∇c[shift .+ (1:len)] = vec(ForwardDiff.jacobian(fc, u))
		shift += len
	end

	c_shift += model.nc * (T - 1)

	# no slip
	for t = 1:T-1
		x⁺ = view(Z, idx.x[t+1])
		u = view(Z, idx.u[t])

		nsx(y) = no_slip(model, y, u, h)
		nsu(y) = no_slip(model, x⁺, y, h)

		r_idx = c_shift + (t - 1) .+ (1:1)
		c_idx = idx.x[t + 1]
		len = length(r_idx) * length(c_idx)
		∇c[shift .+ (1:len)] = ForwardDiff.gradient(nsx, x⁺)
		shift += len

		c_idx = idx.u[t]
		len = length(r_idx) * length(c_idx)
		∇c[shift .+ (1:len)] = ForwardDiff.gradient(nsu, u)
		shift += len
	end

	c_shift += (T - 1)

	return nothing
end

function constraints_sparsity(con::ContactNoSlip, model, idx, T;
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

	# impact complementarity
	for t = 1:T-1
		r_idx = shift_row + con_shift + (t-1) * 1 + 1
		c_idx = shift_col .+ idx.x[t + 1][model.nq .+ (1:model.nq)]
		row_col!(row, col, r_idx, c_idx)

		c_idx = shift_col .+ idx.u[t]
		row_col!(row, col, r_idx, c_idx)
	end

	con_shift += (T - 1)

	# friction cone
	for t = 1:T-1
		r_idx = shift_row + con_shift + (t - 1) * model.nc .+ (1:model.nc)
		c_idx = shift_col .+ idx.u[t]
		row_col!(row, col, r_idx, c_idx)
	end

	con_shift += model.nc * (T - 1)

	# no slip
	for t = 1:T-1
		r_idx = shift_row + con_shift + (t - 1) .+ (1:1)
		c_idx = shift_col .+ idx.x[t + 1]
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

	# impact complementarity
	push!(con_ineq, [i for i = shift .+ (1:(T - 1))])
	shift += (T - 1)

	# friction cone
	push!(con_ineq, [i for i = shift .+ (1:model.nc * (T - 1))])
	shift += model.nc * (T - 1)

	# no slip
	# push!(con_ineq, [i for i = shift .+ (1:(T - 1))])
	shift += (T - 1)

    return vcat(con_ineq...)
end

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
