struct FreeTimeConstraints <: Constraints
	n::Int
	ineq
end

function free_time_constraints(T)
	n = T - 2
	ineq = (1:0)

	return FreeTimeConstraints(n, ineq)
end

function constraints!(c, Z, con::FreeTimeConstraints, model, idx, h, T)
	for t = 1:T-2
		h = view(Z,idx.u[t])[end]
		h⁺ = view(Z,idx.u[t + 1])[end]
		c[t] = h⁺ - h
	end

    return nothing
end

function constraints_jacobian!(∇c, Z, con::FreeTimeConstraints, model, idx, h, T)
	shift = 0
	for t = 1:T-2
		r_idx = (t:t)

		c_idx = (idx.u[t][end]:idx.u[t][end])
		len = 1
		∇c[shift + len] = -1.0
		shift += 1

		c_idx = (idx.u[t + 1][end]:idx.u[t + 1][end])
		len = 1
		∇c[shift + len] = 1.0
		shift += 1
	end

	return nothing
end

function constraints_sparsity(con::FreeTimeConstraints, model, idx, T;
	shift_row = 0, shift_col = 0)
    row = []
    col = []

	for t = 1:T-2
		r_idx = shift_row .+ (t:t)

		c_idx = shift_col .+ (idx.u[t][end]:idx.u[t][end])
		row_col!(row, col, r_idx, c_idx)

		c_idx = shift_col .+ (idx.u[t + 1][end]:idx.u[t + 1][end])
		row_col!(row, col, r_idx, c_idx)
	end
    return collect(zip(row, col))
end
