"""
	loop constraints
"""

struct LoopConstraints <: Constraints
	n
	ineq
	idx_1
	idx_2
end

function loop_constraints(model, idx1, idx2)
	n = model.n
	ineq = (1:0)
	return LoopConstraints(n, ineq, idx1, idx2)
end

function constraints!(c, Z, con::LoopConstraints, model, idx, h, T)
	n = model.n
	a = view(Z, idx.x[con.idx_1])
	b = view(Z, idx.x[con.idx_2])

	c[1:n] = b - a

    return nothing
end

function constraints_jacobian!(∇c, Z, con::LoopConstraints, model, idx, h, T)
	n = model.n
	shift = 0

	r_idx = 1:n
	c_idx = idx.x[con.idx_2]
	len = length(r_idx) * length(c_idx)
	∇c[shift .+ (1:len)] = vec(Diagonal(ones(n)))
	shift += len

	c_idx = idx.x[con.idx_1]
	len = length(r_idx) * length(c_idx)
	∇c[shift .+ (1:len)] = vec(Diagonal(-1.0 * ones(n)))
	shift += len
	return nothing
end

function constraints_sparsity(con::LoopConstraints, model, idx, T;
		r_shift = 0)
	row = []
    col = []

	n = model.n

	r_idx = r_shift .+ (1:n)

	c_idx = idx.x[con.idx_2]
	row_col!(row, col, r_idx, c_idx)

	c_idx = idx.x[con.idx_1]
	row_col!(row, col, r_idx, c_idx)

    return collect(zip(row, col))
end
