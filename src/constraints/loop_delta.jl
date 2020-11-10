"""
	loop constraints
"""

struct LoopDelta <: Constraints
	n
	ineq
	x_idx
	t_idx1
	t_idx2
end

function loop_delta_constraints(model::BipedPinned, x_idx, t_idx1, t_idx2)
	n = length(x_idx)
	ineq = (1:0)
	return LoopDelta(n, ineq, x_idx, t_idx1, t_idx2)
end

function constraints!(c, Z, con::LoopDelta, model::BipedPinned, idx, h, T)
	n = con.n
	a = view(Z, idx.x[con.t_idx1])[con.x_idx]
	b = Δ(view(Z, idx.x[con.t_idx2]))[con.x_idx]

	c[1:n] = b - a

    return nothing
end

function constraints_jacobian!(∇c, Z, con::LoopDelta, model, idx, h, T)
	n = con.n

	shift = 0

	r_idx = 1:con.n
	c_idx = idx.x[con.t_idx2][con.x_idx]
	len = length(r_idx) * length(c_idx)
	∇c[shift .+ (1:len)] = vec(ForwardDiff.jacobian(Δ,
		view(Z, idx.x[con.t_idx2]))[con.x_idx, con.x_idx])
	shift += len

	c_idx = idx.x[con.t_idx1][con.x_idx]
	len = length(r_idx) * length(c_idx)
	∇c[shift .+ (1:len)] = vec(Diagonal(-1.0 * ones(n)))
	shift += len
	return nothing
end

function constraints_sparsity(con::LoopDelta, model, idx, T;
		shift_row = 0, shift_col = 0)
	row = []
    col = []

	n = con.n

	r_idx = shift_row .+ (1:n)

	c_idx = shift_col .+ idx.x[con.t_idx2][con.x_idx]
	row_col!(row, col, r_idx, c_idx)

	c_idx = shift_col .+ idx.x[con.t_idx1][con.x_idx]
	row_col!(row, col, r_idx, c_idx)

    return collect(zip(row, col))
end
