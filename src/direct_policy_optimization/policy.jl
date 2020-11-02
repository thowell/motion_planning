struct PolicyConstraints <: Constraints
	n
	ineq
	control_dim
end

function policy_constraints(model, T; control_dim = model.m)
	n = model.m * (T - 1)
	ineq = (1:0)

	return PolicyConstraints(n, ineq, control_dim)
end

function constraints!(c, Θ, τ_nom, τ, con::PolicyConstraints, model_nom, model_sample,
		idx_nom, idx_sample, idx_policy, h, T)

	p = con.control_dim

	for t = 1:T-1
		θ = view(Θ, idx_policy[t])

		x̄ = view(τ_nom, idx_nom.x[t])
		ū = view(τ_nom, idx_nom.u[t])

		x = view(τ, idx_sample.x[t])
		u = view(τ, idx_sample.u[t])

		c[(t - 1) * p .+ (1:p)] = policy(model_sample, θ, x, x̄, ū) - view(u, 1:p)
	end
    return nothing
end

function constraints_jacobian!(∇c, Θ, τ_nom, τ, con::PolicyConstraints,
		model_nom, model_sample,
		idx_nom, idx_sample, idx_policy, h, T;
		shift_sample = 0, shift_policy = 0)

	p = con.control_dim

	shift = 0

	for t = 1:T-1
		θ = view(Θ, idx_policy[t])

		x̄ = view(τ_nom, idx_nom.x[t])
		ū = view(τ_nom, idx_nom.u[t])

		x = view(τ, idx_sample.x[t])
		u = view(τ, idx_sample.u[t])

		pθ(y) = policy(model_sample, y, x, x̄, ū) - view(u, 1:p)
		px(y) = policy(model_sample, θ, y, x̄, ū) - view(u, 1:p)
		px̄(y) = policy(model_sample, θ, x, y, ū) - view(u, 1:p)
		pū(y) = policy(model_sample, θ, x, x̄, y) - view(u, 1:p)
		pu(y) = policy(model_sample, θ, x, x̄, ū) - view(y, 1:p)

		# c[(t - 1) * p .+ (1:p)] = policy(model_sample, θ, x, x̄, ū) - view(u, 1:p)

		r_idx = (t - 1) * p .+ (1:p)
		c_idx = shift_policy .+ idx_policy[t]
		len = length(r_idx) * length(c_idx)
		∇c[shift .+ (1:len)] = vec(ForwardDiff.jacobian(pθ, θ))
		shift += len

		c_idx = shift_sample .+ idx_sample.x[t]
		len = length(r_idx) * length(c_idx)
		∇c[shift .+ (1:len)] = vec(ForwardDiff.jacobian(px, x))
		shift += len

		c_idx = idx_nom.x[t]
		len = length(r_idx) * length(c_idx)
		∇c[shift .+ (1:len)] = vec(ForwardDiff.jacobian(px̄, x̄))
		shift += len

		c_idx = idx_nom.u[t]
		len = length(r_idx) * length(c_idx)
		∇c[shift .+ (1:len)] = vec(ForwardDiff.jacobian(pū, ū))
		shift += len

		c_idx = shift_sample .+ idx_sample.u[t]
		len = length(r_idx) * length(c_idx)
		∇c[shift .+ (1:len)] = vec(ForwardDiff.jacobian(pu, u))
		shift += len

	end

	return nothing
end

function constraints_sparsity(con::PolicyConstraints, model_nom, model_sample,
		idx_nom, idx_sample, idx_policy, T;
		shift_con = 0, shift_sample = 0, shift_policy = 0)

	row = []
    col = []

	p = con.control_dim

	for t = 1:T-1
		r_idx = shift_con + (t - 1) * p .+ (1:p)

		c_idx = shift_policy .+ idx_policy[t]
		row_col!(row, col, r_idx, c_idx)

		c_idx = shift_sample .+ idx_sample.x[t]
		row_col!(row, col, r_idx, c_idx)

		c_idx = idx_nom.x[t]
		row_col!(row, col, r_idx, c_idx)

		c_idx = idx_nom.u[t]
		row_col!(row, col, r_idx, c_idx)

		c_idx = shift_sample .+ idx_sample.u[t]
		row_col!(row, col, r_idx, c_idx)
	end

    return collect(zip(row, col))
end


# linear state-feedback policy
function policy(model, θ, x, x̄, ū)
	ū - reshape(θ, model.m, model.n) * (x - x̄)
end
