abstract type Policy end

struct LinearFeedback <: Policy
	input
	output
	idx_input
	idx_input_nom
	idx_output
	p
end

function linear_feedback(input, output;
		idx_input = (1:input),
		idx_input_nom = (1:input),
		idx_output = (1:output))

	return LinearFeedback(input, output,
		idx_input, idx_input_nom, idx_output,
		input * output)
end

# linear state-feedback policy
function eval_policy(policy::LinearFeedback, θ, x, x̄, ū)
	view(ū, policy.idx_output) - reshape(θ, policy.output, policy.input) * (view(x, policy.idx_input) - view(x̄, policy.idx_input_nom))
end

struct PolicyConstraint <: Constraints
	n
	ineq
	control_dim
end

function policy_constraint(model, T; control_dim = model.m)
	n = control_dim * (T - 1)
	ineq = (1:0)

	return PolicyConstraint(n, ineq, control_dim)
end

struct PolicyConstraints <: Constraints
	con
	n
	ineq
end

function policy_constraints(prob, N;
		control_dim = prob.mean.model.m)

	T = prob.nom.T

	con = [policy_constraint(prob.mean.model, T,
		control_dim = control_dim)]
	n = con[end].n

	for i = 1:N
		push!(con, policy_constraint(prob.sample[i].model, T,
			control_dim = control_dim))
		n += con[i + 1].n
	end

	ineq = (1:0)

	return PolicyConstraints(con, n, ineq)
end

function constraints!(c, Θ, τ_nom, τ_sample, con::PolicyConstraint,
		policy::Policy,
		model_nom, model_sample,
		idx_τ_nom, idx_τ_sample, idx_θ,
		h, T)

	p = con.control_dim

	for t = 1:T-1
		θ = view(Θ, idx_θ[t])

		x̄ = view(τ_nom, idx_τ_nom.x[t])
		ū = view(τ_nom, idx_τ_nom.u[t])

		x = view(τ_sample, idx_τ_sample.x[t])
		u = view(τ_sample, idx_τ_sample.u[t])

		c[(t - 1) * p .+ (1:p)] = eval_policy(policy, θ, x, x̄, ū) - view(u, 1:p)
	end
    return nothing
end

function constraints_jacobian!(∇c, Θ, τ_nom, τ_sample, con::PolicyConstraint,
		policy::Policy,
		model_nom, model_sample,
		idx_τ_nom, idx_τ_sample, idx_θ,
		h, T)

	p = con.control_dim

	shift = 0

	for t = 1:T-1
		θ = view(Θ, idx_θ[t])

		x̄ = view(τ_nom, idx_τ_nom.x[t])
		ū = view(τ_nom, idx_τ_nom.u[t])

		x = view(τ_sample, idx_τ_sample.x[t])
		u = view(τ_sample, idx_τ_sample.u[t])

		pθ(y) = eval_policy(policy, y, x, x̄, ū) - view(u, 1:p)
		px(y) = eval_policy(policy, θ, y, x̄, ū) - view(u, 1:p)
		px̄(y) = eval_policy(policy, θ, x, y, ū) - view(u, 1:p)
		pū(y) = eval_policy(policy, θ, x, x̄, y) - view(u, 1:p)
		pu(y) = eval_policy(policy, θ, x, x̄, ū) - view(y, 1:p)

		# c[(t - 1) * p .+ (1:p)] = policy(model_sample, θ, x, x̄, ū) - view(u, 1:p)

		r_idx = (t - 1) * p .+ (1:p)
		c_idx = idx_θ[t]
		len = length(r_idx) * length(c_idx)
		∇c[shift .+ (1:len)] = vec(ForwardDiff.jacobian(pθ, θ))
		shift += len

		c_idx = idx_τ_sample.x[t]
		len = length(r_idx) * length(c_idx)
		∇c[shift .+ (1:len)] = vec(ForwardDiff.jacobian(px, x))
		shift += len

		c_idx = idx_τ_nom.x[t]
		len = length(r_idx) * length(c_idx)
		∇c[shift .+ (1:len)] = vec(ForwardDiff.jacobian(px̄, x̄))
		shift += len

		c_idx = idx_τ_nom.u[t]
		len = length(r_idx) * length(c_idx)
		∇c[shift .+ (1:len)] = vec(ForwardDiff.jacobian(pū, ū))
		shift += len

		c_idx = idx_τ_sample.u[t]
		len = length(r_idx) * length(c_idx)
		∇c[shift .+ (1:len)] = vec(ForwardDiff.jacobian(pu, u))
		shift += len
	end

	return nothing
end

function constraints_sparsity(con::PolicyConstraint,
		policy,
		model_nom, model_sample,
		idx_τ_nom, idx_τ_sample, idx_θ,
		idx_z_nom, idx_z_sample, idx_z_policy,
		T;
		shift_row = 0, shift_col = 0)

	row = []
    col = []

	p = con.control_dim

	for t = 1:T-1
		r_idx = shift_row + (t - 1) * p .+ (1:p)

		c_idx = shift_col .+ idx_z_policy[idx_θ[t]]
		row_col!(row, col, r_idx, c_idx)

		c_idx = shift_col .+ idx_z_sample[idx_τ_sample.x[t]]
		row_col!(row, col, r_idx, c_idx)

		c_idx = shift_col .+ idx_z_nom[idx_τ_nom.x[t]]
		row_col!(row, col, r_idx, c_idx)

		c_idx = shift_col .+ idx_z_nom[idx_τ_nom.u[t]]
		row_col!(row, col, r_idx, c_idx)

		c_idx = shift_col .+ idx_z_sample[idx_τ_sample.u[t]]
		row_col!(row, col, r_idx, c_idx)
	end

    return collect(zip(row, col))
end

function constraints!(c, Z, con::PolicyConstraints,
	policy::Policy,
	prob::DPOProblems, idx::DPOIndices, N)

	shift = 0

	# mean
	Θ = view(Z, idx.policy)
	τ_nom = view(Z, idx.nom)
	τ_mean = view(Z, idx.mean)

	constraints!(view(c, shift .+ (1:con.con[1].n)),
			Θ, τ_nom, τ_mean, con.con[1], policy,
			prob.nom.model, prob.mean.model,
			prob.nom.idx, prob.mean.idx, idx.θ,
			prob.nom.h, prob.nom.T)

	shift += con.con[1].n

	for i = 1:N
		τ_sample = view(Z, idx.sample[i])
		constraints!(view(c, shift .+ (1:con.con[i + 1].n)),
				Θ, τ_nom, τ_sample,	con.con[i + 1], policy,
				prob.nom.model, prob.sample[i].model,
				prob.nom.idx, prob.sample[i].idx, idx.θ,
				prob.nom.h, prob.nom.T)
		shift += con.con[i + 1].n
	end
	nothing
end

function constraints_jacobian!(∇c, Z, con::PolicyConstraints,
		policy::Policy,
		prob::DPOProblems, idx::DPOIndices, N)

	shift = 0

	Θ = view(Z, idx.policy)
	τ_nom = view(Z, idx.nom)
	τ_mean = view(Z, idx.mean)

	len = length(constraints_sparsity(con.con[1],
			policy,
	 		prob.nom.model, prob.mean.model,
			prob.nom.idx, prob.mean.idx, idx.θ,
			idx.nom, idx.mean, idx.policy,
			prob.nom.T))

	constraints_jacobian!(view(∇c, shift .+ (1:len)),
			Θ, τ_nom, τ_mean, con.con[1], policy,
			prob.nom.model, prob.mean.model,
			prob.nom.idx, prob.mean.idx, idx.θ,
			prob.nom.h, prob.nom.T)

	shift += len

	for i = 1:N
		τ_sample = view(Z, idx.sample[i])

		len = length(constraints_sparsity(con.con[i + 1],
				policy,
		 		prob.nom.model, prob.sample[i].model,
				prob.nom.idx, prob.sample[i].idx, idx.θ,
				idx.nom, idx.mean, idx.policy,
				prob.nom.T))

		constraints_jacobian!(view(∇c, shift .+ (1:len)),
				Θ, τ_nom, τ_sample, con.con[i + 1], policy,
				prob.nom.model, prob.sample[i].model,
				prob.nom.idx, prob.sample[i].idx, idx.θ,
				prob.nom.h, prob.nom.T)

		shift += len
	end
	nothing
end

function constraints_sparsity(con::PolicyConstraints,
		policy::Policy,
		prob::DPOProblems, idx::DPOIndices, N;
		shift_row = 0, shift_col = 0)

	spar = constraints_sparsity(con.con[1],
			policy,
			prob.nom.model, prob.mean.model,
			prob.nom.idx, prob.mean.idx, idx.θ,
			idx.nom, idx.mean, idx.policy,
			prob.nom.T,
			shift_row = shift_row, shift_col = shift_col)
	shift_row += con.con[1].n

	for i = 1:N
		spar = collect([spar...,
				constraints_sparsity(con.con[i + 1],
				policy,
				prob.nom.model, prob.sample[i].model,
				prob.nom.idx, prob.sample[i].idx, idx.θ,
				idx.nom, idx.sample[i], idx.policy,
				prob.nom.T,
				shift_row = shift_row, shift_col = shift_col)...])
		shift_row += con.con[i + 1].n
	end

	return spar
end
