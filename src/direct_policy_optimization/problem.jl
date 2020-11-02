struct DirectPolicyOptimizationProblem
	# trajectory optimization problems
	prob_nom
	prob_sample

	# objectives
	objective_sample
	objective_policy

	# constraints
	constraints_sample_dynamics
	constraints_policy

	# number of samples
	N

	# disturbance trajectory
	W
	w
	sqrt_type

	# number of variables
	num_var
	num_con

	num_var_sample
	num_var_con

	# indices
	idx_nom
	idx_mean
	idx_sample
	idx_policy

	# shifts
	shift_mean
	shift_sample
	shift_policy
end

function primal_bounds(dpo::DirectPolicyOptimizationProblem)
    T = dpo.prob_nom.T

    idx_nom = dpo.idx_nom
	idx_mean = dpo.idx_mean
	idx_sample = dpo.idx_sample

    Zl = -Inf * ones(dpo.num_var)
    Zu = Inf * ones(dpo.num_var)

	# nominal
	Zl[idx_nom], Zu[idx_nom] = primal_bounds(dpo.prob_nom)

	# mean
	Zl[idx_mean], Zu[idx_mean] = primal_bounds(dpo.prob_sample)

	# sample
	for i = 1:dpo.N
		Zl[idx_sample[i]], Zu[idx_sample[i]] = primal_bounds(dpo.prob_sample)
	end

    return Zl, Zu
end

function constraint_bounds(dpo::DirectPolicyOptimizationProblem)
	idx_nom = dpo.idx_nom
	idx_sample = dpo.idx_sample

    cl = zeros(dpo.num_con)
    cu = zeros(dpo.num_con)

	# nominal
    cu[idx_nom[dpo.prob_nom.con.ineq]] .= Inf

	# sample
	for i = 1:dpo.N
		cu[idx_sample[i][dpo.prob_sample.con.ineq]] .= Inf
	end

    return cl, cu
end

function eval_objective(dpo::DirectPolicyOptimizationProblem, Z)
	# nominal
	J = objective(view(Z, dpo.idx_nom), dpo.prob_nom)

	# mean
	for i = 1:2 * dpo.prob_sample.model.d
		J += objective(view(Z, dpo.idx_mean), dpo.prob_sample)
	end

	# sample
	for i = 1:dpo.N
		J += objective(view(Z, dpo.idx_sample[i]), dpo.prob_sample)
	end

	# sample objective
	if dpo.objective_sample
		for i = 1:dpo.N
			nothing
		end
	end

	# policy objective
	if dpo.objective_policy
		for i = 1:dpo.N
			nothing
		end
	end

	return J
end

function eval_objective_gradient!(∇l, Z, dpo::DirectPolicyOptimizationProblem)
    ∇l .= 0.0

	# nominal
    objective_gradient!(view(∇l, dpo.idx_nom),
		view(Z, dpo.idx_nom), dpo.prob_nom)

	# mean
	for i = 1:2 * dpo.prob_sample.model.d
		objective_gradient!(view(∇l, dpo.idx_mean),
			view(Z, dpo.idx_mean), dpo.prob_sample)
	end

	# samples
	for i = 1:dpo.N
		objective_gradient!(view(∇l, dpo.idx_sample[i]),
			view(Z, dpo.idx_sample[i]), dpo.prob_sample)
	end

	# sample objective
	if dpo.objective_sample
		# mean

		# samples
		for i = 1:dpo.N
			nothing
		end
	end

	# policy objective
	if dpo.objective_policy
		for i = 1:dpo.N
			nothing
		end
	end

    return nothing
end

function eval_constraint!(c, Z, dpo::DirectPolicyOptimizationProblem)
	shift = 0

	# nominal
    constraints!(view(c, 1:dpo.prob_nom.num_con),
		view(Z, dpo.idx_nom), dpo.prob_nom)
	shift += dpo.prob_nom.num_con

	# sample
	for i = 1:dpo.N
		constraints!(view(c, shift .+ (1:dpo.prob_sample.num_con)),
			view(Z, dpo.idx_sample[i]), dpo.prob_sample)
		shift += dpo.prob_sample.num_con
	end

	# sample dynamics
	if dpo.constraints_sample_dynamics
		shift += 0
		nothing
	end

	# policy
	if dpo.constraints.policy
		Θ = view(Z, dpo.idx_policy)
		τ_nom = view(Z, dpo.idx_nom)

		constraints!(view(c, shift .+ (1:dpo.con_policy.n)),
				Θ, τ_nom, view(Z, dpo.idx_mean),
				dpo.con_policy, dpo.prob_nom.model, dpo.prob_sample.model,
				dpo.prob_nom.idx, dpo.prob_sample.idx, dpo.idx_policy,
				dpo.prob_nom.h, dpo.prob_nom.T)

		shift += dpo.con_policy.n

		for i = 1:dpo.N
			constraints!(view(c, shift .+ (1:dpo.con_policy.n)),
					Θ, τ_nom, view(Z, dpo.idx_sample[i]),
					dpo.con_policy, dpo.prob_nom.model, dpo.prob_sample.model,
					dpo.prob_nom.idx, dpo.prob_sample.idx, dpo.idx_policy,
					dpo.prob_nom.h, dpo.prob_nom.T)

			shift += dpo.con_policy.n
			nothing
		end
	end

    return nothing
end

function eval_constraint_jacobian!(∇c, Z, dpo::DirectPolicyOptimizationProblem)
	shift = 0

	# nominal
	len_nom = length(sparsity_jacobian(dpo.prob_nom))
    constraints_jacobian!(view(∇c, 1:len_nom), view(Z, dpo.idx_nom), dpo.prob_nom)
	shift += len_nom

	# sample
	for i = 1:dpo.N
		len = length(sparsity_jacobian(dpo.prob_sample))
		constraints_jacobian!(view(∇c, shift .+ (1:len)),
			view(Z, dpo.idx_sample[i]), dpo.prob_sample)
		shift += len
	end

	# sample dynamics
	if dpo.constraints_sample_dynamics
		shift += 0
		nothing
	end

	# policy
	if dpo.constraints_policy
		Θ = view(Z, dpo.idx_policy)
		τ_nom = view(Z, dpo.idx_nom)

		len = length(constraints_sparsity(dpo.con_policy,
		 		dpo.prob_nom.model, dpo.prob_sample.model,
				dpo.prob_nom.idx, dpo.prob_sample.idx, dpo.idx_policy,
				dpo.prob_nom.T))

		constraints_jacobian!(view(∇c, shift .+ (1:len)),
				Θ, τ_nom, view(Z, dpo.idx_mean),
				dpo.con_policy, dpo.prob_nom.model, dpo.prob_sample.model,
				dpo.prob_nom.idx, dpo.prob_sample.idx, dpo.idx_policy,
				dpo.prob_nom.h, dpo.prob_nom.T)

		shift += len

		for i = 1:dpo.N
			len = length(constraints_sparsity(dpo.con_policy,
			 		dpo.prob_nom.model, dpo.prob_sample.model,
					dpo.prob_nom.idx, dpo.prob_sample.idx, dpo.idx_policy,
					dpo.prob_nom.T))

			constraints_jacobian!(view(∇c, shift .+ (1:len)),
					Θ, τ_nom, view(Z, dpo.idx_sample[i]),
					dpo.con_policy, dpo.prob_nom.model, dpo.prob_sample.model,
					dpo.prob_nom.idx, dpo.prob_sample.idx, dpo.idx_policy,
					dpo.prob_nom.h, dpo.prob_nom.T)

			shift += len
		end
	end

    return nothing
end

function sparsity_jacobian(dpo::DirectPolicyOptimizationProblem)
	shift = 0

	# nominal
    spar = constraints_sparsity(dpo.prob_nom)
	shift += dpo.prob_nom.num_con

	# sample
	for i = 1:dpo.N
		spar = collect([spar...,
			constraints_sparsity(dpo.prob_sample, shift = shift)...])
		shift += dpo.prob_sample.num_con
	end

	# sample dynamics
	if dpo.constraints_sample_dynamics
		shift += 0
		nothing
	end

	# policy
	if dpo.constraints_policy

		spar = collect([spar...,
				constraints_sparsity(dpo.con_policy,
		 		dpo.prob_nom.model, dpo.prob_sample.model,
				dpo.prob_nom.idx, dpo.prob_sample.idx, dpo.idx_policy,
				dpo.prob_nom.T,
				shift_con = shift,
				shift_sample = dpo.shift_mean,
				shift_policy = dpo.shift_policy)]...)
		shift += dpo.con_policy.n

		for i = 1:dpo.N
			spar = collect([spar...,
					constraints_sparsity(dpo.con_policy,
			 		dpo.prob_nom.model, dpo.prob_sample.model,
					dpo.prob_nom.idx, dpo.prob_sample.idx, dpo.idx_policy,
					dpo.prob_nom.T,
					shift_con = shift,
					shift_sample = dpo.shift_sample[i],
					shift_policy = dpo.shift_policy)]...)
			shift += dpo.con_policy.n
		end
	end

	return spar
end
