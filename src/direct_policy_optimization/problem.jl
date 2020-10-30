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

	# disturbance trajectories
	W
	W_sqrt
	sqrt_type

	# number of variables
	num_var
	num_con

	# indices
	idx_nom
	idx_sample
	idx_policy
end

function primal_bounds(dpo::DirectPolicyOptimizationProblem)
    T = dpo.prob_nom.T
    idx_nom = dpo.idx_nom
	idx_sample = dpo.idx_sample

    Zl = -Inf * ones(dpo.num_var)
    Zu = Inf * ones(dpo.num_var)

	# nominal
	Zl[idx_nom], Zu[idx_nom] = primal_bounds(dpo.prob_nom)

	# sample
	for i = 1:dpo.N
		Zl[idx_sample[i]], Zu[idx_sample[i]] = primal_bounds(dpo.prob_sample[i])
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
		cu[idx_sample[i][dpo.prob_sample[i].con.ineq]] .= Inf
	end

    return cl, cu
end

function eval_objective(dpo::DirectPolicyOptimizationProblem, Z)
	# nominal
	J = objective(view(Z, dpo.idx_nom), dpo.prob_nom)

	# sample
	for i = 1:dpo.N
		J += objective(view(Z, dpo.idx_sample[i]), dpo.prob_sample[i])
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

	# sample
	for i = 1:dpo.N
		objective_gradient!(view(∇l, dpo.idx_sample[i]),
			view(Z, dpo.idx_sample[i]), dpo.prob_sample[i])
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
		constraints!(view(c, shift .+ (1:dpo.prob_sample[i].num_con)),
			view(Z, dpo.idx_sample[i]), dpo.prob_sample[i])
		shift += dpo.prob_sample[i].num_con
	end

	# sample dynamics
	if dpo.constraints_sample_dynamics
		shift += 0
		nothing
	end

	# policy
	if dpo.constraints.policy
		for i = 1:dpo.N
			shift += 0
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
		len = length(sparsity_jacobian(dpo.prob_sample[i]))
		constraints_jacobian!(view(∇c, shift .+ (1:len)),
			view(Z, dpo.idx_sample[i]), dpo.prob_sample[i])
		shift += len
	end

	# sample dynamics
	if dpo.constraints_sample_dynamics
		shift += 0
		nothing
	end

	# policy
	if dpo.constraints_policy
		for i = 1:dpo.N
			shift += 0
			nothing
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
			constraints_sparsity(dpo.prob_sample[i], shift = shift)...])
		shift += dpo.prob_sample[i].num_con
	end

	# sample dynamics
	if dpo.constraints_sample_dynamics
		shift += 0
		nothing
	end

	# policy
	if dpo.constraints_policy
		for i = 1:dpo.N
			shift += 0
			nothing
		end
	end

	return spar
end
