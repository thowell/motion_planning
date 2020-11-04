struct DirectPolicyOptimizationProblem
	# problem(s)
	prob

	# objective
	obj

	# constraints
	con

	# number of samples
	N # 2n
	D # 2d

	# disturbance trajectory
	W
	w
	sqrt_type
	β

	# number of variables
	num_var
	num_con

	# indices
	idx
end

function dpo_problem(prob_nom, prob_mean, prob_sample, Q, R, W, β;
		p = prob_nom.model.m * prob_nom.model.n,
		N = 2 * prob_nom.model.n, D = 2 * prob_nom.model.d)

	# problem(s)
	prob = DPOProblems(prob_nom, prob_mean, prob_sample)

	# indices
	idx, num_var = dpo_indices(prob, p, N, D)

	# objective
	obj = SampleObjective(Q, R)

	# constraints
	con_dynamics = sample_dynamics_constraints(prob, N, M)
	con_policy = policy_constraints(prob, N)
	con = [con_dynamics, con_policy]

	num_con = prob.nom.num_con + sum([prob.sample[i].num_con for i = 1:N])
	num_con += con_dynamics.n + con_policy.n

	# disturbances
	w = []
	for t = 1:prob.nom.T-1
		_w = sqrt(W[t])
		tmp = []
		for i = 1:prob.mean.model.d
			push!(tmp, _w[:, i])
			push!(tmp, -1.0 * _w[:, i])
		end
		push!(w, tmp)
	end

	return DirectPolicyOptimizationProblem(prob, obj, con,
		N, D,
		W, w, :principle, β,
		num_var, num_con,
		idx)
end

struct DPOProblems
	nom
	mean
	sample
end

struct DPOIndices
	nom
	mean
	sample

	θ
	policy

	s
	slack
end

function dpo_indices(prob::DPOProblems, p, N, D)
    T = prob.nom.T

    # nominal
    idx_nom = 1:prob.nom.num_var

    # mean
    idx_mean = prob.nom.num_var .+ (1:prob.mean.num_var)

    # sample
    shift = prob.nom.num_var + prob.mean.num_var
    idx_sample = []
    for i = 1:N
        push!(idx_sample, shift .+ (1:prob_sample[i].num_var))
        shift += prob_sample[i].num_var
    end

    # policy
    idx_θ = [(t - 1) * p .+ (1:p) for t = 1:T-1]
    n_policy = p * (T - 1)
    idx_policy = shift .+ (1:n_policy)
    shift += n_policy

    # slack
    idx_s = [(t - 1) * prob.mean.model.n .+ (1:prob.mean.model.n) for t = 1:T-1]
    idx_slack = []
    n_slack = prob.mean.model.n * (T - 1)
    for j = 1:(N + D)
        push!(idx_slack, shift .+ (1:n_slack))
        shift += n_slack
    end
	num_var = shift

    return DPOIndices(idx_nom, idx_mean, idx_sample, idx_θ, idx_policy, idx_s, idx_slack), num_var
end

function primal_bounds(dpo::DirectPolicyOptimizationProblem)
    T = dpo.prob.nom.T

    idx_nom = dpo.idx.nom
	idx_mean = dpo.idx.mean
	idx_sample = dpo.idx.sample

    Zl = -Inf * ones(dpo.num_var)
    Zu = Inf * ones(dpo.num_var)

	# nominal
	Zl[idx_nom], Zu[idx_nom] = primal_bounds(dpo.prob.nom)

	# sample
	for i = 1:dpo.N
		Zl[idx_sample[i]], Zu[idx_sample[i]] = primal_bounds(dpo.prob.sample[i])
	end

    return Zl, Zu
end

function constraint_bounds(dpo::DirectPolicyOptimizationProblem)
	idx_nom = dpo.idx.nom
	idx_sample = dpo.idx.sample

    cl = zeros(dpo.num_con)
    cu = zeros(dpo.num_con)

	# nominal
    cu[idx_nom[dpo.prob.nom.con.ineq]] .= Inf

	# sample
	for i = 1:dpo.N
		cu[idx_sample[i][dpo.prob.sample[i].con.ineq]] .= Inf
	end

    return cl, cu
end

function eval_objective(dpo::DirectPolicyOptimizationProblem, Z)
	# nominal
	J = objective(view(Z, dpo.idx.nom), dpo.prob.nom)

	# samples
	for i = 1:dpo.N
		J += objective(view(Z, dpo.idx.sample[i]), dpo.prob.sample[i])
	end

	# sample objective
	J += objective(Z, dpo.obj, dpo.prob, dpo.idx, dpo.N, dpo.D)

	return J
end

function eval_objective_gradient!(∇J, Z, dpo::DirectPolicyOptimizationProblem)
    ∇J .= 0.0

	# nominal
    objective_gradient!(view(∇J, dpo.idx.nom),
		view(Z, dpo.idx.nom), dpo.prob.nom)

	# samples
	for i = 1:dpo.N
		objective_gradient!(view(∇J, dpo.idx.sample[i]),
			view(Z, dpo.idx.sample[i]), dpo.prob.sample[i])
	end

	# sample objective
	objective_gradient!(∇J, Z, dpo.obj, dpo.prob, dpo.idx, dpo.N, dpo.D)

    return nothing
end

function eval_constraint!(c, Z, dpo::DirectPolicyOptimizationProblem)
	shift = 0

	# nominal
    constraints!(view(c, 1:dpo.prob.nom.num_con),
		view(Z, dpo.idx.nom), dpo.prob.nom)
	shift += dpo.prob.nom.num_con

	# sample
	for i = 1:dpo.N
		constraints!(view(c, shift .+ (1:dpo.prob.sample[i].num_con)),
			view(Z, dpo.idx.sample[i]), dpo.prob.sample[i])
		shift += dpo.prob.sample[i].num_con
	end

	# sample dynamics
	constraints!(view(c, shift .+ (1:dpo.con[1].n)), Z, dpo.con[1],
		dpo.prob, dpo.idx, dpo.N, dpo.D, dpo.w, dpo.β)
	shift += dpo.con[1].n

	# policy
	constraints!(view(c, shift .+ (1:dpo.con[2].n)), Z, dpo.con[2],
		dpo.prob, dpo.idx, dpo.N)
	shift += dpo.con[2].n


    return nothing
end

function eval_constraint_jacobian!(∇c, Z, dpo::DirectPolicyOptimizationProblem)
	shift = 0

	# nominal
	len_nom = length(sparsity_jacobian(dpo.prob.nom))
    constraints_jacobian!(view(∇c, 1:len_nom), view(Z, dpo.idx.nom), dpo.prob.nom)
	shift += len_nom

	# samples
	for i = 1:dpo.N
		len = length(sparsity_jacobian(dpo.prob.sample[i]))
		constraints_jacobian!(view(∇c, shift .+ (1:len)),
			view(Z, dpo.idx.sample[i]), dpo.prob.sample[i])
		shift += len
	end

	# sample dynamics
	len = length(constraints_sparsity(dpo.con[1], dpo.prob, dpo.idx,
		dpo.N, dpo.D))
	constraints_jacobian!(view(∇c, shift .+ (1:len)), Z, dpo.con[1],
		dpo.prob, dpo.idx,
		dpo.N, dpo.D,
		dpo.w, dpo.β)
	shift += len

	# policy
	len = length(constraints_sparsity(dpo.con[2], dpo.prob, dpo.idx,
		dpo.N))
	constraints_jacobian!(view(∇c, shift .+ (1:len)), Z, dpo.con[2],
		dpo.prob, dpo.idx, dpo.N)
	shift += len

    return nothing
end

function sparsity_jacobian(dpo::DirectPolicyOptimizationProblem)

	# nominal
    spar = constraints_sparsity(dpo.prob.nom)
	con_shift = dpo.prob.nom.num_con

	# samples
	shift_col = dpo.prob.nom.num_var + dpo.prob.mean.num_var
	for i = 1:dpo.N
		spar = collect([spar...,
			constraints_sparsity(dpo.prob.sample[i],
				shift_row = con_shift,
				shift_col = shift_col)...])
		con_shift += dpo.prob.sample[i].num_con
		shift_col += dpo.prob.sample[i].num_var
	end

	# sample dynamics
	spar = collect([spar...,
					constraints_sparsity(dpo.con[1], dpo.prob, dpo.idx,
						dpo.N, dpo.D, shift_row = con_shift)...])
	con_shift += dpo.con[1].n

	# policy
	spar = collect([spar...,
					constraints_sparsity(dpo.con[2], dpo.prob, dpo.idx,
					dpo.N, shift_row = con_shift)...])
	con_shift += dpo.con[2].n

	return spar
end
