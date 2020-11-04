"""
    test direct policy optimization
"""

# objective
z0 = rand(prob_dpo.num_var)
tmp_o(z) = eval_objective(prob_dpo,z)
∇j = zeros(prob_dpo.num_var)
eval_objective_gradient!(∇j, z0, prob_dpo)
@assert norm(ForwardDiff.gradient(tmp_o, z0) - ∇j) < 1.0e-10

# sample dynamics
con_dynamics = sample_dynamics_constraints(prob_dpo.prob,
    prob_dpo.N, prob_dpo.N + prob_dpo.D)
c0 = zeros(con_dynamics.n)
_c(c,z) = constraints!(c, z, con_dynamics,
    prob_dpo.prob, prob_dpo.idx, prob_dpo.N, prob_dpo.D,
    prob_dpo.w, prob_dpo.β)
∇c_fd = zeros(con_dynamics.n, prob_dpo.num_var)
FiniteDiff.finite_difference_jacobian!(∇c_fd, _c, z0)
# ∇c_fd = ForwardDiff.jacobian(_c, c0, z0)
spar = constraints_sparsity(con_dynamics, prob_dpo.prob, prob_dpo.idx,
    prob_dpo.N, prob_dpo.D)
∇c_vec = zeros(length(spar))
∇c = zeros(con_dynamics.n, prob_dpo.num_var)
constraints_jacobian!(∇c_vec, z0, con_dynamics,
    prob_dpo.prob, prob_dpo.idx, prob_dpo.N, prob_dpo.D,
    prob_dpo.w, prob_dpo.β)
for (i,k) in enumerate(spar)
    ∇c[k[1],k[2]] = ∇c_vec[i]
end
@assert norm(vec(∇c) - vec(∇c_fd)) < 1.0e-6
@assert sum(∇c) - sum(∇c_fd) < 1.0e-6

# policy
con_policy = policy_constraints(prob_dpo.prob, prob_dpo.N)
c0 = zeros(con_policy.n)
_c(c,z) = constraints!(c, z, con_policy,
    prob_dpo.prob, prob_dpo.idx, prob_dpo.N)

# ∇c_fd = zeros(con_policy.n, prob_dpo.num_var)
# FiniteDiff.finite_difference_jacobian!(∇c_fd, _c, z0)
∇c_fd = ForwardDiff.jacobian(_c, c0, z0)
spar = constraints_sparsity(con_policy, prob_dpo.prob, prob_dpo.idx,
    prob_dpo.N)
∇c_vec = zeros(length(spar))
∇c = zeros(con_policy.n, prob_dpo.num_var)
constraints_jacobian!(∇c_vec, z0, con_policy,
    prob_dpo.prob, prob_dpo.idx, prob_dpo.N)

for (i,k) in enumerate(spar)
    ∇c[k[1],k[2]] = ∇c_vec[i]
end
@assert norm(vec(∇c) - vec(∇c_fd)) < 1.0e-10
@assert sum(∇c) - sum(∇c_fd) < 1.0e-10

# constraints
c0 = zeros(prob_dpo.num_con)
_c(c,z) = eval_constraint!(c, z, prob_dpo)

∇c_fd = zeros(prob_dpo.num_con, prob_dpo.num_var)
FiniteDiff.finite_difference_jacobian!(∇c_fd, _c, z0)
spar = sparsity_jacobian(prob_dpo)
∇c_vec = zeros(length(spar))
∇c = zero(∇c_fd)
eval_constraint_jacobian!(∇c_vec, z0, prob_dpo)
for (i,k) in enumerate(spar)
    ∇c[k[1],k[2]] = ∇c_vec[i]
end
@assert norm(vec(∇c) - vec(∇c_fd)) < 1.0e-5
@assert sum(∇c) - sum(∇c_fd) < 1.0e-5
