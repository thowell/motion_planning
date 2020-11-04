"""
    test trajectory optimization
"""

# objective
z0 = rand(prob_traj.num_var)
tmp_o(z) = eval_objective(prob_traj, z)
∇obj = zeros(prob_traj.num_var)
eval_objective_gradient!(∇obj, z0, prob_traj)
@assert norm(ForwardDiff.gradient(tmp_o, z0) - ∇obj) < 1.0e-10

# constraints
c0 = zeros(prob_traj.M)
eval_constraint!(c0, z0, prob_traj)
tmp_c(c, z) = eval_constraint!(c, z, prob_traj)
∇c_fd = ForwardDiff.jacobian(tmp_c, c0, z0)
spar = sparsity_jacobian(prob_traj)
∇c_vec = zeros(length(spar))
∇c = zeros(prob_traj.M, prob_traj.num_var)
eval_constraint_jacobian!(∇c_vec, z0, prob_traj)
for (i,k) in enumerate(spar)
    ∇c[k[1],k[2]] = ∇c_vec[i]
end
@assert norm(vec(∇c) - vec(∇c_fd)) < 1.0e-10
@assert sum(∇c) - sum(∇c_fd) < 1.0e-10
