using LinearAlgebra, ForwardDiff, SparseArrays, Optim, LineSearches
# include("al.jl")
prob = prob
function f(x)
    MOI.eval_objective(prob, x)
end

function g!(G, x)
    MOI.eval_objective_gradient(prob, G, x)
    nothing
end


function c!(c, x)
    MOI.eval_constraint(prob, c, x)
    c .*= -1.0
    nothing
end

spar = sparsity_jacobian(prob)
global jac = zeros(length(spar))
global ii = [s[1] for s in spar]
global jj = [s[2] for s in spar]
function d!(D, x)
    MOI.eval_constraint_jacobian(prob, jac, x)
    D .= sparse(ii, jj, -1.0 .* jac)
end

function f_al(x, al::AugmentedLagrangian)
    # evaluate constraints
    c!(al.c, x)
    bounds!(al, x)
    active_set_update!(al)

    # compute objective
    J = f(x)

    # add augmented Lagrangian terms
    J += al.λ' * al.c + 0.5 * sum(al.as .* al.ρ .* (al.c.^2.0))
    J += al.λl' * al.cl + 0.5 * sum(al.asl .* al.ρl .* (al.cl.^2.0))
    J += al.λu' * al.cu + 0.5 * sum(al.asu .* al.ρu .* (al.cu.^2.0))
end

function g_al!(G, x, al::AugmentedLagrangian)
    # compute objective gradient
    g!(G, x)

    # evaluate constraints
    # ForwardDiff.jacobian!(al.∇c, c!, al.c, x)
    c!(al.c, x)
    d!(al.∇c, x)

    bounds!(al, x)
    active_set_update!(al)

    # add augmented Lagrangian gradient terms
    G .+= al.∇c' * (al.λ + al.as .* al.ρ .* al.c)
    G[al.idx_l] -= (al.λl + al.asl .* al.ρl .* al.cl)
    G[al.idx_u] += (al.λu + al.asu .* al.ρu .* al.cu)
    return nothing
end

function solve(x, al; alg = :LBFGS, max_iter = 5, c_tol = 1.0e-3)
    # reset augmented Lagrangian
    reset!(al)
    println("solving...")
    for i = 1:max_iter
        # x̄, ū = unpack(x, prob)
        #
        # visualize!(vis, model, state_to_configuration(x̄), Δt = h)

        # update augmented Lagrangian methods
        _f(z) = f_al(z, al)
        _g!(G, z) = g_al!(G, z, al)

        # solve
        opt = Optim.Options()
                # x_tol = 1.0e-6,
                # f_tol = 1.0e-6,
                # g_tol = 1.0e-2)
                # iterations = convert(Int,1e4))

        sol = optimize(_f, _g!, x,
                        eval(alg)(),#linesearch = LineSearches.BackTracking()),
                        opt)

        # evaluate constraints
        x = sol.minimizer
        c_max = constraint_violation(al, x)
        println("iter: $i")
        println("c_max: $c_max")

        # check for convergence -> update augmented Lagrangian
        if c_max < c_tol
            return x, sol
        else
            c!(al.c, x)
            bounds!(al, x)
            update!(al)
        end
    end

    return x, sol
end



n = prob.num_var
m = prob.num_con
xl, xu = prob.primal_bounds
cl, cu = prob.constraint_bounds
idx_ineq = (1:m)[cu .> cl]
sum(isfinite.(xl))
sum(isfinite.(xu))
sum(cu .> cl)

al = augmented_lagrangian(n, m,
    xl = xl, xu = xu, ρ0 = 10.0, s = 10.0,
    idx_ineq = idx_ineq)

@time x_sol_al, sol = solve(copy(z0), al,
    alg = :MomentumGradientDescent, max_iter = 20, c_tol = 1.0e-2)

# Visualize
# using Plots
# x̄, ū = unpack(x_sol_al, prob)
# visualize!(vis, model, state_to_configuration(x̄), Δt = h)

# plot(hcat(x̄...)', width = 2.0)
# plot(hcat(ū...)[1:model.m, :]', width = 2.0, linetype = :steppost)
# s̄ = [ū[t][model.idx_s] for t = 1:T-1]
# plot(hcat(s̄...)', width = 2.0, linetype = :steppost)
