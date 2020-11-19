struct AugmentedLagrangian
    λ               # duals - general constraints
    λl              # duals - primal lower bounds
    λu              # duals - primal upper bounds

    ρ               # penalties - general constraints
    ρl              # penalties - primal lower bounds
    ρu              # penalties - primal upper bounds
    ρ0              # initial penalty
    s               # penalty scaling

    as              # active set for general constraints
    asl             # active set for primal lower bounds
    asu             # active set for primal upper bounds

    c               # memory for constraints
    ∇c              # memory for constraints Jacobian

    xl              # primal lower bounds
    xu              # primal upper bounds
    cl              # memory for xl - x
    cu              # memory for x - xu

    idx_ineq        # inequality constraint indices
    idx_l           # primal lower bound indices
    idx_u           # primal upper bound indices
    idx_soc         # second-order cone constraint indices
end

function augmented_lagrangian(n, m;
        ρ0 = 1.0, s = 10.0,
        idx_ineq = (1:0),
        xl = -Inf * ones(n), xu = Inf * ones(n),
        idx_soc = (1:0))

    bool_l = isfinite.(xl)
    bool_u = isfinite.(xu)
    idx_l = (1:length(xl))[bool_l]
    idx_u = (1:length(xu))[bool_u]
    ml = sum(bool_l)
    mu = sum(bool_u)

    λ = zeros(m)
    λl = zeros(ml)
    λu = zeros(mu)

    ρ = ρ0 * ones(m)
    ρl = ρ0 * ones(ml)
    ρu = ρ0 * ones(mu)

    as = ones(Bool, m)
    asl = ones(Bool, ml)
    asu = ones(Bool, mu)

    c = zeros(m)
    ∇c = spzeros(m, n)

    cl = zeros(ml)
    cu = zeros(mu)

    AugmentedLagrangian(λ, λl, λu,
        ρ, ρl, ρu, ρ0, s,
        as, asl, asu,
        c, ∇c,
        xl, xu, cl, cu,
        idx_ineq, idx_l, idx_u, idx_soc)
end

function update!(al::AugmentedLagrangian)
    dual_update!(al)
    penalty_update!(al)
    active_set_update!(al)
end

function dual_update!(al::AugmentedLagrangian)
    al.λ .+= al.ρ .* al.c
    al.λl .+= al.ρl .* al.cl
    al.λu .+= al.ρu .* al.cu

    inequality_projection!(al)
    #TODO: soc project
end

function inequality_projection!(al::AugmentedLagrangian)
    al.λ[al.idx_ineq] = max.(0.0, view(al.λ, al.idx_ineq))
    al.λl .= max.(0.0, al.λl)
    al.λu .= max.(0.0, al.λu)
end

function soc_projection!(al::AugmentedLagrangian)
    return nothing
end

function penalty_update!(al::AugmentedLagrangian)
    al.ρ .*= al.s
    al.ρl .*= al.s
    al.ρu .*= al.s
end

function active_set_update!(al::AugmentedLagrangian)
    fill!(al.as, 1)
    fill!(al.asl, 1)
    fill!(al.asu, 1)

    al.as[al.idx_ineq] = .!((view(al.λ, al.idx_ineq) .== 0.0) .& (view(al.c, al.idx_ineq) .< 0.0)) # (view(al.c, al.idx_ineq) .> 0.0) #
    al.asl .= .!((al.λl .== 0.0) .& (al.cl .< 0.0)) # (al.cl .> 0.0) #
    al.asu .= .!((al.λu .== 0.0) .& (al.cu .< 0.0)) # (al.cu .> 0.0) #
    #TODO: soc as
end

function bounds!(al::AugmentedLagrangian, x)
    al.cl .= view(al.xl, al.idx_l) - view(x, al.idx_l)
    al.cu .= view(x, al.idx_u) - view(al.xu, al.idx_u)
end

function constraint_violation(al::AugmentedLagrangian, x)
    c!(al.c, x)
    bounds!(al, x)
    al.c[al.idx_ineq] = max.(0.0, al.c[al.idx_ineq])
    return max(norm(al.c, Inf),
               norm(max.(0.0, al.cl), Inf),
               norm(max.(0.0, al.cu), Inf))
end

function reset!(al::AugmentedLagrangian)
    fill!(al.λ, 0.0)
    fill!(al.λl, 0.0)
    fill!(al.λu, 0.0)
    fill!(al.ρ, al.ρ0)
    fill!(al.ρl, al.ρ0)
    fill!(al.ρu, al.ρ0)
    fill!(al.as, 1)
    fill!(al.asl, 1)
    fill!(al.asu, 1)
    fill!(al.c, 0.0)
    fill!(al.cl, 0.0)
    fill!(al.cu, 0.0)
end

# n = 20
# m = 10
# al = augmented_lagrangian(n, m, xl = zeros(n))
# update!(al)
# reset!(al)
#
# al.asl
# al.λl
# al.cl
# ((al.λl .== 0.0) .& (al.cl .< 0.0))
