abstract type Constraints end

"""
    constraints template
"""
function constraints!(c, Z, con::Constraints, model, idx, h, T)
    return nothing
end

function constraints_jacobian!(∇c, Z, con::Constraints, model, idx, h, T)
   return nothing
end

function constraints_sparsity(con::Constraints, model, idx, T; r_shift = 0)
    row = []
    col = []
    return collect(zip(row, col))
end

function constraints_inequality(con::Constraints)
    return (1:0)
end

"""
    empty constraints
"""

struct EmptyConstraints
    n
    ineq

    function EmptyConstraints()
        EmptyConstraints(0, (1:0))
    end
end

"""
    multiple constraints
"""
struct MultiConstraints <: Constraints
    con::Vector{Constraints}
    n
    ineq
end

function multiple_constraints(con)
    n = sum([_con.n for _con in con])
    ineq = vcat([i == 1 ? _con.ineq : sum([con[k].n for k = 1:i-1]) .+ _con.ineq for (i,_con) in enumerate(con)]...)

    return MultiConstraints(con, n, ineq)
end

function add_constraints(c1::Constraints, c2::Constraints)
    multiple_constraints([c1, c2])
end

function add_constraints(c1::MultiConstraints, c2::Constraints)
    multiple_constraints([c1.con..., c2])
end

function constraints!(c, Z, con::MultiConstraints, model, idx, h, T)
    shift = 0
    for _con in con.con
        c_idx = shift .+ (1:_con.n)
        constraints!(view(c, c_idx), Z, _con, model, idx, h, T)
        shift += _con.n
    end
    return nothing
end

function constraints_jacobian!(∇c, Z, con::MultiConstraints, model, idx, h, T)
    c_shift = 0
    jac_shift = 0
    for _con in con.con
        jac_len = length(constraints_sparsity(_con, model, idx, T, r_shift = c_shift))
        jac_idx = jac_shift .+ (1:jac_len)
        constraints_jacobian!(view(∇c, jac_idx), Z, _con, model, idx, h, T)
        c_shift += _con.n
        jac_shift += jac_len
    end
    return nothing
end

function constraints_sparsity(con::MultiConstraints, model, idx, T; r_shift = 0)
    c_shift = r_shift
    sparsity = []
    for _con in con.con
        spar = constraints_sparsity(_con, model, idx, T, r_shift = c_shift)
        sparsity = collect([sparsity..., spar...])
        c_shift += _con.n
    end
    return sparsity
end
