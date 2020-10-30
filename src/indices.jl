"""
    indices for state and control trajectories over horizon T
"""
struct Indices
    x
    u
end

function indices(n, m, T;
    shift = 0)
    x = []
    u = []

    for t = 1:T
        push!(x, shift + (t - 1) * (n + m) .+ (1:n))

        t == T && continue
        push!(u, shift + (t - 1) * (n + m) + n .+ (1:m))
    end

    return Indices(x,u)
end
