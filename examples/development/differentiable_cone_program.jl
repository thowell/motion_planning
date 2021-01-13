using LinearAlgebra, ForwardDiff
using Convex, SCS, ECOS

function Pκ(z)
    z0 = z[1]
    z̄ = z[2:end]
    n = length(z)

    η1 = z0 - norm(z̄)
    η2 = z0 + norm(z̄)

    w̄ = ones(n - 1) ./ norm(ones(n - 1))

    if z̄ != 0.0
        u1 = 0.5 * [1.0; -z̄ ./ norm(z̄)]

        u2 = 0.5 * [1.0; z̄ ./ norm(z̄)]
    else
        u1 = 0.5 * [1.0; -w̄]
        u2 = 0.5 * [1.0; w̄]
    end

    return max(0.0, η1) * u1 + max(0.0, η2) * u2
end

function ∇Pκ(z)
    z0 = z[1]
    z̄ = z[2:end]
    n = length(z)

    if z0 != norm(z̄) || z0 != norm(z̄)
        if z0 < -norm(z̄)
            @warn "projection weirdness"
            return zeros(n, n)
        elseif z0 > norm(z̄)
            return Diagonal(ones(n))
        elseif -norm(z̄) < z0 & z0 < norm(z̄)
            w̄ = z̄ ./ norm(z̄)
            H = ((1.0 + z0 / norm(z̄)) * Diagonal(ones(n-1))
                - z0 / norm(z̄) * w̄ * w̄')
            return 0.5 * [1.0 w̄'; w̄ H]
        end

    else
        if all(z̄ .!= 0.0) & z0 == norm(z̄)
            V1 = Diagonal(ones(n))

            w̄ = z̄ ./ norm(z̄)
            H = (2.0 * Diagonal(ones(n-1))
                - w̄ * w̄')
            V2 = 0.5 * [1.0 w̄'; w̄ H]
            return V2
        elseif all(z̄ .!= 0.0) & z0 == -norm(z̄)
            V1 = zeros(n, n)

            w̄ = z̄ ./ norm(z̄)
            H = - w̄ * w̄'
            V2 = 0.5 * [1.0 w̄'; w̄ H]
            return V2
        elseif all(z̄ .== 0.0) & z0 .== 0.0
            V1 = zeros(n, n)
            V2 = Diagonal(ones(n))

            w̄ = ones(n - 1) ./ norm(ones(n - 1))
            ρ = 0.5
            H = ((1.0 + ρ) * Diagonal(ones(n-1))
                - ρ * w̄ * w̄')
            V3 = 0.5 * [1.0 w̄'; w̄ H]

            return V3
        else
            @error "projection jacobian weirdness"
        end
    end
end

n = 3
m = 3
k = n + m + 1

c = zeros(n)
A = Diagonal(ones(n))
b = zeros(n)

"Convex.jl"
x = Variable(n)
prob = minimize(c' * x)
prob.constraints += norm(b - A * x) <= 0.0
@time solve!(prob, ECOS.Optimizer)

@show prob.status
@show x.value
@show prob.constraints[1].dual
prob.optval

Q = Array([zeros(n, n) A' c;
     -A zeros(m, m) b;
     -c' -b'      0.0])
Q_vec = vec(Q)

function F(z, Q_vec)
    ũ = z[1:k]
    u = z[k .+ (1:k)]
    v = z[2 * k .+ (1:k)]

    [(I + reshape(Q_vec, k, k)) * ũ - (u + v);
     u - P_soc(ũ - v);
     ũ - u]
end

z = rand(3k)
F(z, Q_vec)
Fz(x) = F(x, Q_vec)
FQ(x) = F(z, x)
Jz = ForwardDiff.jacobian(Fz, z)
JQ = ForwardDiff.jacobian(FQ, Q_vec)

norm((Jz' * Jz) \ (Jz' * JQ), Inf)
