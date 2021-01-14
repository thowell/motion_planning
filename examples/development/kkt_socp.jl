using LinearAlgebra, ForwardDiff, IterativeSolvers

function Pκ(z)
    z0 = z[1]
    z̄ = z[2:end]
    n = length(z)

    η1 = z0 - norm(z̄)
    η2 = z0 + norm(z̄)

    w̄ = ones(n - 1) ./ norm(ones(n - 1))

    if all(z̄ .!= 0.0)
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

    if z0 != norm(z̄) && z0 != norm(z̄)
        if z0 < -norm(z̄)
            @warn "projection weirdness"
            return zeros(n, n)
        elseif z0 > norm(z̄)
            return Diagonal(ones(n))
        elseif -norm(z̄) < z0 && z0 < norm(z̄)
            w̄ = z̄ ./ norm(z̄)
            H = ((1.0 + z0 / norm(z̄)) * Diagonal(ones(n-1))
                - z0 / norm(z̄) * w̄ * w̄')
            return 0.5 * [1.0 w̄'; w̄ H]
        end

    else
        if any(z̄ .!= 0.0) && z0 == norm(z̄)
            V1 = Diagonal(ones(n))

            w̄ = z̄ ./ norm(z̄)
            H = (2.0 * Diagonal(ones(n-1))
                - w̄ * w̄')
            V2 = 0.5 * [1.0 w̄'; w̄ H]
            return V2
        elseif any(z̄ .!= 0.0) && z0 == -norm(z̄)
            V1 = zeros(n, n)

            w̄ = z̄ ./ norm(z̄)
            H = - w̄ * w̄'
            V2 = 0.5 * [1.0 w̄'; w̄ H]
            return V2
        elseif all(z̄ .== 0.0) && z0 .== 0.0
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

z = zeros(3)

Pκ(z)
∇Pκ(z)

ForwardDiff.jacobian(Pκ, z)
# - ∇Pκ(z)
#
# function Πsoc(z)
# 	s = z[1]
# 	v = z[2:end]
#
# 	z_proj = zero(z)
#
# 	if norm(v) <= -s
# 		# @warn "below cone"
# 		z_proj[1] = 0.0
# 		z_proj[2:end] .= 0.0
# 	elseif norm(v) <= s
# 		# @warn "in cone"
# 		z_proj .= copy(z)
# 	elseif norm(v) > abs(s)
# 		# @warn "outside cone"
# 		a = 0.5 * (1.0 + s / norm(v))
# 		z_proj[1] = a * norm(v)
# 		z_proj[2:end] = a * v
# 	else
# 		@warn "soc projection error"
# 		z_proj .= 0.0
# 	end
#
# 	return z_proj
# end
#
# Πsoc(z)
# ForwardDiff.jacobian(Πsoc, z)

# example 3.6
ϵ = 0.1
f(x) = 0.5 * x[1]^2.0 + 0.5 * (x[2] - 2.0)^2.0 - 0.5 * ϵ * x[3]^2.0

m = 0
A = zeros(m, n)
b = zeros(m)

function M(z)
    x = z[1:n]
    λ = z[n .+ (1:n)]
    μ = z[2n .+ (1:m)]

    return [ForwardDiff.gradient(f, x) - A' * μ - λ;
            A * x - b;
            x - Pκ(x - λ)]
end

function ∇M(z)
    x = z[1:n]
    λ = z[n .+ (1:n)]
    μ = z[2n .+ (1:m)]

    return [ForwardDiff.hessian(f, x) -A' -Diagonal(ones(n));
            A zeros(m, m) zeros(m, n)
            (Diagonal(ones(n)) - ∇Pκ(x - λ)) zeros(n, m) ∇Pκ(x - λ)]
end

z = rand(2n + m)
M(z)
∇M(z)

function solve()
    x = zeros(n)
    λ = zeros(n)
    μ = zeros(m)
    z = [x; λ; μ]

    ρ = 1.0e-5

    for i = 1:10
        r = M(z)
        println("norm: $(norm(r))")

        if norm(r) < 1.0e-8
            return z
        end

        ∇r = ∇M(z)
        Δ = -1.0 * (∇r' * ∇r + ρ * I) \ (∇r' * r)
        # Δ = -1.0 * ∇r \ r
        # Δ = gmres(∇r, -1.0 * r)

        α = 1.0
        iter = 0
        while norm(M(z + α * Δ)) > (1.0 - 0.001 * α) * norm(r)
            α = 0.5 * α
            iter += 1

            if iter > 25
                @error "line search failed"
            end
        end

        z .+= α * Δ

    end

    return z
end

z_sol = solve()
