using LinearAlgebra, ForwardDiff, Distributions
using Convex, SCS, ECOS
using IterativeSolvers

# problem setup
p = 60
N = 30
x_sol = max.(0.0, randn(p))
A1 = Diagonal(-1.0 * ones(p))
b1 = zeros(p)
A2 = randn(N, p)
b2 = A2 * x_sol
ν_sol = randn(N)
λ_sol = rand(p)
λ_sol[x_sol .> 0.0] .= 0.0
c = -A2' * ν_sol + λ_sol

"Convex.jl"
x = Variable(p)
prob = minimize(c' * x)
prob.constraints += b2 - A2 * x == 0.0
prob.constraints += b1 - A1 * x >= 0.0

@time solve!(prob, ECOS.Optimizer)

@show prob.status
@show x.value
# @show prob.constraints[1].dual
# @show prob.constraints[2].dual
A = [A1; A2; -A2]
b = [b1; b2; -b2]
m = 2N + p
k = p + m + 1

Q = Array([zeros(p, p) A' c;
     -A zeros(m, m) b;
     -c' -b'      0.0])

Q_vec = vec(Q)

function Pκ(z)
    z[p .+ (1:m)] = max.(0.0, z[p .+ (1:m)])
    z[end] = max.(0.0, z[end])
    z
end

function F(z)
    ũ = z[1:k]
    u = z[k .+ (1:k)]
    v = z[2 * k .+ (1:k)]

    [(I + Q) * ũ - (u + v);
     u - Pκ(ũ - v);
     ũ - u]
end

function Ju(z)
    ũ = z[1:k]
    u = z[k .+ (1:k)]
    v = z[2k .+ (1:k)]

    # JP = ForwardDiff.jacobian(Pκ, ũ[p .+ (1:m)] - v[p .+ (1:m)])

    JP = zeros(m, m)
    dif = ũ[p .+ (1:m)] - v[p .+ (1:m)]
    for i = 1:m
        if dif[i] >= 0.0
            JP[i, i] = 1.0
        end
    end

    if z[k] - z[3k] >= 0.0
        ℓ = 1.0
    else
        ℓ = 0.0
    end

    [-I zeros(p, m) zeros(p) I zeros(p, m) zeros(p) I zeros(p, m) zeros(p);
     zeros(m, p) -JP zeros(m) zeros(m, p) I zeros(m) zeros(m, p) JP zeros(m);
     [zeros(1, p) zeros(1, m) -ℓ zeros(1, p) zeros(1, m) 1.0 zeros(1, p) zeros(1, m) ℓ]]
end

function J(z)
    [(I + Q) -I -I
     Ju(z);
     I -I zeros(k, k)]
end

ũ = zeros(k)
u = zeros(k)
v = zeros(k)
ũ[end] = 1.0
u[end] = 1.0
v[end] = 1.0

z = [ũ; u; v]
# z = rand(3k)
F(z)
rank(Ju(z))
J(z)
rank(ForwardDiff.jacobian(F, z))
norm(J(z) - ForwardDiff.jacobian(F, z), Inf)

(J(z)' * J(z)) \ (J(z)' * F(z))
Δ = zero(z)
# Δ[k] = 1.0
# Δ[2k] = 1.0
# Δ[3k] = 1.0
gmres!(Δ, J(z), -1.0 * F(z), abstol = 1.0)

function solve()
    ũ = zeros(k)
    u = zeros(k)
    v = zeros(k)
    ũ[end] = 1.0
    u[end] = 1.0
    v[end] = 1.0

    z = [ũ; u; v]
    # z = rand(3k)
    Δ = zero(z)

    for i = 1:100
        # Fz(x) = F(x, Q_vec)
        # FQ(x) = F(z, x)

        _F = F(z)
        _J = J(z) #ForwardDiff.jacobian(F, z)
        # Δ = -1.0 * _J' *  inv(_J * _J') * _F
        # Δ = -1.0 * (_J' * _J) \ (_J' * _F)
        # Δ = rand(length(z))
        gmres!(Δ, _J, -1.0 * _F, abstol = 1.0e-12, maxiter = 1000)

        # print(Δ)
        # @error " stop"
        iter = 0
        α = 1.0
        while norm(F(z + α * Δ))^2.0 >= (1.0 - 0.001 * α) * norm(_F)^2.0
            α = 0.5 * α
            # println("   α = $α")
            iter += 1
            if iter > 100
                @error "line search fail"
                return z
            end
        end
        println("norm: $(norm(F(z)))")
        z .+= α * Δ
    end

    return z
    x = z[k .+ (1:p)]
    s = z[k + k .+ (1:m)]
    τ = z[k + k]
    return x ./ τ
end

x_sol = solve()

x_sol[1:k]
x_sol[k]
x_sol[3k]
F(x_sol)
maximum(x_sol)
G * x_sol - h
minimum(x_sol)

# norm(x_sol - x.value)
