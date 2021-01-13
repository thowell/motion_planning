using LinearAlgebra, ForwardDiff
using Convex, SCS, ECOS

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
