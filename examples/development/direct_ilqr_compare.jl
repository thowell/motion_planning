using LinearAlgebra, SparseArrays, ForwardDiff
using Convex, ECOS

# Dimensions
n = 2 # state
m = 1 # control
T = 2 # horizon
num_var = n * T + m * (T - 1)
num_con = n * T
num_dec = num_var + num_con
z0 = rand(num_dec)

function traj(z)
    x1 = view(z, 1:n)
    u1 = view(z, n .+ (1:m))
    x2 = view(z, n + m .+ (1:n))
    λ1 = view(z, 2 * n + m .+ (1:n))
    λ2 = view(z, 3 * n + m .+ (1:n))
    return x1, u1, x2, λ1, λ2
end

x1, u1, x2, λ1, λ2 = traj(z0)

# Dynamics
x_init = ones(n)

A = [1.0 1.0;
     0.0 1.0]

B = [0.1;
     0.0]

In = Diagonal(ones(n))

# Objective
Q = [1.0 0.0; 0.0 1.0]
R = Array([1.0e-1])


# Convex.jl
y = Variable(num_var) # decision variables
objective = 0.5 * quadform(y[1:n], Q) + 0.5 * quadform(y[n + m .+ (1:n)], Q) + 0.5 * R[1] * square(y[n .+ (1:m)][1])

problem = minimize(objective) # setup problem

# constraint setup
problem.constraints += A * y[1:n] + B * y[n .+ (1:m)][1] - y[n + m .+ (1:n)] == 0.0
problem.constraints += y[1:n] - x_init == 0.0

solve!(problem, ECOS.Optimizer) # solve

# Convex solution
x1_sol = y.value[1:n]
u1_sol = y.value[n .+ (1:m)]
x2_sol = y.value[n + m .+ (1:n)]
λ1_sol = -1.0 * problem.constraints[1].dual
λ2_sol = -1.0 * problem.constraints[2].dual

z_sol = vec([x1_sol; u1_sol; x2_sol; λ1_sol; λ2_sol])

# kkt system r + Hz = 0
function r(z)
    x1, u1, x2, λ1, λ2 = traj(z)
    [Q * x1 + A' * λ1 + λ2;
     (R * u1[1])[1] + B' * λ1;
     Q * x2 - λ1;
     A * x1 + B * u1[1] - x2;
     x1 - x_init]
end
r(z_sol)

H = [Q zeros(n, m) zeros(n, n) A' In;
     zeros(m, n) R zeros(m, n) B' zeros(m, n);
     zeros(n, n) zeros(n, m) Q -In zeros(n, n);
     A B -In zeros(n, n) zeros(n, n);
     In zeros(n, m) zeros(n, n) zeros(n, n) zeros(n, n)]

r(z0)
ForwardDiff.jacobian(r, z0) - H

# function con(z)
#     x1, u1, x2, λ1, λ2 = traj(z)
#
#     [A * x1 + B * u1[1] - x2
#      x1 - ones(2);]
# end
#
# C = [A B -In;
#      In zeros(n, m + n)]
#
# con(z0)
# ForwardDiff.jacobian(con, z0)[:, 1:num_var] - C

# initialize variables
x1 = rand(n) #copy(x_init) #
u1 = rand(m)
x2 = A * x1 + B * u1[1]
λ1 = rand(n) # copy(λ1_sol)
λ2 = rand(n) # copy(λ2_sol)

z = [x1; u1; x2; λ1; λ2]

# kkt solve
Δz = -1.0 * H \ r(z)
Δx1, Δu1, Δx2, Δλ1, Δλ2 = traj(Δz)

# iLQR solve
function compute_Δu1(Δx1)
    Quu = R[1] + B' * Q * B
    Qux = B' * Q * A
    Qu = (R * u1[1])[1] + B' * Q * x2
    -1.0 * Quu \ (Qux * Δx1 + Qu)
end

Δu1[1] - compute_Δu1(Δx1)
