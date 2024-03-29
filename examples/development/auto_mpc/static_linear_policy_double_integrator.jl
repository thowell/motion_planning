function pack(X, U)
    z = zeros(num_var)
    for t = 1:T
        z[x_idx[t]] = X[t]
        t == T && continue
        z[u_idx[t]] = U[t]
    end
    return z
end

function unpack(z)
    x = [z[x_idx[t]] for t = 1:T]
    u = [z[u_idx[t]] for t = 1:T-1]
    return x, u
end

function primal_bounds(x1, xT)
    zl = -Inf * ones(num_var)
    zu = Inf * ones(num_var)

	zl[x_idx[1]] = x1
	zu[x_idx[1]] = x1

	# zl[x_idx[T]] = xT
	# zu[x_idx[T]] = xT

    return zl, zu
end

function constraint_bounds()
    cl = zeros(num_con)
    cu = zeros(num_con)
    return cl, cu
end

function eval_objective(prob::MOIGeneralProblem, z)
	J = 0.0
	for t = 1:T
		if t < T
			x = z[x_idx[t]]
			u = z[u_idx[t]]
			J += objective(x, u, t)
		else
			x = z[x_idx[t]]
			J += objective(x, nothing, t)
		end
	end
	# parameter regularization
	θ = z[p_idx]
	J += 1.0 * θ' * θ
	return J
end

function eval_objective_gradient!(∇J, z, prob::MOIGeneralProblem)
    ∇J .= 0.0
	for t = 1:T
		if t < T
			x = z[x_idx[t]]
			u = z[u_idx[t]]
			obj_x(a) = objective(a, u, t)
			obj_u(a) = objective(x, a, t)
			∇J[x_idx[t]] = ForwardDiff.gradient(obj_x, x)
			∇J[u_idx[t]] = ForwardDiff.gradient(obj_u, u)
		else
			x = z[x_idx[t]]
			obj_xT(a) = objective(a, nothing, t)
			∇J[x_idx[t]] = ForwardDiff.gradient(obj_xT, x)
		end
	end
	θ = z[p_idx]
	∇J[p_idx] = 2.0 * θ

    return nothing
end

function eval_constraint!(c, z, prob::MOIGeneralProblem)
	# dynamics
    for t = 1:T-1
		x = z[x_idx[t]]
		u = z[u_idx[t]]
		y = z[x_idx[t+1]]
		c[n_idx[t]] = dynamics(y, x, u, t)
	end

	# policy
	θ = z[p_idx]
	for t = 1:T-1
		x = z[x_idx[t]]
		u = z[u_idx[t]]
		c[num_dyn .+ (m_idx[t])] = u - policy(x, θ)
	end
    return nothing
end

function eval_constraint_jacobian!(∇c, z, prob::MOIGeneralProblem)
	shift = 0

    for t = 1:T-1
        x = view(z, x_idx[t])
        u = view(z, u_idx[t])
        y = view(z, x_idx[t + 1])

        dyn_x(a) = dynamics(y, a, u, t)
        dyn_u(a) = dynamics(y, x, a, t)
        dyn_y(a) = dynamics(a, x, u, t)

        r_idx = n_idx[t]

        s = n * n
        ∇c[shift .+ (1:s)] = vec(ForwardDiff.jacobian(dyn_x, x))
        shift += s

        s = n * m
        ∇c[shift .+ (1:s)] = vec(ForwardDiff.jacobian(dyn_u, u))
        shift += s

        s = n * n
        ∇c[shift .+ (1:s)] = vec(ForwardDiff.jacobian(dyn_y, y))
        shift += s
    end

	# policy
	θ = z[p_idx]
	for t = 1:T-1
		x = z[x_idx[t]]
		u = z[u_idx[t]]

		px(a) = policy(a, θ)
		pθ(a) = policy(x, a)

		r_idx = num_dyn .+ (m_idx[t])

		s = m * m
		∇c[shift .+ (1:s)] = vec(Diagonal(ones(m)))
		shift += s

		s = m * n
		∇c[shift .+ (1:s)] = -vec(ForwardDiff.jacobian(px, x))
		shift += s

		s = m * p
		∇c[shift .+ (1:s)] = -vec(ForwardDiff.jacobian(pθ, θ))
		shift += s
	end

    return nothing
end

function sparsity_jacobian(prob::MOIGeneralProblem)

	row = []
	col = []

	for t = 1:T-1
		r_idx = n_idx[t]
		row_col!(row, col, r_idx, x_idx[t])
		row_col!(row, col, r_idx, u_idx[t])
		row_col!(row, col, r_idx, x_idx[t + 1])
	end

	for t = 1:T-1
		r_idx = num_dyn .+ (m_idx[t])
		row_col!(row, col, r_idx, u_idx[t])
		row_col!(row, col, r_idx, x_idx[t])
		row_col!(row, col, r_idx, p_idx)
	end

	return collect(zip(row, col))
end

## Trajectory Optimization

# time
T = 11
h = 1.0

# dynamics
n = 2
m = 1

A = SMatrix{n,n}([1.0 1.0; 0.0 1.0])
B = SMatrix{n,m}([0.0; 1.0])

function dynamics(y, x, u, t)
	y - (A * x + B * u)
end

# objective
function objective(x, u, t)
	J = 0.0
	if t == T
		Q = Diagonal(100.0 * ones(n))
		J += x' * Q  * x
	else
		Q = Diagonal(ones(n))
		R = Diagonal(0.1 * ones(m))
		J += x' * Q  * x
		J += transpose(u) * R * u
	end
	return J
end

# policy
function policy(x, θ)
	M = reshape(θ, m, n)
	M * x
end

p = m * n

# trajectory indices
x_idx = [(t - 1) * (n + m) .+ (1:n) for t = 1:T]
u_idx = [(t - 1) * (n + m) + n .+ (1:m) for t = 1:T-1]
n_idx = [(t - 1) * n .+ (1:n) for t = 1:T-1]
m_idx = [(t - 1) * m .+ (1:m) for t = 1:T-1]
p_idx = n * T + m * (T - 1) .+ (1:p)

# MOI setup
num_x = n * T
num_u = m * (T - 1)
num_var = num_x + num_u + p

num_dyn = n * (T - 1)
num_policy = m * (T - 1)
num_con = num_dyn + num_policy

# initialization
x1 = [1.0; 0.0]
xT = [0.0; 0.0]
X = linear_interpolation(x1, xT, T)
U = [0.01 * randn(m) for t = 1:T-1]

z = [pack(X, U); 0.01 * randn(p)]

prob = MOIGeneralProblem(num_var, num_con, primal_bounds(x1, xT), constraint_bounds())
prob_moi = moi_problem(prob)

z_sol, info = solve(prob_moi, z,
		tol = 1.0e-3,
		c_tol = 1.0e-3,
		max_iter = 1000,
		nlp = :ipopt)

x_sol, u_sol = unpack(z_sol)
θ_sol = z_sol[num_x + num_u .+ (1:p)]
M = reshape(θ_sol, m, n)

using Plots

plot(hcat(x_sol...)')
plot(hcat(u_sol..., u_sol[end])', linetype = :steppost)


## simulate policy
x1_sim = copy(x1) + 0.01 * randn(n)

x_hist = [x1_sim]
u_hist = []
for t = 1:T-1
	push!(u_hist, M * x_hist[end])
	# push!(u_hist, u_sol[t])

	push!(x_hist, A * x_hist[end] + B * u_hist[end])
end

plot(hcat(x_hist...)')
plot(hcat(u_hist..., u_hist[end])', linetype = :steppost)
