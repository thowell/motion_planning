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

	zl[x_idx[T]] = xT
	zu[x_idx[T]] = xT

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
		Q = Diagonal(10.0 * ones(n))
		J += x' * Q  * x
	else
		Q = Diagonal(0.1 * ones(n))
		R = Diagonal(0.01 * ones(m))
		J += x' * Q  * x
		J += transpose(u) * R * u
	end
	return J
end

# policy
H = 5
li = H * 2
l1 = m#5
# l2 = m
# l2 = m
function policy(o, u, θ)
	# input
	# input = vec(o) #[vec(o); vec(u)]
	# mu = mean(input)
	#
	#
	# # layer 1
	# M1 = reshape(θ[1:(l1 * li)], l1, li)
	# b1 = θ[l1 * li .+ (1:l1)]
	# z1 = tanh.(M1 * (input .- mu) + b1)
	#
	# # layer 2
	# M2 = reshape(θ[l1 * li + l1 .+ (1:(l2 * l1))], l2, l1)
	# b2 = θ[l1 * li + l1 + l2 * l1 .+ (1:l2)]
	# z2 = M2 * z1 + b2
	#
	# # output
	# return z2
	return reshape(θ[1:(l1 * li)], l1, li) * [vec(o); vec(u)]
	# # layer 1
	# W1 = reshape(θ[1:(l1 * n)], l1, n)
	# b1 = θ[l1 * n .+ (1:l1)]
	#
	# z1 = W1 * x + b1
	# o1 = tanh.(z1)
	#
	# # layer 2
	# W2 = reshape(θ[l1 * n + l1 .+ (1:(l2 * l1))], l2, l1)
	# b2 = θ[l1 * n + l1 + l2 * l1 .+ (1:l2)]
	#
	# z2 = W2 * o1 + b2
	#
	# return z2
	# o2 = tanh.(z2)
	#
	# W3 = reshape(θ[l1 * n + l1 + l2 * l1 + l2 .+ (1:(m * l2))], m, l2)
	# b3 = θ[l1 * n + l1 + l2 * l1 + l2 + m * l2 .+ (1:m)]
	#
	# return W3 * o2 + b3
end

p = l1 * li #+ l1 + l2 * l1 + l2#l1 * n + l1 + l2 * l1 + l2
# p = l1 * n + l1 + l2 * l1 + l2 #+ m * l2 + m

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
x1_alt = [-1.0; 0.0]
xT = [0.0; 0.0]
# X = linear_interpolation(x1, xT, T)
# U = [0.01 * randn(m) for t = 1:T-1]

function cost(x1, θ)
	x = zeros(eltype(θ), n, T)
	u = zeros(eltype(θ), m, T-1)
	y = zeros(eltype(θ), 1, T)

	u_policy = zeros(eltype(θ), m, T + H)
	y_policy = zeros(eltype(θ), 1, T + H)



	x[:, 1] = copy(x1)
	y[1, 1] = copy(abs(x1[1]))

	y_policy[1, H+1] = y[1, 1]

	J = 0.0
	for t = 1:T-1
		# simulate
		# u[:, t] = policy(x[:, t], nothing, θ)
		u[:, t] = policy(y_policy[:, t .+ (1:H)], u_policy[:, t .+ (1:H)], θ)
		J += objective(x[:, t], u[:, t], t)

		x[:, t + 1] = A * x[:, t] + B * u[:, t]
		y[1, t + 1] = copy(abs(x[t + 1]))

		# update policy history
		y_policy[1, t + H+1] = y[1, t + 1]
		u_policy[1, t + H+1] = u[1, t]
	end

	J += objective(x[:, T], nothing, T)

	return J
end

θ0 = 0.01 * randn(p)
policy(zeros(1, H), zeros(1, H), θ0)



cost(x1, θ0)
∇cost(θ) = ForwardDiff.gradient(a -> cost(x1, a), θ)
∇cost(θ0)

function optimize()
	θ = 1.0e-3 * randn(p)

	for i = 1:10
		# w1 = 0.001 * randn(n)
		# w1_alt = 0.001 * randn(n)
		#
		# J = 0.5 * (cost(x1 + w1, θ) + cost(x1_alt + w1_alt, θ))
		# ∇J = 0.5 * (ForwardDiff.gradient(a -> cost(x1 + w1, a), θ) + ForwardDiff.gradient(a -> cost(x1_alt + w1_alt, a), θ))

		J = cost(x1, θ)
		∇J = ForwardDiff.gradient(a -> cost(x1, a), θ)
		∇²J = ForwardDiff.hessian(a -> cost(x1, a), θ)

		Δ = ∇²J \ ∇J
		α = 1.0
		θ_cand = θ - α * Δ
		iter = 1
		while cost(x1, θ_cand) >= J + 0.0001 * α * ∇J' * Δ
			iter += 1
			α *= 0.5
			θ_cand = θ - α * Δ

			if iter > 100
				@error "line search failure"
			end
		end

		θ = θ_cand

		if i % 10 == 0
			println("iter: ", i)
			println("	J: ", J)
			println("	norm(∇J): ", norm(∇J))
		end
	end

	return θ
end

θ_sol = optimize()

# plot(hcat(x_roll...)')
# plot(hcat(u_roll..., u_roll[end])', linetype = :steppost)
#
# z = [pack(x_roll, u_roll); θ0]
#
# prob = MOIGeneralProblem(num_var, num_con, primal_bounds(x1, xT), constraint_bounds())
# prob_moi = moi_problem(prob)
#
# z_sol, info = solve(prob_moi, z,
# 		tol = 1.0e-3,
# 		c_tol = 1.0e-3,
# 		max_iter = 1000,
# 		nlp = :ipopt)
#
# x_sol, u_sol = unpack(z_sol)
# θ_sol = z_sol[num_x + num_u .+ (1:p)]
#
# using Plots
#
# plot(hcat(x_sol...)')
# plot(hcat(u_sol..., u_sol[end])', linetype = :steppost)


## simulate policy
x1_sim = copy(x1) + 0.0 * randn(n)

x_hist = zeros(n, T)
x_hist[:, 1] = copy(x1)
u_hist = zeros(m, T)
y_hist = zeros(1, T)
y_hist[1, 1] = abs(x_hist[1, 1])

u_policy = zeros(m, T + H)
y_policy = zeros(1, T + H)
y_policy[1, H + 1] = y_hist[1, 1]
for t = 1:T-1
	u_hist[:, t] = policy(y_policy[:, t .+ (1:H)], u_policy[:, t .+ (1:H)], θ_sol)
	# u_hist[:, t] = policy(x_hist[:, t], u_policy[:, t .+ (1:H)], θ_sol)

	x_hist[:, t + 1] = A * x_hist[:, t] + B * u_hist[:, t]
	y_hist[1, t + 1] = abs(x_hist[1, t + 1])

	# update policy history
	y_policy[1, t + H+1] = y_hist[1, t + 1]
	u_policy[1, t + H+1] = u_hist[1, t]
end

plot(x_hist')
plot(vec(u_hist), linetype = :steppost)

u_hist
