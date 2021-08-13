function pack(X, U)
    z = zeros(num_var - p)
    for i = 1:N
	    for t = 1:T
	        z[(i - 1) * num_traj .+ x_idx[t]] = X[i][t]
	        t == T && continue
	        z[(i - 1) * num_traj .+ u_idx[t]] = U[i][t]
	    end
	end
    return z
end

function unpack(z)
    x = [[z[(i - 1) * num_traj .+ x_idx[t]] for t = 1:T] for i = 1:N]
    u = [[z[(i - 1) * num_traj .+ u_idx[t]] for t = 1:T-1] for i = 1:N]
    return x, u
end

function primal_bounds(x1, xT)
    zl = -Inf * ones(num_var)
    zu = Inf * ones(num_var)

	for i = 1:N
		zl[(i - 1) * num_traj .+ x_idx[1]] = x1[i]
		zu[(i - 1) * num_traj .+ x_idx[1]] = x1[i]

		# zl[(i - 1) * num_traj .+ x_idx[T]] = xT[i]
		# zu[(i - 1) * num_traj .+ x_idx[T]] = xT[i]
	end

    return zl, zu
end

function constraint_bounds()
    cl = zeros(num_con)
    cu = zeros(num_con)
    return cl, cu
end

param_cost = 1.0e-3
function eval_objective(prob::MOIGeneralProblem, z)
	J = 0.0
	for i = 1:N
		for t = 1:T
			if t < T
				x = z[(i - 1) * num_traj .+ x_idx[t]]
				u = z[(i - 1) * num_traj .+ u_idx[t]]
				J += objective(x, u, t) / (N * T)
			else
				x = z[(i - 1) * num_traj .+ x_idx[t]]
				J += objective(x, nothing, t)
			end
		end
	end
	# parameter regularization
	θ = z[p_idx]
	J += param_cost * θ' * θ
	return J
end

function eval_objective_gradient!(∇J, z, prob::MOIGeneralProblem)
    ∇J .= 0.0
	for i = 1:N
		for t = 1:T
			if t < T
				x = z[(i - 1) * num_traj .+ x_idx[t]]
				u = z[(i - 1) * num_traj .+ u_idx[t]]
				obj_x(a) = objective(a, u, t)
				obj_u(a) = objective(x, a, t)
				∇J[(i - 1) * num_traj .+ x_idx[t]] = ForwardDiff.gradient(obj_x, x) / (N * T)
				∇J[(i - 1) * num_traj .+ u_idx[t]] = ForwardDiff.gradient(obj_u, u) / (N * T)
			else
				x = z[(i - 1) * num_traj .+ x_idx[t]]
				obj_xT(a) = objective(a, nothing, t)
				∇J[(i - 1) * num_traj .+ x_idx[t]] = ForwardDiff.gradient(obj_xT, x)
			end
		end
	end
	θ = z[p_idx]
	∇J[p_idx] = 2.0 * param_cost * θ

    return nothing
end

function eval_constraint!(c, z, prob::MOIGeneralProblem)
	# policy
	θ = z[p_idx]

	# dynamics
	for i = 1:N
	    for t = 1:T-1
			x = z[(i - 1) * num_traj .+ x_idx[t]]
			u = z[(i - 1) * num_traj .+ u_idx[t]]
			y = z[(i - 1) * num_traj .+ x_idx[t+1]]
			c[(i - 1) * num_dyn .+ n_idx[t]] = dynamics(y, x, u, t)
		end

		for t = 1:T-1
			x = z[(i - 1) * num_traj .+ x_idx[t]]
			u = z[(i - 1) * num_traj .+ u_idx[t]]
			c[N * num_dyn + (i - 1) * num_policy .+ (m_idx[t])] = u - policy(x, θ)
		end
	end
    return nothing
end

function eval_constraint_jacobian!(∇c, z, prob::MOIGeneralProblem)
	shift = 0

	# policy
	θ = z[p_idx]

	for i = 1:N
	    for t = 1:T-1
	        x = view(z, (i - 1) * num_traj .+ x_idx[t])
	        u = view(z, (i - 1) * num_traj .+ u_idx[t])
	        y = view(z, (i - 1) * num_traj .+ x_idx[t + 1])

	        dyn_x(a) = dynamics(y, a, u, t)
	        dyn_u(a) = dynamics(y, x, a, t)
	        dyn_y(a) = dynamics(a, x, u, t)

	        r_idx = (i - 1) * num_dyn .+ n_idx[t]

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

		for t = 1:T-1
			x = z[(i - 1) * num_traj .+ x_idx[t]]
			u = z[(i - 1) * num_traj .+ u_idx[t]]

			px(a) = policy(a, θ)
			pθ(a) = policy(x, a)

			r_idx = N * num_dyn + (i - 1) * num_policy .+ m_idx[t]

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
	end

    return nothing
end

function sparsity_jacobian(prob::MOIGeneralProblem)

	row = []
	col = []

	for i = 1:N
		for t = 1:T-1
			r_idx = (i - 1) * num_dyn .+ n_idx[t]
			row_col!(row, col, r_idx, (i - 1) * num_traj .+ x_idx[t])
			row_col!(row, col, r_idx, (i - 1) * num_traj .+ u_idx[t])
			row_col!(row, col, r_idx, (i - 1) * num_traj .+ x_idx[t + 1])
		end

		for t = 1:T-1
			r_idx = N * num_dyn + (i - 1) * num_policy .+ m_idx[t]
			row_col!(row, col, r_idx, (i - 1) * num_traj .+ u_idx[t])
			row_col!(row, col, r_idx, (i - 1) * num_traj .+ x_idx[t])
			row_col!(row, col, r_idx, p_idx)
		end
	end

	return collect(zip(row, col))
end

## Trajectory Optimization

# time
T = 101
h = 0.05

include_model("acrobot")

# dynamics
n = 4
m = 1

function dynamics(y, x, u, t)
	fd(model, y, x, u, zeros(model.d), h, t)
end

function dynamics(x, u, t)
	fd(model, x, u, zeros(model.d), h, t)
end

# objective

function objective(x, u, t)
	J = 0.0
	if t == T
		Q = Diagonal(ones(n))
		J += 100.0 * (x - xT)' * Q  * (x - xT)
	else
		Q = Diagonal([1.0; 1.0; 25.0; 25.0])
		R = Diagonal(1.0 * ones(m))
		J += (x - xT)' * Q * (x - xT)
		J += transpose(u) * R * u
	end
	return J
end

# policy
l1 = 4
l2 = 4
function policy(x, θ)

    x_mean = mean(x)
    x_std = π
    x_input = (x .- x_mean) ./ x_std
   # layer 1
   W1 = reshape(θ[1:(l1 * n)], l1, n)
   b1 = θ[l1 * n .+ (1:l1)]

   z1 = W1 * x_input + b1
   o1 = tanh.(z1)

   # layer 2
   W2 = reshape(θ[l1 * n + l1 .+ (1:(l2 * l1))], l2, l1)
   b2 = θ[l1 * n + l1 + l2 * l1 .+ (1:l2)]

   z2 = W2 * o1 + b2

   o2 = tanh.(z2)

   W3 = reshape(θ[l1 * n + l1 + l2 * l1 + l2 .+ (1:(m * l2))], m, l2)
   b3 = θ[l1 * n + l1 + l2 * l1 + l2 + m * l2 .+ (1:m)]

   return W3 * o2 + b3
end

p = l1 * n + l1 + l2 * l1 + l2 + m * l2 + m


# trajectory indices
x_idx = [(t - 1) * (n + m) .+ (1:n) for t = 1:T]
u_idx = [(t - 1) * (n + m) + n .+ (1:m) for t = 1:T-1]
n_idx = [(t - 1) * n .+ (1:n) for t = 1:T-1]
m_idx = [(t - 1) * m .+ (1:m) for t = 1:T-1]

N = 8

# MOI setup
num_x = n * T
num_u = m * (T - 1)
num_traj = num_x + num_u
num_var = N * num_traj + p

num_dyn = n * (T - 1)
num_policy = m * (T - 1)
num_con = N * (num_dyn + num_policy)

p_idx = N * num_traj .+ (1:p)

# initialization
x1 = [0.0; 0.0; 0.0; 0.0]
xT = [π; 0.0; 0.0; 0.0]
X1 = [x1 + (i == 1 ? 0.0 : 1.0) * [0.1; 0.1; 0.01; 0.01] .* randn(n) for i = 1:N]
X = [linear_interpolation(X1[i], xT, T) for i = 1:N]
U = [[0.01 * randn(m) for t = 1:T-1] for i = 1:N]

z = [pack(X, U); 0.01 * randn(p)]
z0 = rand(num_var)
X_, U_ = unpack(z0)

prob = MOIGeneralProblem(num_var, num_con,
	primal_bounds(X1, [xT for i = 1:N]),
	constraint_bounds())
prob_moi = moi_problem(prob)

z_sol, info = solve(prob_moi, z,
		tol = 1.0e-3,
		c_tol = 1.0e-3,
		max_iter = 10000,
		nlp = :ipopt)

x_sol, u_sol = unpack(z_sol)
θ_sol = z_sol[p_idx]

using Plots

plt = plot();
for i = 1:N
	plt = plot!(hcat(x_sol[i]...)', label = "")
end
display(plt)

@show x_sol[1][end]

plt = plot();
for i = 1:N
	plt = plot!(hcat(u_sol[i]..., u_sol[i][end])', linetype = :steppost)
end
display(plt)

## simulate policy
N_sim = 1
x_sim = []
u_sim = []
for i = 1:N_sim
    x1_sim = copy(x1) + 0.0 * randn(n)

    x_hist = [x1_sim]
    u_hist = []
    for t = 1:T-1
        println("t = $t")
    	push!(u_hist, policy(x_hist[end], θ_sol))
    	push!(x_hist, dynamics(x_hist[end], u_hist[end], t))
    end
    push!(x_sim, x_hist)
    push!(u_sim, u_hist)
end

plt = plot();
for i = 1:N_sim
	plt = plot!(hcat(x_sim[i]...)', label = "")
end
display(plt)

plt = plot();
for i = 1:N_sim
	plt = plot!(hcat(u_sim[i]..., u_sim[i][end])', linetype = :steppost)
end
display(plt)
