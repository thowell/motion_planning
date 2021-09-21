using Plots
using Random
Random.seed!(1)

include_ddp()

# Model
include_model("double_integrator")

function f(model::DoubleIntegratorContinuous, x, u, w)
    [x[2]; (1.0 + w[1]) * u[1]]
end

function fd(model::DoubleIntegratorContinuous{Midpoint, FixedTime}, x, u, w, h, t)
	x + h * f(model, x + 0.5 * h * f(model, x, u, w), u, w)
end

model = DoubleIntegratorContinuous{Midpoint, FixedTime}(2, 1, 1)
n = model.n
m = model.m

# Time
T = 11
h = 0.1

# reference
x_ref = [[0.0; 0.0] for t = 1:T]
xT = [0.0; 0.0]
ū_init = [1.0 * randn(model.m) for t = 1:T-1]
u_ref = [zeros(model.m) for t = 1:T-1]
w = [zeros(model.d) for t = 1:T-1]

# Objective
Q = [(t < T ? h : 1.0) * (t < T ?
	 Diagonal([1.0; 1.0])
		: Diagonal([1.0; 1.0])) for t = 1:T]
q = [-2.0 * Q[t] * x_ref[t] for t = 1:T]
R = h * [Diagonal(1.0 * ones(model.m)) for t = 1:T-1]
r = [zeros(model.m) for t = 1:T-1]

obj = StageCosts([QuadraticCost(Q[t], q[t],
	t < T ? R[t] : nothing, t < T ? r[t] : nothing) for t = 1:T], T)

function g(obj::StageCosts, x, u, t)
	T = obj.T
    if t < T
		Q = obj.cost[t].Q
		q = obj.cost[t].q
	    R = obj.cost[t].R
		r = obj.cost[t].r
        return x' * Q * x + q' * x + u' * R * u + r' * u
    elseif t == T
		Q = obj.cost[T].Q
		q = obj.cost[T].q
        return x' * Q * x + q' * x
    else
        return 0.0
    end
end

# Constraints
ul = [-5.0]
uu = [5.0]
p = [t < T ? 2 * m : n for t = 1:T]
info_t = Dict(:ul => ul, :uu => uu, :inequality => (1:2 * m))
info_T = Dict(:xT => xT)
con_set = [StageConstraint(p[t], t < T ? info_t : info_T) for t = 1:T]

function c!(c, cons::StageConstraints, x, u, t)
	T = cons.T
	p = cons.con[t].p

	if t < T
		ul = cons.con[t].info[:ul]
		uu = cons.con[t].info[:uu]
		c .= [ul - u; u - uu]
	else
		c .= x - cons.con[T].info[:xT]
	end
end

# generate data
N = 1
x_data = zeros(model.n, 0) 
u_data = zeros(model.m, 0) 

# offset = [[0.0; 0.0], [-2.0; 0.0]]
for i = 1:N
	# Initial conditions, controls, disturbances
	x1 = [1.0; 0.0] + 0.0 * randn(model.n)
	# x1 = [1.0; 0.0] + offset[i]

	# Rollout
	x̄ = rollout(model, x1, copy(ū_init), w, h, T)

	prob = problem_data(model, obj, con_set, copy(x̄), copy(ū_init), w, h, T,
		analytical_dynamics_derivatives = false)

	# Solve
	@time constrained_ddp_solve!(prob,
		linesearch = :armijo,
		max_iter = 1000, 
		max_al_iter = 10,
		ρ_init = 1.0, 
		ρ_scale = 10.0,
		con_tol = 1.0e-5)

	x, u = current_trajectory(prob)
	x̄, ū = nominal_trajectory(prob)

	x_data = hcat(x_data, hcat(x[1:end-1]...))
	u_data = hcat(u_data, hcat(u...))
end

# Visualize
# plot(hcat([[0.0; 0.0] for t = 1:T]...)',
#     width = 2.0, color = :black, label = "")
# plt = plot(hcat(x...)',
# 	width = 2.0,
# 	color = [:cyan :orange],
# 	label = ["x" "ẋ"],
# 	xlabel = "time step",
# 	ylabel = "state")

# policy 
pass_through(x) = x
l = [model.n, 2, 2, model.m] # input -> output layers
activation = [:tanh, :tanh, :pass_through]
np = sum([l[i] * l[i+1] for i = 1:length(l)-1]) + sum(l[2:end])
reg_θ = 1.0e-5 

function policy(x, θ) 
	W = [] 
	B = []
	shift = 0 
	
	p = x 

	for i = 1:length(l)-1
		s = l[i] * l[i+1]
		w = θ[shift .+ (1:s)] 
		shift += s 

		b = θ[shift .+ (1:l[i+1])] 
		shift += l[i+1] 
		# p = reshape(w, l[i+1], l[i]) * p + b

		p = eval(activation[i]).(reshape(w, l[i+1], l[i]) * p + b)
	end

	return p
end

policy(ones(model.n), randn(np))

@variables x_sym[1:model.n], u_sym[1:model.m], θ_sym[1:np] 

p_sym = policy(x_sym, θ_sym)
pθ_sym = Symbolics.jacobian(p_sym, θ_sym) 

p_func = eval(Symbolics.build_function(p_sym, x_sym, θ_sym)[1])
pθ_func = eval(Symbolics.build_function(pθ_sym, x_sym, θ_sym)[1])

p_func(ones(model.n), randn(np))
pθ_func(ones(model.n), randn(np))

# loss 
function err(θ, x, u) 
	r = u - policy(x, θ)

	return 0.5 * norm(r)^2.0
end

e_sym = err(θ_sym, x_sym, u_sym)
eθ_sym = Symbolics.gradient(e_sym, θ_sym)
eθ²_sym = Symbolics.sparsehessian(e_sym, θ_sym)

eθ_func = eval(Symbolics.build_function(eθ_sym, θ_sym, x_sym, u_sym)[1])
eθ²_func = eval(Symbolics.build_function(eθ²_sym, θ_sym, x_sym, u_sym)[1])

eθ_func(ones(np), ones(model.n), ones(model.m)) 
eθ²_func(ones(np), ones(model.n), ones(model.m)) 

function mse(θ, x, u) 
	N = size(x, 2)
	J = 0.0 

	for t = 1:N
		J += 0.5 * norm(u[:, t] - p_func(x[:, t], θ))^2.0
	end 

	return J / N + reg_θ * transpose(θ) * θ
end

mse(randn(np), x_data, u_data)

p_func(x_data[:, 1], randn(np)) - u_data[:, 1]

function mse_grad(θ, x, u) 
	N = size(x, 2)
	grad = zero(θ) 

	for t = 1:N
		grad .+= -1.0 * transpose(pθ_func(x[:, t], θ)) * (u[:, t] - p_func(x[:, t], θ))
		# grad .+= eθ_func(θ, x[:, t], u[:, t])
	end 

	return grad ./ N + reg_θ * 2.0 * θ
end

norm(ForwardDiff.gradient(a -> mse(a, x_data, u_data), ones(np)) - mse_grad(ones(np), x_data, u_data))

function mse_hess(θ, x, u) 
	N = size(x, 2)
	np = length(θ) 
	hess = zeros(np, np)

	for t = 1:N
		hess .+= eθ²_func(θ, x[:, t], u[:, t])
	end 

	return hess ./ N .+ reg_θ * 2.0
end

mse_hess(randn(np), x_data, u_data)

function gd(θ0, x_data, u_data; iters=1000, verbose=true, α=0.001)
    θ = copy(θ0)

	grad = zero(θ) 

    for i = 1:iters
        grad = mse_grad(θ, x_data, u_data)
        θ -= α * grad

		if verbose && (i % 500 == 0)
			println("iter: $i")
			println("obj: ", mse(θ, x_data, u_data))
			println("grad. norm: ", norm(grad))
		end
    end


    return θ
end

function gd_ls(θ0, x_data, u_data; grad_tol = 1.0e-5, iters=100, verbose=true, )
    θ = copy(θ0)
	obj = mse(θ, x_data, u_data)
	grad = mse_grad(θ, x_data, u_data)

	iter = 1
	while norm(grad, 1) > grad_tol && iter < iters
		# obj = mse(θ, x_data, u_data)
		# grad = mse_grad(θ, x_data, u_data)

		# hess = mse_hess(θ, x_data, u_data)
		# reg = -1.0 * minimum(min.(eigen(mse_hess(θ, x_data, u_data)).values, 0.0)) + 1.0e-5

		Δ = grad

		α = 1.0
		while α > 1.0e-8 
			θ_cand = θ - α * Δ 
			obj_cand = mse(θ_cand, x_data, u_data)
			grad_cand = mse_grad(θ_cand, x_data, u_data)

			cond1 = obj_cand <= obj - 1.0e-4 * α * transpose(Δ) * grad 
			# cond2 = transpose(Δ) * grad_cand <= 0.9 * transpose(Δ) * grad 
			if cond1 #&& cond2
				θ = θ_cand 
				obj = obj_cand
				grad = grad_cand 
				break
			else 
				α *= 0.5 
			end
		end
		# println("alpha: $α")

		if verbose && (iter % 100 == 0)
			if α <= 1.0e-8 
				@warn "line search failure"
			end
			println("iter: $iter")
			println("obj: ", mse(θ, x_data, u_data))
			println("grad. norm: ", norm(grad))
		end

		iter += 1
    end

    return θ
end

# -1.0 * minimum(min.(eigen(mse_hess(θ0, x_data, u_data)).values, 0.0))

function newton(θ0, x_data, u_data; grad_tol = 1.0e-5, iters=100, verbose=true, )
    θ = copy(θ0)
	obj = mse(θ, x_data, u_data)
	grad = mse_grad(θ, x_data, u_data)

	iter = 1
	while norm(grad, 1) > grad_tol && iter < iters
		# obj = mse(θ, x_data, u_data)
		# grad = mse_grad(θ, x_data, u_data)

		hess = mse_hess(θ, x_data, u_data)
		reg = -1.0 * minimum(min.(eigen(mse_hess(θ, x_data, u_data)).values, 0.0)) + 1.0e-5

		Δ = (hess + reg * I) \ grad

		α = 1.0
		while α > 1.0e-8 
			θ_cand = θ - α * Δ 
			obj_cand = mse(θ_cand, x_data, u_data)
			grad_cand = mse_grad(θ_cand, x_data, u_data)

			cond1 = obj_cand <= obj - 1.0e-4 * α * transpose(Δ) * grad 
			# cond2 = transpose(Δ) * grad_cand <= 0.9 * transpose(Δ) * grad 
			if cond1 #&& cond2
				θ = θ_cand 
				obj = obj_cand
				grad = grad_cand 
				break
			else 
				α *= 0.5 
			end
		end
		# println("alpha: $α")

		if verbose && (iter % 1 == 0)
			if α <= 1.0e-8 
				@warn "line search failure"
			end
			println("iter: $iter")
			println("obj: ", mse(θ, x_data, u_data))
			println("grad. norm: ", norm(grad))
		end

		iter += 1
    end

    return θ
end

# optimize
θ0 = randn(np) 
θ_sol = gd(θ0, x_data, u_data, α=0.05, iters=100000);
# θ_sol = gd(θ_sol, x_data, u_data, α=0.05, iters=1000);

θ_sol = gd_ls(θ0, x_data, u_data, iters=10000);
θ_sol = newton(θ0, x_data, u_data, grad_tol=1.0e-5, iters=1000);

# rollout policy
x_data[:, 1]
# x_roll = [[1.0; 0.0] + offset[2]]
x_roll = [[1.0; 0.0] + 0.00 * randn(model.n)]
u_roll = [] 
for t = 1:T-1 
	push!(u_roll, policy(x_roll[end], θ_sol)) 
	push!(x_roll, fd(model, x_roll[end], 
		u_roll[end],
		# u_data[:, t], 
		zeros(model.d), h, t)) 
end


plt = plot(hcat(x_roll...)',
	width = 2.0,
	color = [:cyan :orange],
	label = ["x" "ẋ"],
	xlabel = "time step",
	ylabel = "state")

x_roll[end]