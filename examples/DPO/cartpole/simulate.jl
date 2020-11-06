using Distributions, Interpolations

function simulate(
		model::CartpoleFriction,
		policy,
		K,
		z_nom, u_nom,
		Q, R,
		T_sim, Δt,
		z0, w;
        _norm = 2,
        ul = -Inf * ones(length(u_nom[1])),
        uu = Inf * ones(length(u_nom[1])),
        friction = false,
        μ = 0.1)

    T = length(K) + 1
    times = [(t-1) * Δt for t = 1:T-1]
    tf = Δt * T
    t_sim = range(0, stop = tf, length = T_sim)
    t_ctrl = range(0, stop = tf, length = T)
    dt_sim = tf / (T_sim - 1)

    p = 1:policy.output

    A_state = hcat(z_nom...)
    A_ctrl = hcat(u_nom...)

    z_rollout = [z0]
    u_rollout = []
    J = 0.0
    Jx = 0.0
    Ju = 0.0
    for tt = 1:T_sim-1
        t = t_sim[tt]
        k = searchsortedlast(times, t)

        z_cubic = zeros(model.n)
        for i = 1:model.n
            interp_cubic = CubicSplineInterpolation(t_ctrl, A_state[i,:])
            z_cubic[i] = interp_cubic(t)
        end

        z = z_rollout[end] + dt_sim * w[:,tt]
        u = eval_policy(policy, K[k], z, z_cubic, u_nom[k])

        # clip controls
        u = max.(u, ul[p])
        u = min.(u, uu[p])

        if friction
            _u = [u[1] - μ * sign(z_cubic[3]) * model.g * (model.mp + model.mc);
				  0.0;
				  0.0]
        else
            _u = u[1]
        end

        push!(z_rollout, rk3(model, z, _u, zeros(model.d), dt_sim))
        push!(u_rollout, u)

        if _norm == 2
            J += (z_rollout[end] - z_cubic)' * Q[k + 1] * (z_rollout[end] - z_cubic)

			J += (u_rollout[end] - u_nom[k][p])' * R[k][p,
                p] * (u_rollout[end] - u_nom[k][p])

			Jx += (z_rollout[end] - z_cubic)' * Q[k + 1] * (z_rollout[end] - z_cubic)

			Ju += (u_rollout[end] - u_nom[k][p])' * R[k][p,
                p] * (u_rollout[end] - u_nom[k][p])
        else
            J += norm(sqrt(Q[k + 1]) * (z_rollout[end] - z_cubic), _norm)
            J += norm(sqrt(R[k][p, p]) * (u - u_nom[k][p]), _norm)
            Jx += norm(sqrt(Q[k + 1]) * (z_rollout[end] - z_cubic), _norm)
            Ju += norm(sqrt(R[k][p, p]) * (u - u_nom[k][p]), _norm)
        end
    end
    return z_rollout, u_rollout, J / (T_sim - 1), Jx / (T_sim - 1), Ju / (T_sim - 1)
	nothing
end

# Nominal trajectories
X̄_nominal, Ū_nominal = unpack(Z̄_nominal, prob_nominal)
X̄_friction, Ū_friction = unpack(Z̄_friction, prob_friction)
X̄_dpo, Ū_dpo = unpack(Z[prob_dpo.prob.idx.nom], prob_dpo.prob.prob.nom)

# Policies
K_nominal = tvlqr(model, X̄_nominal, [Ū_nominal[t][1:1] for t = 1:T-1],
 	Q, [R[t][1:1, 1:1] for t = 1:T-1], h)
K_friction = tvlqr(model, X̄_friction, [Ū_friction[t][1:1] for t = 1:T-1],
 	Q, [R[t][1:1, 1:1] for t = 1:T-1], h)
θ = [reshape(Z[prob_dpo.prob.idx.policy[prob_dpo.prob.idx.θ[t]]],
		1, model.n) for t = 1:T-1]

# Simulation
T_sim = 10T
Δt = h
dt_sim = tf / (T_sim - 1)

W = Distributions.MvNormal(zeros(model.n), Diagonal(1.0e-5 * ones(model.n)))
w = rand(W, T_sim)

W0 = Distributions.MvNormal(zeros(model.n), Diagonal(1.0e-5 * ones(model.n)))
w0 = rand(W0, 1)

model_sim = model_friction
μ_sim = 0.1

t_sim = range(0, stop = tf, length = T_sim)

z_lqr, u_lqr, J_lqr, Jx_lqr, Ju_lqr = simulate(model_sim,
	policy,
	K_nominal,
    X̄_nominal, Ū_nominal,
	Q, R,
	T_sim, h,
	x1 + vec(w0),
	w,
	ul = ul_friction, uu = uu_friction,
    friction = true,
    μ = μ0)

z_lqr_fr, u_lqr_fr, J_lqr_fr, Jx_lqr_fr, Ju_lqr_fr = simulate(model_sim,
	policy,
	K_friction,
    X̄_friction, Ū_friction,
	Q, R,
	T_sim, h,
	x1 + vec(w0),
	w,
	ul = ul_friction, uu = uu_friction,
    friction = true,
    μ = μ0)

z_dpo, u_dpo, J_dpo, Jx_dpo, Ju_dpo = simulate(model_sim,
	policy,
	θ,
    X̄_dpo, Ū_dpo,
	Q, R,
	T_sim, h,
	x1 + vec(w0),
	w,
	ul = ul_friction, uu = uu_friction,
    friction = true,
    μ = μ0)

# objective value
J_lqr
J_lqr_fr
J_dpo

# state tracking
Jx_lqr
Jx_lqr_fr
Jx_dpo

# control tracking
Ju_lqr
Ju_lqr_fr
Ju_dpo
