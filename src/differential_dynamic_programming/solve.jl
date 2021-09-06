function ddp_solve!(prob::ProblemData;
    max_iter = 10,
    obj_tol = 1.0e-5,
    grad_tol = 1.0e-5,
    α_min = 1.0e-5,
    linesearch = :armijo,
    verbose = true,
	cache = false)

	println()
    (verbose && prob.m_data.obj isa StageCosts) && printstyled("Differential Dynamic Programming (iLQR)\n",
		color = :red, bold = true)

	# data
	p_data = prob.p_data
	m_data = prob.m_data
	s_data = prob.s_data

	objective!(s_data, m_data, mode = :nominal)
    derivatives!(m_data, mode = :nominal)
    backward_pass!(p_data, m_data, mode = :nominal)

    stats = Dict(:iters => 0)
    obj_prev = s_data.obj

    for i = 1:max_iter
        forward_pass!(p_data, m_data, s_data,
            α_min = α_min,
            linesearch = linesearch)

        if linesearch != :none
            derivatives!(m_data, mode = :nominal)
            backward_pass!(p_data, m_data, mode = :nominal)
            lagrangian_gradient!(s_data, p_data, m_data)
        end

		# cache solver data
		cache && cache!(s_data)

        # check convergence
        stats[:iters] = i
        grad_norm = norm(s_data.gradient, Inf)
        verbose && println("     iter: $i
             cost: $(s_data.obj)
			 grad_norm: $(grad_norm)
			 c_max: $(s_data.c_max)
			 α: $(s_data.α)")
		grad_norm < grad_tol && break
        abs(s_data.obj - obj_prev) < obj_tol ? break : (obj_prev = s_data.obj)
        !s_data.status && break
    end



    return stats
end

"""
    gradient of Lagrangian
        https://web.stanford.edu/class/ee363/lectures/lqr-lagrange.pdf
"""
function lagrangian_gradient!(s_data::SolverData, p_data::PolicyData, n, m, T)
    p = p_data.p
    Qx = p_data.Qx
    Qu = p_data.Qu

    for t = 1:T-1
        s_data.gradient[s_data.idx_x[t]] = Qx[t] - p[t] # should always be zero by construction
        s_data.gradient[s_data.idx_u[t]] = Qu[t]
    end
    # NOTE: gradient wrt xT is satisfied implicitly
end

function lagrangian_gradient!(s_data::SolverData, p_data::PolicyData, m_data::ModelData)
	lagrangian_gradient!(s_data, p_data,
		m_data.n, m_data.m, m_data.T)
end

"""
    augmented Lagrangian solve
"""
function constrained_ddp_solve!(prob::ProblemData;
    linesearch = :armijo,
    max_iter = 10,
	max_al_iter = 5,
    α_min = 1.0e-5,
    obj_tol = 1.0e-5,
    grad_tol = 1.0e-5,
	con_tol = 1.0e-3,
	con_norm_type = Inf,
	ρ_init = 1.0,
	ρ_scale = 10.0,
	ρ_max = 1.0e8,
	cache = false,
    verbose = true)

	println()
	verbose && printstyled("Differential Dynamic Programming\n",
		color = :red, bold = true)

	# initial penalty
	for (t, ρ) in enumerate(prob.m_data.obj.ρ)
		prob.m_data.obj.ρ[t] = ρ_init .* ρ
	end

    stats_al = Dict(:iters => 0, :stats_ilqr => Dict{Symbol,Int}[])

	for i = 1:max_al_iter
		verbose && println("  al iter: $i")

		# primal minimization
		stats = ddp_solve!(prob,
            linesearch = linesearch,
            α_min = α_min,
		    max_iter = max_iter,
            obj_tol = obj_tol,
		    grad_tol = grad_tol,
			cache = cache,
		    verbose = verbose)

        push!(stats_al[:stats_ilqr], stats)

		# update trajectories
		objective!(prob.s_data, prob.m_data, mode = :nominal)

		# constraint violation
		# c_max = constraint_violation(prob.m_data.obj.cons,
		# 	prob.m_data.x̄, prob.m_data.ū,
		# 	norm_type = con_norm_type)
		# verbose && println("    c_max: $(prob.s_data.c_max)\n")
		prob.s_data.c_max <= con_tol && break

		# dual ascent
		augmented_lagrangian_update!(prob.m_data.obj,
			s = ρ_scale, max_penalty = ρ_max)
	end

    return stats_al
end

function ilqr_iterations(stats)
    sum([s[:iters] for s in stats[:stats_ilqr]])
end
