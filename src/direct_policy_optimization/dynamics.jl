struct SampleDynamics <: Constraints
	n
	ineq
	dim_nom
	dim_sample
	w0
end

function sample_dynamics_constraints(model_sample, T)
	n =
	ineq = (1:0)
	w0 = zeros(model.n)

	return SampleDynamics(n, ineq, w0)
end

function constraints!(c, τ_mean, τ_sample, S, con::SampleDynamics, model,
	idx_τ, idx_s, h, T, N, M, w, β)

	n = model.n

	con_shift = 0

	# sample mean
	for t = 1:T

		# mean
		μ = view(τ_mean, idx_τ.x[t])

		# samples
		x = [view(τ_sample[i], idx_τ[i].x[t]) for i = 1:N]

		# sample mean
		c[con_shift .+ (1:n)] = μ - sample_mean(x)
		con_shift += n
	end

	# propagate samples sigma points (xj, uj, wj)
	for t = 1:T-1

		# mean
		μ = view(τ_mean, idx_τ.x[t])
		ν = view(τ_mean, idx_τ.u[t])

		# samples
		x = [view(τ_sample[i], idx_τ[i].x[t]) for i = 1:N]
		u = [view(τ_sample[i]), idx_τ[i].u[t] for i = 1:N]

		# slacks
		s⁺ = [view(S[j], idx_s[j].x[t]) for j = 1:M]

		# propagate
		for j = 1:M
			if j <= N
				c[con_shift .+ (1:n)] = fd(model, s⁺[j], x[j], u[j],
					con.w0, h, t)

				con_shift += n
			else
				k = j - N
				c[con_shift .+ (1:n)] = fd(model, s⁺[j], μ, ν,
					β * w[k],
					h, t)

				con_shift += n
			end
		end
	end

	# resample
	for t = 1:T-1
		# samples
		x⁺ = [view(τ_sample[i], idx_τ[i].x[t + 1]) for i = 1:N]

		# slacks
		s⁺ = [view(S[j], idx_s[j].x[t]) for j = 1:M]

		xs⁺ = resample(s⁺, β)

		for i = 1:N
			c[con_shift .+ (1:n)] = x⁺[i] - xs⁺[i]
			con_shift += n
		end
	end


    return nothing
end

function constraints_jacobian!(∇c, τ_mean, τ_sample, S, con::SampleDynamics, model,
	idx_τ, idx_s, h, T, N, M, w, β, shift_sample = 0)

	n = model.n

	con_shift = 0
	jac_shift = 0

	# sample mean
	for t = 1:T
		# samples
		x_vec = vcat([view(τ_sample[i], idx_τ[i].x[t]) for i = 1:N]...)
		sample_mean_vec(y) = sample_mean([view(y, (i - 1) * n .+ (1:n)) for i = 1:N])

		# sample mean
		c[con_shift .+ (1:n)] = μ - sample_mean(x)
		con_shift += n

		r_idx = con_shift .+ (1:n)

		c_idx = idx_τ.x[t]
		len = length(r_idx, c_idx)
		∇c[jac_shift .+ (1:len)] = vec(Diagonal(ones(n)))
		jac_shift += len

		c_idx = vcat([idx_τ[i].x[t] for i = 1:N]...)
		len = length(r_idx, c_idx)
		∇c[jac_shift .+ (1:len)] = vec(ForwardDiff.jacobian(sample_mean_vec, x_vec))
		jac_shift += len
	end

	# propagate samples sigma points (xj, uj, wj)
	for t = 1:T-1

		# mean
		μ = view(τ_mean, idx_τ.x[t])
		ν = view(τ_mean, idx_τ.u[t])

		# samples
		x = [view(τ_sample[i], idx_τ[i].x[t]) for i = 1:N]
		u = [view(τ_sample[i]), idx_τ[i].u[t] for i = 1:N]

		# slacks
		s⁺ = [view(S[j], idx_s[j].x[t]) for j = 1:M]

		# propagate
		for j = 1:M
			if j <= N
				c[con_shift .+ (1:n)] = fd(model, s⁺[j], x[j], u[j],
					con.w0, h, t)

				fds(y) = fd(model, y, x[j], u[j], con.w0, h, t)

				con_shift += n
			else
				k = j - N
				c[con_shift .+ (1:n)] = fd(model, s⁺[j], μ, ν,
					β * w[k],
					h, t)

				con_shift += n
			end
		end
	end

	# resample
	for t = 1:T-1
		# samples
		x⁺ = [view(τ_sample[i], idx_τ[i].x[t + 1]) for i = 1:N]

		# slacks
		s⁺ = [view(S[j], idx_s[j].x[t]) for j = 1:M]

		xs⁺ = resample(s⁺, β)

		for i = 1:N
			c[con_shift .+ (1:n)] = x⁺[i] - xs⁺[i]
			con_shift += n
		end
	end

	return nothing
end

function constraints_sparsity(con::SampleDynamics, model_sample,
		idx_sample T; r_shift = 0)

	row = []
    col = []


	for t = 1:T-1

	end

    return collect(zip(row, col))
end
