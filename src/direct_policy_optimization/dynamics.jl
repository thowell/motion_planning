struct SampleDynamics <: Constraints
	n
	ineq
	w0
end

function sample_dynamics_constraints(prob, N, M)
	T = prob.nom.T

	# sample mean
	n = prob.mean.model.n * T

	# propagate samples
	for j = 1:M
		if j <= N
			n += prob.sample[j].model.n * (T - 1)
		else
			n += prob.mean.model.n * (T - 1)
		end
	end

	# resample
	for i = 1:N
		n += prob.sample[i].model.n * (T - 1)
	end

	ineq = (1:0)
	w0 = [zeros(p.model.n) for p in prob.sample]

	return SampleDynamics(n, ineq, w0)
end

function constraints!(c, Z, con::SampleDynamics,
	prob::DPOProblems, idx::DPOIndices,
	N, D,
	w, β)

	T = prob.nom.T
	h = prob.nom.h

	con_shift = 0

	# sample mean
	for t = 1:T

		# mean
		μ = view(Z, idx.mean[prob.mean.idx.x[t]])

		# samples
		x = [view(Z, idx.sample[i][prob.sample[i].idx.x[t]]) for i = 1:N]

		# sample mean
		c[con_shift .+ (1:prob.mean.model.n)] = μ - sample_mean(x)
		con_shift += prob.mean.model.n
	end

	# propagate samples sigma points (xj, uj, wj)
	for t = 1:T-1

		# mean
		μ = view(Z, idx.mean[prob.mean.idx.x[t]])
		ν = view(Z, idx.mean[prob.mean.idx.u[t]])

		# samples
		x = [view(Z, idx.sample[i][prob.sample[i].idx.x[t]]) for i = 1:N]
		u = [view(Z, idx.sample[i][prob.sample[i].idx.u[t]]) for i = 1:N]

		# state slacks
		s⁺ = [view(Z, idx.slack[j][idx.s[t]]) for j = 1:M]

		# propagate
		for j = 1:M
			if j <= N
				c[con_shift .+ (1:prob.sample[i].model.n)] = fd(prob.sample[i].model,
					s⁺[j], x[j], u[j],
					con.w0[j], h, t)

				con_shift += prob.sample[i].model.n
			else
				k = j - N
				c[con_shift .+ (1:prob.mean.model.n)] = fd(prob.mean.model,
					s⁺[j], μ, ν,
					β * w[t][k],
					h, t)

				con_shift += prob.mean.model.n
			end
		end
	end

	# resample
	for t = 1:T-1
		# samples
		x⁺ = [view(Z, idx.sample[i][prob.sample[i].idx.x[t + 1]]) for i = 1:N]

		# slacks
		s⁺ = [view(Z, idx.slack[j][idx.s[t]]) for j = 1:M]

		xs⁺ = resample(s⁺, β)

		for i = 1:N
			c[con_shift .+ (1:prob.sample[i].model.n)] = x⁺[i] - xs⁺[i]
			con_shift += prob.sample[i].model.n
		end
	end

    return nothing
end

function constraints_jacobian!(∇c, Z, con::SampleDynamics,
	prob::DPOProblems, idx::DPOIndices,
	N, D,
	w, β)

	T = prob.nom.T
	h = prob.nom.h
	M = N + D

	con_shift = 0
	jac_shift = 0

	# sample mean
	for t = 1:T
		# samples
		x_vec = vcat([view(Z, idx.sample[i][prob.mean.idx.x[t]]) for i = 1:N]...)
		sample_mean_vec(y) = sample_mean([view(y, (i - 1) * prob.mean.model.n .+ (1:prob.mean.model.n)) for i = 1:N])

		r_idx = con_shift .+ (1:prob.mean.model.n)

		c_idx = idx.mean[prob.mean.idx.x[t]]
		len = length(r_idx) * length(c_idx)
		∇c[jac_shift .+ (1:len)] = vec(Diagonal(ones(prob.mean.model.n)))
		jac_shift += len

		c_idx = vcat([idx.sample[i][prob.sample[i].idx.x[t]] for i = 1:N]...)
		len = length(r_idx) * length(c_idx)
		∇c[jac_shift .+ (1:len)] = vec(-1.0 * ForwardDiff.jacobian(sample_mean_vec, x_vec))
		jac_shift += len

		con_shift += prob.mean.model.n
	end

	# propagate samples sigma points (xj, uj, wj)
	for t = 1:T-1

		# mean
		μ = view(Z, idx.mean[prob.mean.idx.x[t]])
		ν = view(Z, idx.mean[prob.mean.idx.u[t]])

		# samples
		x = [view(Z, idx.sample[i][prob.sample[i].idx.x[t]]) for i = 1:N]
		u = [view(Z, idx.sample[i][prob.sample[i].idx.u[t]]) for i = 1:N]

		# state slacks
		s⁺ = [view(Z, idx.slack[j][idx.s[t]]) for j = 1:M]

		# propagate
		for j = 1:M
			if j <= N
				# c[con_shift .+ (1:prob.sample[j].model.n)] = fd(prob.sample[j].model,
				# 	s⁺[j], x[j], u[j],
				# 	con.w[j], h, t)

				fds(y) = fd(prob.sample[j].model, y, x[j], u[j], con.w0[j], h, t)
				fdx(y) = fd(prob.sample[j].model, s⁺[j], y, u[j], con.w0[j], h, t)
				fdu(y) = fd(prob.sample[j].model, s⁺[j], x[j], y, con.w0[j], h, t)

				r_idx = con_shift .+ (1:prob.sample[j].model.n)

				c_idx = idx.slack[j][idx.s[t]]
				len = length(r_idx) * length(c_idx)
				∇c[jac_shift .+ (1:len)] = vec(ForwardDiff.jacobian(fds, s⁺[j]))
				jac_shift += len

				c_idx = idx.sample[j][prob.sample[j].idx.x[t]]
				len = length(r_idx) * length(c_idx)
				∇c[jac_shift .+ (1:len)] = vec(ForwardDiff.jacobian(fdx, x[j]))
				jac_shift += len

				c_idx = idx.sample[j][prob.sample[j].idx.u[t]]
				len = length(r_idx) * length(c_idx)
				∇c[jac_shift .+ (1:len)] = vec(ForwardDiff.jacobian(fdu, u[j]))
				jac_shift += len

				con_shift += prob.sample[j].model.n
			else
				k = j - N
				c[con_shift .+ (1:prob.mean.model.n)] = fd(prob.mean.model,
					s⁺[j], μ, ν,
					β * w[t][k],
					h, t)

				_fds(y) = fd(prob.mean.model, y, μ, ν, β * w[k], h, t)
				_fdx(y) = fd(prob.mean.model, s⁺[j], y, ν, β * w[k], h, t)
				_fdu(y) = fd(prob.mean.model, s⁺[j], μ, y, β * w[k], h, t)

				r_idx = con_shift .+ (1:prob.mean.model.n)

				c_idx = idx.slack[j][idx.s[t]]
				len = length(r_idx) * length(c_idx)
				∇c[jac_shift .+ (1:len)] = vec(ForwardDiff.jacobian(_fds, s⁺[j]))
				jac_shift += len

				c_idx = idx.sample[j][prob.mean.idx.x[t]]
				len = length(r_idx) * length(c_idx)
				∇c[jac_shift .+ (1:len)] = vec(ForwardDiff.jacobian(_fdx, μ))
				jac_shift += len

				c_idx = idx.sample[j][prob.mean.idx.u[t]]
				len = length(r_idx) * length(c_idx)
				∇c[jac_shift .+ (1:len)] = vec(ForwardDiff.jacobian(_fdu, ν))
				jac_shift += len

				con_shift += prob.mean.model.n
			end
		end
	end

	# resample
	for t = 1:T-1
		# samples
		x⁺ = [view(Z, idx.sample[i][prob.sample[i].idx.x[t + 1]]) for i = 1:N]

		# slacks
		s⁺_vec = vcat([view(Z, idx.slack[j][idx.s[t]]) for j = 1:M]...)
		s_idx_vec = vcat([idx.slack[j][idx.s[t]] for j = 1:M]...)

		for i = 1:N
			# c[con_shift .+ (1:prob.sample[i].model.n)] = x⁺[i] - xs⁺[i]

			r_idx = con_shift .+ (1:prob.sample[i].model.n)

			c_idx = idx.sample[i][prob.sample[i].idx.x[t + 1]]
			len = length(r_idx) * length(c_idx)
			∇c[jac_shift .+ (1:len)] = vec(Diagonal(ones(prob.sample[i].model.n)))
			jac_shift += len

			c_idx = s_idx_vec
			len = length(r_idx) * length(c_idx)
			r_vec(q) = resample_vec(q, prob.mean.model.n, M, i, β)
			∇c[jac_shift .+ (1:len)] = vec(real.(-1.0 * FiniteDiff.finite_difference_jacobian(r_vec, s⁺_vec)))
			jac_shift += len

			con_shift += prob.sample[i].model.n
		end
	end

	return nothing
end

function constraints_sparsity(con::SampleDynamics,
	prob::DPOProblems, idx::DPOIndices,
	N, D;
	shift_row = 0, shift_col = 0)

	row = []
    col = []

	T = prob.nom.T
	h = prob.nom.h
	M = N + D

	con_shift = 0

	# sample mean
	for t = 1:T
		# samples
		r_idx = shift_row + con_shift .+ (1:prob.mean.model.n)

		c_idx = shift_col .+ idx.mean[prob.mean.idx.x[t]]
		row_col!(row, col, r_idx, c_idx)

		c_idx = shift_col .+ vcat([idx.sample[i][prob.sample[i].idx.x[t]] for i = 1:N]...)
		row_col!(row, col, r_idx, c_idx)

		con_shift += prob.mean.model.n
	end

	# propagate samples sigma points (xj, uj, wj)
	for t = 1:T-1
		# propagate
		for j = 1:M
			if j <= N
				r_idx = shift_row + con_shift .+ (1:prob.sample[j].model.n)

				c_idx = shift_col .+ idx.slack[j][idx.s[t]]
				row_col!(row, col, r_idx, c_idx)

				c_idx = shift_col .+ idx.sample[j][prob.sample[j].idx.x[t]]
				row_col!(row, col, r_idx, c_idx)

				c_idx = shift_col .+ idx.sample[j][prob.sample[j].idx.u[t]]
				row_col!(row, col, r_idx, c_idx)

				con_shift += prob.sample[j].model.n
			else
				k = j - N

				r_idx = shift_row + con_shift .+ (1:prob.mean.model.n)

				c_idx = shift_col .+ idx.slack[j][idx.s[t]]
				row_col!(row, col, r_idx, c_idx)

				c_idx = shift_col .+ idx.sample[j][prob.mean.idx.x[t]]
				row_col!(row, col, r_idx, c_idx)

				c_idx = shift_col .+ idx.sample[j][prob.mean.idx.u[t]]
				row_col!(row, col, r_idx, c_idx)

				con_shift += prob.mean.model.n
			end
		end
	end

	# resample
	for t = 1:T-1
		for i = 1:N
			r_idx = shift_row + con_shift .+ (1:prob.sample[i].model.n)

			c_idx = shift_col .+ idx.sample[i][prob.sample[i].idx.x[t + 1]]
			row_col!(row, col, r_idx, c_idx)

			s_idx_vec = vcat([idx.slack[j][idx.s[t]] for j = 1:M]...)
			c_idx = shift_col .+ s_idx_vec
			row_col!(row, col, r_idx, c_idx)

			con_shift += prob.sample[i].model.n
		end
	end

    return collect(zip(row, col))
end
