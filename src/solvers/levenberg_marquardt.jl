"""
	Levenberg-Marquardt
		(damped least squares)
"""
function levenberg_marquardt(res::Function, x;
		reg = 1.0e-8, tol_r = 1.0e-8, tol_d = 1.0e-6)

	y = copy(x)

	merit(z) = res(z)' * res(z)

	α = 1.0
	iter = 0

	while iter < 100
		me = merit(y)
		r = res(y)
		∇r = ForwardDiff.jacobian(res, y)

		_H = ∇r' * ∇r
		Is = Diagonal(diag(_H))
		H = (_H + reg * Is)

		pd_iter = 0
		while !isposdef(Hermitian(Array(H)))
			reg *= 2.0
			H = (_H + reg * Is)
			pd_iter += 1

			if pd_iter > 100 || reg > 1.0e12
				@error "regularization failure"
			end
		end

		Δy = -1.0 * H \ (∇r' * r)

		ls_iter = 0
		while merit(y + α * Δy) > me + 1.0e-4 * r' * (α * Δy)
			α *= 0.5
			reg = reg
			ls_iter += 1

			if ls_iter > 100 || reg > 1.0e12
				@error "line search failure"
			end
		end

		y .+= α * Δy
		α = min(1.2 * α, 1.0)
		reg = 0.5 * reg

		iter += 1

		norm(α * Δy, Inf) < tol_d && (return y)
		norm(r, Inf) < tol_r && (return y)
	end

	@warn "Gauss Newton failure"
	return y
end
