"""
	Newton
"""
function newton(res::Function, x;
		tol_r = 1.0e-8, tol_d = 1.0e-6, reg = 1.0e-8)
	y = copy(x)
	Δy = copy(x)

    r = res(y)

    iter = 0

    while norm(r, 2) > tol_r && iter < 25
        ∇r = ForwardDiff.jacobian(res, y)

		try
        	Δy = -1.0 * (∇r + reg * I) \ r
		catch
			@warn "implicit-function failure"
			return y
		end

        α = 1.0

		iter_ls = 0
        while α > 1.0e-8 && iter_ls < 10
            ŷ = y + α * Δy
            r̂ = res(ŷ)

            if norm(r̂) < norm(r)
                y = ŷ
                r = r̂
                break
            else
                α *= 0.5
				iter_ls += 1
            end

			iter_ls == 10 && (@warn "line search failed")
        end

        iter += 1
    end

	iter == 10 && (@warn "Newton failure")

    return y
end
