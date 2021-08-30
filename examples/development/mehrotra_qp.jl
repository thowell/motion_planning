using LinearAlgebra, ForwardDiff, Random

# test Mehrotra predictor-corrector algorithm
m = 15
n = 10
p = 5
x0 = Random.randn(n)
Q = Random.randn(n, n)
Q = Q' * Q
c = Random.randn(n)
G = Random.randn(m, n)
h = G * x0 + ones(m)
A = Random.randn(p, n)
b = A * x0

nθ = 0
p0 = zeros(nθ)
a0 = zeros(m)

# unpack
function unpack(w)
    x = view(w, 1:n)
    s = view(w, n .+ (1:m))
    yx = view(w, n + m .+ (1:p))
    ys = view(w, n + m + p .+ (1:m))
    z = view(w, n + m + p + m .+ (1:m))

    return x, s, yx, ys, z
end

function obj(w)
    x, s, yx, ys, z = unpack(w)
    0.5 * x' * Q * x + c' * x
end

function con(w)
    x, s, yx, ys, z = unpack(w)
    A * x - b
end

# barrier check
function check_variables(w)
    x, s, yx, ys, z = unpack(w)
    if any(s .<= 0.0)
        return true
    end
    if any(z .<= 0.0)
        return true
    end
    return false
end

function r(w, θ, a)
    x, s, yx, ys, z = unpack(w)

    lag_x = Q * x + c + transpose(A) * yx + transpose(G) * ys
    lag_s = ys - z
    c_eq = A * x - b
    c_ineq = s - (h - G * x)

    comp = s .* z
    comp -= a

    return [lag_x; lag_s; c_eq; c_ineq; comp]
end

function Rz(w, θ)
    return ForwardDiff.jacobian(q -> r(q, θ, a0), w)
end

function Rθ(w, θ)
    return ForwardDiff.jacobian(q -> r(w, q, a0), θ)
end

function interior_point(x0, θ;
    x_init_tol = 0.1,
    r_tol = 1.0e-8,
    μ_tol = 1.0e-6,
    max_iter_inner = 100,
    max_iter_ls = 25)

    # initialization
    x = max.(x0, x_init_tol)
    s = ones(m)
    yx = zeros(p)
    ys = zeros(m)
    z = ones(m)


    w = [x; s; yx; ys; z]
    μ = 1.0
    e = ones(m)

    flag = false
    total_iter = 0

    while μ > μ_tol
        for i = 1:max_iter_inner

            # compute residual, residual Jacobian
            res = r(w, θ, μ * e)
            res_norm = norm(res)

            # println("iteration: $i, residual: $res_norm")

            if res_norm < r_tol
                flag = true
                continue
            end

            jac = Rz(w, θ)

            # compute step
            Δ = -jac \ res

            # line search the step direction
            α = 1.0

            iter = 0
            while check_variables(w + α * Δ) # backtrack inequalities
                α = 0.5 * α
                # println("   α = $α")
                iter += 1
                if iter > max_iter_ls
                    @error "backtracking line search fail"
                    flag = false
                    return w, zeros(n + me + mi, p), false, total_iter
                end
            end

            while norm(r(w + α * Δ, θ, μ * e))^2.0 >= (1.0 - 0.001 * α) * res_norm^2.0
                α = 0.5 * α
                # println("   α = $α")
                iter += 1
                if iter > max_iter_ls
                    @error "line search fail"
                    flag = false
                    return w, zeros(n + m + p + m + m, ), false, total_iter
                end
            end

            # update
            w .+= α * Δ

            total_iter += 1
        end

        μ = 0.1 * μ
        println("μ: $μ")
    end

    δw = -1.0 * Rz(w, θ) \ Rθ(w, θ)

    return w, δw, flag, total_iter
end

w_soli, δw_soli, flag, total_iter = interior_point(rand(n), p0)
@show total_iter

function mehrotra(x0, θ;
    x_init_tol = 0.1,
    r_tol = 1.0e-8,
    μ_tol = 1.0e-6,
    max_iter_inner = 100,
    max_iter_ls = 25)

    # initialization
    x = max.(x0, x_init_tol)
    s = ones(m)
    yx = zeros(p)
    ys = zeros(m)
    z = ones(m)
    w = [x; s; yx; ys; z]
    μ = 1.0
    e = ones(m)

    flag = false
    total_iter = 0

    while μ > μ_tol
        for i = 1:max_iter_inner

            # unpack
            x, s, yx, ys, z = unpack(w)

            # compute residual, residual Jacobian
            res_aff = r(w, θ, a0)
            res_aff_norm = norm(res_aff, 1)

            # println("iteration: $i, residual: $res_norm")

            if res_aff_norm < r_tol
                flag = true
                continue
            end

            jac = Rz(w, θ)

            # compute affine step
            Δ_aff = -jac \ res_aff

            Δx_aff, Δs_aff, Δyx_aff, Δys_aff, Δz_aff = unpack(Δ_aff)

            α_aff_pr = min(1.0, minimum([Δsi < 0.0 ? -s[i] / Δsi : Inf for (i, Δsi) in enumerate(Δs_aff)]))
            α_aff_du = min(1.0, minimum([Δzi < 0.0 ? -z[i] / Δzi : Inf for (i, Δzi) in enumerate(Δz_aff)]))
            μ_aff = transpose(s + α_aff_pr * Δs_aff) * (z + α_aff_du * Δz_aff) / m

            # compute corrector step
            σ = (μ_aff / μ)^3.0

            res_corr = r(w, θ, -Δs_aff .* Δz_aff + σ * μ * e)

            Δ_corr = -jac \ res_corr

            # line search the step direction
            η = 1.0
            α_pr = α_aff_pr
            α_du = α_aff_du

            α = η .* [α_pr * ones(n + m + p + m); α_du * ones(m)]

            iter = 0
            while check_variables(w + α .* Δ_corr) # backtrack inequalities
                η = 0.99 * η
                α = η .* [α_pr * ones(n + m + p + m); α_du * ones(m)]

                # println("   α = $α")
                iter += 1
                if iter > max_iter_ls
                    @error "backtracking line search fail"
                    flag = false
                    return w, zeros(n + m + p + m + m, nθ), false, total_iter
                end
            end

            while norm(r(w + α .* Δ_corr, θ, a0))^2.0 >= (1.0 - 0.001 * η) * res_aff_norm^2.0
                η = 0.99 * η
                α = η .* [α_pr * ones(n + m + p + m); α_du * ones(m)]
                # println("   α = $α")
                iter += 1
                if iter > max_iter_ls
                    @error "line search fail"
                    flag = false
                    return w, zeros(n + m + p + m + m, nθ), false, total_iter
                end
            end

            # update
            w .+= α .* Δ_corr

            total_iter += 1
        end

        μ = 0.1 * μ
        println("μ: $μ")
    end

    δw = -1.0 * Rz(w, θ) \ Rθ(w, θ)

    return w, δw, flag, total_iter
end

w_solm, δw_solm, flag, total_iter = mehrotra(rand(n), p0)
@show total_iter

obj(w_soli)
obj(w_solm)

norm(con(w_soli), Inf)
norm(con(w_solm), Inf)
