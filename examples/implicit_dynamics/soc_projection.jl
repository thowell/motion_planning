nz = 3 + 2 + 1 + 1 + 3 + 3 + 3 + 2
nθ = 3 + 1 + 1 + 1

idx = [3; 1; 2]
function residual(z, θ, κ)
    m = 3

    f = z[1:m]
    s = z[m .+ (1:2)]
    y1 = z[m + 2 .+ (1:1)]
    y2 = z[m + 2 + 1 .+ (1:1)]
    y3 = z[m + 2 + 1 + 1 .+ (1:3)]
    β = z[m + 2 + 1 + 1 + 3 .+ (1:3)]
    η = z[m + 2 + 1 + 1 + 3 + 3 .+ (1:3)]
    z = z[m + 2 + 1 + 1 + 3 + 3 + 3 .+ (1:2)]

    f̄ = θ[1:m]
    f_min = θ[m .+ (1:1)]
    f_max = θ[m + 1 .+ (1:1)]
    μ = θ[m + 1 + 1 .+ (1:1)]

    A = [0.0 0.0 μ; 1.0 0.0 0.0; 0.0 1.0 0.0]

    return [f - f̄ + [0.0; 0.0; -y1[1] + y2[1]] + transpose(A) * y3;
            [-y1[1]; -y2[1]] - z;
            f_max[1] - f[3] - s[1];
            f[3] - f_min[1] - s[2];
            A * f - β;
            -y3 - η;
            second_order_cone_product(β, η) - [κ; 0.0; 0.0];
            s .* z .- κ]
end

@variables z[1:nz], θ[1:nθ], κ[1:1]

r = residual(z, θ, κ)
r .= simplify(r)
r_func = eval(Symbolics.build_function(r, z, θ, κ)[2])

rz = Symbolics.jacobian(r, z)
rz = simplify.(rz)
rz_func = eval(Symbolics.build_function(rz, z, θ)[2])

rθ = Symbolics.jacobian(r, θ)
rθ = simplify.(rθ)
rθ_func = eval(Symbolics.build_function(rθ, z, θ)[2])

idx_ineq = collect([4, 5, 17, 18])
idx_soc = [collect([11, 12, 13]), collect([14, 15, 16])]

# ul = -1.0 * ones(m)
# uu = 1.0 * ones(m)
ū = [0.0; 0.0; 0.0]
u_min = 0.0
u_max = 1.0
scaling = 1.0
z0 = [ū; 1.0; 1.0; 0.0; 0.0; 0.0; 0.0; 0.0; 1.0; 0.1; 0.1; 1.0; 0.1; 0.1; 1.0; 1.0]

θ0 = [copy(ū); u_min; u_max; scaling]

# solver
opts_con = InteriorPointOptions(
    κ_init = 1.0,
    κ_tol = 1.0e-4,
    r_tol = 1.0e-8,
    diff_sol = false)

ip_con = interior_point(z0, θ0,
    r! = r_func, rz! = rz_func,
    rz = similar(rz, Float64),
    rθ! = rθ_func,
    rθ = similar(rθ, Float64),
    idx_ineq = idx_ineq,
    idx_soc = idx_soc,
    opts = opts_con)

opts_jac = InteriorPointOptions(
	κ_init = 1.0,
	κ_tol = 1.0e-3,
	r_tol = 1.0e-8,
	diff_sol = true)

ip_jac = interior_point(z0, θ0,
	r! = r_func, rz! = rz_func,
	rz = similar(rz, Float64),
	rθ! = rθ_func,
	rθ = similar(rθ, Float64),
	idx_ineq = idx_ineq,
  idx_soc = idx_soc,
	opts = opts_jac)

# interior_point_solve!(ip_con)
# interior_point_solve!(ip_jac)

function soc_projection(x, x_min, x_max, scaling)
    ip_con.z .= [x; 1.0; 1.0; 0.0; 0.0; 0.0; 0.0; 0.0; 1.0; 0.1; 0.1; 1.0; 0.1; 0.1; 1.0; 1.0]
    ip_con.θ .= [x; x_min; x_max; scaling]

    status = interior_point_solve!(ip_con)

    !status && (@warn "projection failure (res norm: $(norm(ip_con.r, Inf))) \n
    	               z = $(ip_con.z), \n
    				   θ = $(ip_con.θ)")

	return @view ip_con.z[1:3]
end

# soc_projection([100.0, 0.0, -1.0], 0.0, 1.0, 1.0)

function soc_projection_jacobian(x, x_min, x_max, scaling)
    ip_con.z .= [x; 1.0; 1.0; 0.0; 0.0; 0.0; 0.0; 0.0; 1.0; 0.1; 0.1; 1.0; 0.1; 0.1; 1.0; 1.0]
    ip_con.θ .= [x; x_min; x_max; scaling]

	interior_point_solve!(ip_jac)

	return view(ip_jac.δz, 1:3, 1:3)
end

# soc_projection_jacobian([0.0, 0.0, 1.0], 0.0, 10.0, 1.0)
