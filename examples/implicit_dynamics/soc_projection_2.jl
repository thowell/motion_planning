# control projection problem
mf = 3
nz = mf + 1 + 1 + 1 + 1 + mf
nθ = mf + 1

idx = [3; 1; 2]
function residual(z, θ, κ)
  u = z[1:mf]
  s = z[mf .+ (1:1)]
  y = z[mf + 1 .+ (1:1)]
  w = z[mf + 1 + 1 .+ (1:1)]
  p = z[mf + 1 + 1 + 1 .+ (1:1)]
  v = z[mf + 1 + 1 + 1 + 1 .+ (1:mf)]

  ū = θ[1:mf]
  T = θ[mf .+ (1:1)]

  [
   u - ū - v - [0.0; 0.0; y[1] + p[1]];
   -y[1] - w[1];
   T[1] - u[3] - s[1];
   w .* s .- κ
   p .* u[3] .- κ
   second_order_cone_product(v[idx], u[idx]) .- κ .* [1.0; 0.0; 0.0]
  ]
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

idx_ineq = collect([3, 4, 6, 7])
idx_soc = [collect([3, 1, 2]), collect([10, 8, 9])]#collect([3, 1, 2]), collect([10, 8, 9])]

# ul = -1.0 * ones(m)
# uu = 1.0 * ones(m)
ū = [0.0, 0.0, 0.0]

z0 = 0.1 * ones(nz)
z0[1:mf] = copy(ū)
z0[3] += 1.0
z0[10] += 1.0

θ0 = [copy(ū); 1.0]

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
	κ_tol = 1.0e-2,
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

interior_point_solve!(ip_con)
interior_point_solve!(ip_jac)

function soc_projection(x, x_min, x_max, scaling)
  proj = second_order_cone_projection(x[[3;1;2]])[1]
	ip_con.z .= [proj[2:3]; proj[1]; 0.1 * ones(7)]
  ip_con.z[3] += max(1.0, norm(proj[2:3])) * 2.0
  ip_con.z[10] += 1.0
	ip_con.θ .= [x; scaling * x_max]

	status = interior_point_solve!(ip_con)

  !status && (@warn "projection failure (res norm: $(norm(ip_con.r, Inf))) \n
		               z = $(ip_con.z), \n
					   θ = $(ip_con.θ)")

	return ip_con.z[1:mf]
end

soc_projection([100.0, 0.0, -1.0], 0.0, 10.0, 0.5)

function soc_projection_jacobian(x, x_min, x_max, scaling)
  proj = second_order_cone_projection(x[[3;1;2]])[1]
	ip_con.z .= [proj[2:3]; proj[1]; 0.1 * ones(7)]
  ip_con.z[3] += max(1.0, norm(proj[2:3])) * 2.0
  ip_con.z[10] += 1.0
	ip_con.θ .= [x; scaling * x_max]

	interior_point_solve!(ip_jac)

	return ip_jac.δz[1:mf, 1:mf]
end

soc_projection_jacobian([10.0, 0.0, 1.0], 0.0, 10.0, 0.5)
