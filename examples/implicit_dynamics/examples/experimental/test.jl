z = [-13.641864412648351, 3.6504483668273315, 14.121835501518698, 0.013348281926179581, -2.173224411310659, 2.236391646347381, 0.045013523891151146, 6.290967771428232, -1.8697297768312837, 6.958860123769445]
θ = [-29.08851105461329, 0.0003060763661500973, 6.066121573001358, 12.5]

ip_con.z .= copy(z)
ip_con.z[idx_ineq] .+ 1.0
ip_con.opts.κ_init = 1.0
second_order_cone_projection([ip_con.z[idx] for idx in idx_soc][2])
ip_con.θ .= copy(θ)


interior_point_solve!(ip_con)

cone_check(z, idx_ineq, idx_soc)


data.ip_dyn.z .= copy(z)
# data.ip_dyn.z[1:2] .+= 1.0 * randn(2)

data.ip_dyn.θ .= copy(θ)

data.ip_dyn.opts.κ_init = 0.01
interior_point_solve!(data.ip_dyn)

data.ip_dyn.z̄
data.ip_dyn.κ

cone_check(data.ip_dyn.z̄, data.ip_dyn.idx_ineq, data.ip_dyn.idx_soc)

model.dim.q

data.ip_dyn.opts.reg_pr_init = 0.0

data.ip_dyn.z

visualize_elbow!(vis, model, [z, data.ip_dyn.z[1:2]], Δt = h)
