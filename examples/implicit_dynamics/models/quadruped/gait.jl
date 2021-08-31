using Plots

# Horizon
T = 101
Tm = 51

# Time step
tf = 1.0
h = tf / (T - 1)

function ellipse_traj(x_start, x_goal, z, T)
	dist = x_goal - x_start
	a = 0.5 * dist
	b = z
	z̄ = 0.0
	# x = range(x_start, stop = x_goal, length = T)
	x = circular_projection_range(x_start, stop = x_goal, length = T)
	z = sqrt.(max.(0.0, (b^2) * (1.0 .- ((x .- (x_start + a)).^2.0) / (a^2.0))))
	return x, z
end

function circular_projection_range(start; stop=1.0, length=10)
	dist = stop - start
	θr = range(π, stop=0, length=length)
	r = start .+ dist * ((1 .+ cos.(θr))./2)
	return r
end


function initial_configuration(model;
    l_torso = 0.19,
    w_torso = 0.1025,
    offset = 0.025)
	q_body = zeros(6)

	# position
	q_body[3] = body_height

	# orientation
	mrp = MRP(RotZ(0.0))
	q_body[4:6] = [mrp.x; mrp.y; mrp.z]

	# feet positions (in body frame)
	q_f1 = [l_torso + offset; w_torso; 0.0]
	q_f2 = [l_torso - offset; -w_torso; 0.0]
	q_f3 = [-l_torso - offset; w_torso; 0.0]
	q_f4 = [-l_torso + offset; -w_torso; 0.0]

	return q_body, q_f1, q_f2, q_f3, q_f4
end


q_body, q_f1, q_f2, q_f3, q_f4 = initial_configuration(model)
visualize!(vis, model, [q_body], [q_f1], [q_f2], [q_f3], [q_f4])

# feet positions
pf1 = copy(q_f1)
pf2 = copy(q_f2)

pf3 = copy(q_f3)
pf4 = copy(q_f4)

strd = 2 * (pf1 - pf2)[1]

# q_shift1 = zeros(model.nq)
# q_shift1[1] = 0.5 * strd
# q_shift1[10] = strd
# q_shift1[13] = strd

body_shift = [0.5 * strd; 0.0; 0.0; 0.0; 0.0; 0.0]
foot_shift = [1.0 * strd; 0.0; 0.0]

qM_body = q_body + body_shift
qM_f1 = copy(q_f1)
qM_f2 = q_f2 + foot_shift
qM_f3 = q_f3 + foot_shift
qM_f4 = copy(q_f4)

qT_body = qM_body + body_shift
qT_f1 = copy(qM_f1) + foot_shift
qT_f2 = copy(qM_f2)
qT_f3 = copy(qM_f3)
qT_f4 = copy(qM_f4) + foot_shift

q_body_ref = linear_interpolation(q_body, qT_body, T)#[q_body, linear_interpolation(q_body, qM_body, Tm - 1)..., linear_interpolation(qM_body, qT_body, Tm - 1)...]
q_f1_ref = [q_f1, linear_interpolation(q_f1, qM_f1, Tm - 1)..., linear_interpolation(qM_f1, qT_f1, Tm - 1)...]
q_f2_ref = [q_f2, linear_interpolation(q_f2, qM_f2, Tm - 1)..., linear_interpolation(qM_f2, qT_f2, Tm - 1)...]
q_f3_ref = [q_f3, linear_interpolation(q_f3, qM_f3, Tm - 1)..., linear_interpolation(qM_f3, qT_f3, Tm - 1)...]
q_f4_ref = [q_f4, linear_interpolation(q_f4, qM_f4, Tm - 1)..., linear_interpolation(qM_f4, qT_f4, Tm - 1)...]

v_body_ref = (qT_body - q_body) / tf

visualize!(vis, model, q_body_ref, q_f1_ref, q_f2_ref, q_f3_ref, q_f4_ref,
    Δt = h)

T_fix = 5
zh = 0.05
xf1_el, zf1_el = ellipse_traj(pf1[1], pf1[1] + strd, zh, Tm - T_fix)
xf1 = [[pf1[1] for t = 1:Tm + T_fix]..., xf1_el[2:end]...]
zf1 = [[pf1[3] for t = 1:Tm + T_fix]..., zf1_el[2:end]...]
pf1_ref = [[xf1[t]; pf1[2];  zf1[t]] for t = 1:T]
f1_contact = [t < (Tm + T_fix) ? 1 : 0 for t = 1:T]

xf4_el, zf4_el = ellipse_traj(pf4[1], pf4[1] + strd, zh, Tm - T_fix)
xf4 = [[pf4[1] for t = 1:Tm + T_fix]..., xf4_el[2:end]...]
zf4 = [[pf4[3] for t = 1:Tm + T_fix]..., zf4_el[2:end]...]
pf4_ref = [[xf4[t]; pf4[2]; zf4[t]] for t = 1:T]
f4_contact = [t < (Tm + T_fix) ? 1 : 0 for t = 1:T]

xf2_el, zf2_el = ellipse_traj(pf2[1], pf2[1] + strd, zh, Tm - T_fix)
xf2 = [[xf2_el[1] for t = 1:T_fix]..., xf2_el..., [xf2_el[end] for t = 1:Tm-1 + T_fix]...]
zf2 = [[zf2_el[1] for t = 1:T_fix]..., zf2_el..., [zf2_el[end] for t = 1:Tm-1 + T_fix]...]
pf2_ref = [[xf2[t]; pf2[2]; zf2[t]] for t = 1:T]
f2_contact = [t <= T_fix ? 1 : (t > Tm ? 1 : 0) for t = 1:T]

xf3_el, zf3_el = ellipse_traj(pf3[1], pf3[1] + strd, zh, Tm - T_fix)
xf3 = [[xf3_el[1] for t = 1:T_fix]..., xf3_el..., [xf3_el[end] for t = 1:Tm-1]...]
zf3 = [[zf3_el[1] for t = 1:T_fix]..., zf3_el..., [zf3_el[end] for t = 1:Tm-1]...]
pf3_ref = [[xf3[t]; pf3[2]; zf3[t]] for t = 1:T]
f3_contact = [t <= T_fix ? 1 : (t > Tm ? 1 : 0) for t = 1:T]

tr = range(0, stop = tf, length = T)
plot(tr, hcat(pf1_ref...)', labels = "")
plot!(tr, hcat(pf4_ref...)', labels = "")

plot!(tr, hcat(pf2_ref...)', labels = "")
plot!(tr, hcat(pf3_ref...)', labels = "")

visualize!(vis, model, q_body_ref,
    pf1_ref, pf2_ref, pf3_ref, pf4_ref,
    Δt = h)

p_fr = plot(tr, f1_contact, linetype = :steppost, color = :black, label = "FR", xaxis = false, x_ticks = false, yaxis = false, yticks = false)
p_rl = plot(tr, f4_contact, linetype = :steppost, color = :black, label = "RL", xaxis = false, x_ticks = false, yaxis = false, yticks = false)

p_fl = plot(tr, f2_contact, linetype = :steppost, color = :black, label = "FL", xaxis = false, x_ticks = false, yaxis = false, yticks = false)
p_rr = plot(tr, f3_contact, linetype = :steppost, color = :black, label = "RR", yaxis = false, yticks = false, xaxis = "time (s)")

plot(p_fr, p_rl, p_fl, p_rr, layout = (4, 1))

contact_modes = [[f1_contact[t];
                  f2_contact[t];
                  f3_contact[t];
                  f4_contact[t]] for t = 1:T]

contact_modes_2 = [contact_modes..., contact_modes[2:end]...]

plot(hcat(contact_modes_2...)', linetype = :steppost, label = "")

T2 = 2 * T - 1
tf2 = 2 * tf
tr2 = range(0, stop = tf2, length = T2)
pf1_ref_2 = [pf1_ref..., [p + foot_shift for p in pf1_ref[2:end]]...]
pf2_ref_2 = [pf2_ref..., [p + foot_shift for p in pf2_ref[2:end]]...]
pf3_ref_2 = [pf3_ref..., [p + foot_shift for p in pf3_ref[2:end]]...]
pf4_ref_2 = [pf4_ref..., [p + foot_shift for p in pf4_ref[2:end]]...]
qT_body_2 = q_body + 4 * body_shift
q_body_ref_2 = linear_interpolation(q_body, qT_body_2, T2)#[q_body, linear_interpolation(q_body, qM_body, Tm - 1)..., linear_interpolation(qM_body, qT_body, Tm - 1)...]

plot(tr2, hcat(pf1_ref_2...)', labels = "")
plot!(tr2, hcat(pf2_ref_2...)', labels = "")
plot!(tr2, hcat(pf3_ref_2...)', labels = "")
plot!(tr2, hcat(pf4_ref_2...)', labels = "")
