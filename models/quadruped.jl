struct Quadruped{T}
    n::Int
    m::Int
    d::Int

    g::T
    μ::T

    # torso
    l1
    d1
    m1
    J1

    # leg 1
        # thigh
    l2
    d2
    m2
    J2
        # calf
    l3
    d3
    m3
    J3

    # leg 2
        # thigh
    l4
    d4
    m4
    J4
        # calf
    l5
    d5
    m5
    J5

	# leg 3
        # thigh
    l6
    d6
    m6
    J6
        # calf
    l7
    d7
    m7
    J7

	# leg 4
        # thigh
    l8
    d8
    m8
    J8
        # calf
    l9
    d9
    m9
    J9

    # joint limits
    qL
    qU

    # torque limits
    uL
    uU

    nq
    nu
    nc
    nf
    nb
    ns

    idx_u
    idx_λ
    idx_b
    idx_ψ
    idx_η
    idx_s
end

# Dimensions
nq = 2 + 5 + 4            # configuration dimension
nu = 4 + 4                # control dimension
nc = 4                    # number of contact points
nf = 2                    # number of parameters for friction cone
nb = nc * nf
ns = 1

# World parameters
μ = 1.0      # coefficient of friction
g = 9.81     # gravity

# Model parameters
m_torso = 0.5 + 0.48 * 2.0
m_thigh = 0.8112
m_leg = 0.3037

J_torso = 0.0029
J_thigh = 0.00709
J_leg = 0.00398

l_torso = 0.75
l_thigh = 0.2755
l_leg = 0.308

d_torso = 0.75 / 2.0
d_thigh = 0.2176
d_leg = 0.1445

n = 2 * nq
m = nu + nc + nb + nc + nb + ns
d = 0

idx_u = (1:nu)
idx_λ = nu .+ (1:nc)
idx_b = nu + nc .+ (1:nb)
idx_ψ = nu + nc + nb .+ (1:nc)
idx_η = nu + nc + nb + nc .+ (1:nb)
idx_s = nu + nc + nb + nc + nb .+ (1:ns)

qL = -Inf * ones(nq)
qU = Inf * ones(nq)

uL = -25.0 * ones(nu) # -16.0 * ones(nu)
uU = 25.0 * ones(nu) # 16.0 * ones(nu)

function kinematics_1(model::Quadruped, q; body = :torso, mode = :ee)
	x = q[1]
	z = q[2]

	if body == :torso
		l = model.l1
		d = model.d1
		θ = q[3]
	elseif body == :thigh_1
		l = model.l2
		d = model.d2
		θ = q[4]
	elseif body == :thigh_2
		l = model.l4
		d = model.d4
		θ = q[6]
	else
		@error "incorrect body specification"
	end

	if mode == :ee
		return [x + l * sin(θ); z - l * cos(θ)]
	elseif mode == :com
		return [x + d * sin(θ); z - d * cos(θ)]
	else
		@error "incorrect mode specification"
	end
end

function jacobian_1(model::Quadruped, q; body = :torso, mode = :ee)
	# jac = [1.0 0.0 0 0.0 0.0 0.0 0.0;
	# 	   0.0 1.0 0 0.0 0.0 0.0 0.0]
	jac = zeros(eltype(q), 2, model.nq)
	jac[1, 1] = 1.0
	jac[2, 2] = 1.0
	if body == :torso
		r = mode == :ee ? model.l1 : model.d1
		θ = q[3]
		jac[1, 3] = r * cos(θ)
		jac[2, 3] = r * sin(θ)
	elseif body == :thigh_1
		r = mode == :ee ? model.l2 : model.d2
		θ = q[4]
		jac[1, 4] = r * cos(θ)
		jac[2, 4] = r * sin(θ)
	elseif body == :thigh_2
		r = mode == :ee ? model.l4 : model.d4
		θ = q[6]
		jac[1, 6] = r * cos(θ)
		jac[2, 6] = r * sin(θ)
	else
		@error "incorrect body specification"
	end

	return jac
end

function kinematics_2(model::Quadruped, q; body = :leg_1, mode = :ee)

	if body == :leg_1
		p = kinematics_1(model, q, body = :thigh_1, mode = :ee)

		θa = q[4]
		θb = q[5]

		lb = model.l3
		db = model.d3
	elseif body == :leg_2
		p = kinematics_1(model, q, body = :thigh_2, mode = :ee)

		θa = q[6]
		θb = q[7]

		lb = model.l5
		db = model.d5
	elseif body == :thigh_3
		p = kinematics_1(model, q, body = :torso, mode = :ee)

		θa = q[3]
		θb = q[8]

		lb = model.l6
		db = model.d6
	elseif body == :thigh_4
		p = kinematics_1(model, q, body = :torso, mode = :ee)

		θa = q[3]
		θb = q[10]

		lb = model.l8
		db = model.d8
	else
		@error "incorrect body specification"
	end

	if mode == :ee
		return p + [lb * sin(θa + θb); -1.0 * lb * cos(θa + θb)]
	elseif mode == :com
		return p + [db * sin(θa + θb); -1.0 * db * cos(θa + θb)]
	else
		@error "incorrect mode specification"
	end
end

function jacobian_2(model::Quadruped, q; body = :leg_1, mode = :ee)

	if body == :leg_1
		jac = jacobian_1(model, q, body = :thigh_1, mode = :ee)

		θa = q[4]
		θb = q[5]

		r = mode == :ee ? model.l3 : model.d3

		jac[1, 4] += r * cos(θa + θb)
		jac[1, 5] += r * cos(θa + θb)

		jac[2, 4] += r * sin(θa + θb)
		jac[2, 5] += r * sin(θa + θb)
	elseif body == :leg_2
		jac = jacobian_1(model, q, body = :thigh_2, mode = :ee)

		θa = q[6]
		θb = q[7]

		r = mode == :ee ? model.l5 : model.d5

		jac[1, 6] += r * cos(θa + θb)
		jac[1, 7] += r * cos(θa + θb)

		jac[2, 6] += r * sin(θa + θb)
		jac[2, 7] += r * sin(θa + θb)
	elseif body == :thigh_3
		jac = jacobian_1(model, q, body = :torso, mode = :ee)

		θa = q[3]
		θb = q[8]

		r = mode == :ee ? model.l6 : model.d6

		jac[1, 3] += r * cos(θa + θb)
		jac[1, 8] += r * cos(θa + θb)

		jac[2, 3] += r * sin(θa + θb)
		jac[2, 8] += r * sin(θa + θb)
	elseif body == :thigh_4
		jac = jacobian_1(model, q, body = :torso, mode = :ee)

		θa = q[3]
		θb = q[10]

		r = mode == :ee ? model.l8 : model.d8

		jac[1, 3] += r * cos(θa + θb)
		jac[1, 10] += r * cos(θa + θb)

		jac[2, 3] += r * sin(θa + θb)
		jac[2, 10] += r * sin(θa + θb)
	else
		@error "incorrect body specification"
	end

	return jac
end

function kinematics_3(model::Quadruped, q; body = :leg_3, mode = :ee)

	if body == :leg_3
		p = kinematics_2(model, q, body = :thigh_3, mode = :ee)

		θa = q[3]
		θb = q[8]
		θc = q[9]


		lb = model.l7
		db = model.d7
	elseif body == :leg_4
		p = kinematics_2(model, q, body = :thigh_4, mode = :ee)

		θa = q[3]
		θb = q[10]
		θc = q[11]

		lb = model.l9
		db = model.d9
	else
		@error "incorrect body specification"
	end

	if mode == :ee
		return p + [lb * sin(θa + θb + θc); -1.0 * lb * cos(θa + θb + θc)]
	elseif mode == :com
		return p + [db * sin(θa + θb + θc); -1.0 * db * cos(θa + θb + θc)]
	else
		@error "incorrect mode specification"
	end
end

function jacobian_3(model::Quadruped, q; body = :leg_3, mode = :ee)

	if body == :leg_3
		jac = jacobian_2(model, q, body = :thigh_3, mode = :ee)

		θa = q[3]
		θb = q[8]
		θc = q[9]


		r = mode == :ee ? model.l7 : model.d7

		jac[1, 3] += r * cos(θa + θb + θc)
		jac[1, 8] += r * cos(θa + θb + θc)
		jac[1, 9] += r * cos(θa + θb + θc)

		jac[2, 3] += r * sin(θa + θb + θc)
		jac[2, 8] += r * sin(θa + θb + θc)
		jac[2, 9] += r * sin(θa + θb + θc)

	elseif body == :leg_4
		jac = jacobian_2(model, q, body = :thigh_4, mode = :ee)

		θa = q[3]
		θb = q[10]
		θc = q[11]


		r = mode == :ee ? model.l9 : model.d9

		jac[1, 3] += r * cos(θa + θb + θc)
		jac[1, 10] += r * cos(θa + θb + θc)
		jac[1, 11] += r * cos(θa + θb + θc)

		jac[2, 3] += r * sin(θa + θb + θc)
		jac[2, 10] += r * sin(θa + θb + θc)
		jac[2, 11] += r * sin(θa + θb + θc)
	else
		@error "incorrect body specification"
	end

	return jac
end

#test kinematics
# q = rand(nq)
#
# @assert norm(kinematics_1(model, q, body = :torso, mode = :ee) - [q[1] + model.l1 * sin(q[3]); q[2] - model.l1 * cos(q[3])]) ≈ 0.0
# @assert norm(kinematics_1(model, q, body = :torso, mode = :com) - [q[1] + model.d1 * sin(q[3]); q[2] - model.d1 * cos(q[3])]) ≈ 0.0
# @assert norm(kinematics_1(model, q, body = :thigh_1, mode = :ee) - [q[1] + model.l2 * sin(q[4]); q[2] - model.l2 * cos(q[4])]) ≈ 0.0
# @assert norm(kinematics_1(model, q, body = :thigh_1, mode = :com) - [q[1] + model.d2 * sin(q[4]); q[2] - model.d2 * cos(q[4])]) ≈ 0.0
# @assert norm(kinematics_1(model, q, body = :thigh_2, mode = :ee) - [q[1] + model.l4 * sin(q[6]); q[2] - model.l4 * cos(q[6])]) ≈ 0.0
# @assert norm(kinematics_1(model, q, body = :thigh_2, mode = :com) - [q[1] + model.d4 * sin(q[6]); q[2] - model.d4 * cos(q[6])]) ≈ 0.0
#
# @assert norm(kinematics_2(model, q, body = :leg_1, mode = :ee) - [q[1] + model.l2 * sin(q[4]) + model.l3 * sin(q[4] + q[5]); q[2] - model.l2 * cos(q[4]) - model.l3 * cos(q[4] + q[5])]) ≈ 0.0
# @assert norm(kinematics_2(model, q, body = :leg_1, mode = :com) - [q[1] + model.l2 * sin(q[4]) + model.d3 * sin(q[4] + q[5]); q[2] - model.l2 * cos(q[4]) - model.d3 * cos(q[4] + q[5])]) ≈ 0.0
# @assert norm(kinematics_2(model, q, body = :leg_2, mode = :ee) - [q[1] + model.l4 * sin(q[6]) + model.l5 * sin(q[6] + q[7]); q[2] - model.l4 * cos(q[6]) - model.l5 * cos(q[6] + q[7])]) ≈ 0.0
# @assert norm(kinematics_2(model, q, body = :leg_2, mode = :com) - [q[1] + model.l4 * sin(q[6]) + model.d5 * sin(q[6] + q[7]); q[2] - model.l4 * cos(q[6]) - model.d5 * cos(q[6] + q[7])]) ≈ 0.0
# @assert norm(kinematics_2(model, q, body = :thigh_3, mode = :ee) - [q[1] + model.l1 * sin(q[3]) + model.l6 * sin(q[3] + q[8]); q[2] - model.l1 * cos(q[3]) - model.l6 * cos(q[3] + q[8])]) ≈ 0.0
# @assert norm(kinematics_2(model, q, body = :thigh_3, mode = :com) - [q[1] + model.l1 * sin(q[3]) + model.d6 * sin(q[3] + q[8]); q[2] - model.l1 * cos(q[3]) - model.d6 * cos(q[3] + q[8])]) ≈ 0.0
# @assert norm(kinematics_2(model, q, body = :thigh_4, mode = :ee) - [q[1] + model.l1 * sin(q[3]) + model.l8 * sin(q[3] + q[10]); q[2] - model.l1 * cos(q[3]) - model.l8 * cos(q[3] + q[10])]) ≈ 0.0
# @assert norm(kinematics_2(model, q, body = :thigh_4, mode = :com) - [q[1] + model.l1 * sin(q[3]) + model.d8 * sin(q[3] + q[10]); q[2] - model.l1 * cos(q[3]) - model.d8 * cos(q[3] + q[10])]) ≈ 0.0
#
# @assert norm(kinematics_3(model, q, body = :leg_3, mode = :ee) - [q[1] + model.l1 * sin(q[3]) + model.l6 * sin(q[3] + q[8]) + model.l7 * sin(q[3] + q[8] + q[9]); q[2] - model.l1 * cos(q[3]) - model.l6 * cos(q[3] + q[8]) - model.l7 * cos(q[3] + q[8] + q[9])]) ≈ 0.0
# @assert norm(kinematics_3(model, q, body = :leg_3, mode = :com) - [q[1] + model.l1 * sin(q[3]) + model.l6 * sin(q[3] + q[8]) + model.d7 * sin(q[3] + q[8] + q[9]); q[2] - model.l1 * cos(q[3]) - model.l6 * cos(q[3] + q[8]) - model.d7 * cos(q[3] + q[8] + q[9])]) ≈ 0.0
# @assert norm(kinematics_3(model, q, body = :leg_4, mode = :ee) - [q[1] + model.l1 * sin(q[3]) + model.l8 * sin(q[3] + q[10]) + model.l9 * sin(q[3] + q[10] + q[11]); q[2] - model.l1 * cos(q[3]) - model.l8 * cos(q[3] + q[10]) - model.l9 * cos(q[3] + q[10] + q[11])]) ≈ 0.0
# @assert norm(kinematics_3(model, q, body = :leg_4, mode = :com) - [q[1] + model.l1 * sin(q[3]) + model.l8 * sin(q[3] + q[10]) + model.d9 * sin(q[3] + q[10] + q[11]); q[2] - model.l1 * cos(q[3]) - model.l8 * cos(q[3] + q[10]) - model.d9 * cos(q[3] + q[10] + q[11])]) ≈ 0.0
#
# k1(z) = kinematics_1(model, z, body = :torso, mode = :ee)
# @assert norm(ForwardDiff.jacobian(k1,q) - jacobian_1(model, q, body = :torso, mode = :ee)) ≈ 0.0
# k1(z) = kinematics_1(model, z, body = :torso, mode = :com)
# @assert norm(ForwardDiff.jacobian(k1,q) - jacobian_1(model, q, body = :torso, mode = :com)) ≈ 0.0
# k1(z) = kinematics_1(model, z, body = :thigh_1, mode = :ee)
# @assert norm(ForwardDiff.jacobian(k1,q) - jacobian_1(model, q, body = :thigh_1, mode = :ee)) ≈ 0.0
# k1(z) = kinematics_1(model, z, body = :thigh_1, mode = :com)
# @assert norm(ForwardDiff.jacobian(k1,q) - jacobian_1(model, q, body = :thigh_1, mode = :com)) ≈ 0.0
# k1(z) = kinematics_1(model, z, body = :thigh_2, mode = :ee)
# @assert norm(ForwardDiff.jacobian(k1,q) - jacobian_1(model, q, body = :thigh_2, mode = :ee)) ≈ 0.0
# k1(z) = kinematics_1(model, z, body = :thigh_2, mode = :com)
# @assert norm(ForwardDiff.jacobian(k1,q) - jacobian_1(model, q, body = :thigh_2, mode = :com)) ≈ 0.0
#
# k2(z) = kinematics_2(model, z, body = :leg_1, mode = :ee)
# @assert norm(ForwardDiff.jacobian(k2,q) - jacobian_2(model, q, body = :leg_1, mode = :ee)) ≈ 0.0
# k2(z) = kinematics_2(model, z, body = :leg_1, mode = :com)
# @assert norm(ForwardDiff.jacobian(k2,q) - jacobian_2(model, q, body = :leg_1, mode = :com)) ≈ 0.0
# k2(z) = kinematics_2(model, z, body = :leg_2, mode = :ee)
# @assert norm(ForwardDiff.jacobian(k2,q) - jacobian_2(model, q, body = :leg_2, mode = :ee)) ≈ 0.0
# k2(z) = kinematics_2(model, z, body = :leg_2, mode = :com)
# @assert norm(ForwardDiff.jacobian(k2,q) - jacobian_2(model, q, body = :leg_2, mode = :com)) ≈ 0.0
# k2(z) = kinematics_2(model, z, body = :thigh_3, mode = :ee)
# @assert norm(ForwardDiff.jacobian(k2,q) - jacobian_2(model, q, body = :thigh_3, mode = :ee)) ≈ 0.0
# k2(z) = kinematics_2(model, z, body = :thigh_3, mode = :com)
# @assert norm(ForwardDiff.jacobian(k2,q) - jacobian_2(model, q, body = :thigh_3, mode = :com)) ≈ 0.0
# k2(z) = kinematics_2(model, z, body = :thigh_4, mode = :ee)
# @assert norm(ForwardDiff.jacobian(k2,q) - jacobian_2(model, q, body = :thigh_4, mode = :ee)) ≈ 0.0
# k2(z) = kinematics_2(model, z, body = :thigh_4, mode = :com)
# @assert norm(ForwardDiff.jacobian(k2,q) - jacobian_2(model, q, body = :thigh_4, mode = :com)) ≈ 0.0
#
# k3(z) = kinematics_3(model, z, body = :leg_3, mode = :ee)
# @assert norm(ForwardDiff.jacobian(k3, q) - jacobian_3(model, q, body = :leg_3, mode = :ee)) ≈ 0.0
# k3(z) = kinematics_3(model, z, body = :leg_3, mode = :com)
# @assert norm(ForwardDiff.jacobian(k3, q) - jacobian_3(model, q, body = :leg_3, mode = :com)) ≈ 0.0
# k3(z) = kinematics_3(model, z, body = :leg_4, mode = :ee)
# @assert norm(ForwardDiff.jacobian(k3, q) - jacobian_3(model, q, body = :leg_4, mode = :ee)) ≈ 0.0
# k3(z) = kinematics_3(model, z, body = :leg_4, mode = :com)
# @assert norm(ForwardDiff.jacobian(k3, q) - jacobian_3(model, q, body = :leg_4, mode = :com)) ≈ 0.0

# Lagrangian

function lagrangian(model::Quadruped, q, q̇)
	L = 0.0

	# torso
	p_torso = kinematics_1(model, q, body = :torso, mode = :com)
	J_torso = jacobian_1(model, q, body = :torso, mode = :com)
	v_torso = J_torso * q̇

	L += 0.5 * model.m1 * v_torso' * v_torso
	L += 0.5 * model.J1 * q̇[3]^2.0
	L -= model.m1 * model.g * p_torso[2]

	# thigh 1
	p_thigh_1 = kinematics_1(model, q, body = :thigh_1, mode = :com)
	J_thigh_1 = jacobian_1(model, q, body = :thigh_1, mode = :com)
	v_thigh_1 = J_thigh_1 * q̇

	L += 0.5 * model.m2 * v_thigh_1' * v_thigh_1
	L += 0.5 * model.J2 * q̇[4]^2.0
	L -= model.m2 * model.g * p_thigh_1[2]

	# leg 1
	p_leg_1 = kinematics_2(model, q, body = :leg_1, mode = :com)
	J_leg_1 = jacobian_2(model, q, body = :leg_1, mode = :com)
	v_leg_1 = J_leg_1 * q̇

	L += 0.5 * model.m3 * v_leg_1' * v_leg_1
	L += 0.5 * model.J3 * q̇[5]^2.0
	L -= model.m3 * model.g * p_leg_1[2]

	# thigh 2
	p_thigh_2 = kinematics_1(model, q, body = :thigh_2, mode = :com)
	J_thigh_2 = jacobian_1(model, q, body = :thigh_2, mode = :com)
	v_thigh_2 = J_thigh_2 * q̇

	L += 0.5 * model.m4 * v_thigh_2' * v_thigh_2
	L += 0.5 * model.J4 * q̇[6]^2.0
	L -= model.m4 * model.g * p_thigh_2[2]

	# leg 2
	p_leg_2 = kinematics_2(model, q, body = :leg_2, mode = :com)
	J_leg_2 = jacobian_2(model, q, body = :leg_2, mode = :com)
	v_leg_2 = J_leg_2 * q̇

	L += 0.5 * model.m5 * v_leg_2' * v_leg_2
	L += 0.5 * model.J5 * q̇[7]^2.0
	L -= model.m5 * model.g * p_leg_2[2]

	# thigh 3
	p_thigh_3 = kinematics_2(model, q, body = :thigh_3, mode = :com)
	J_thigh_3 = jacobian_2(model, q, body = :thigh_3, mode = :com)
	v_thigh_3 = J_thigh_3 * q̇

	L += 0.5 * model.m6 * v_thigh_3' * v_thigh_3
	L += 0.5 * model.J6 * q̇[8]^2.0
	L -= model.m6 * model.g * p_thigh_3[2]

	# leg 3
	p_leg_3 = kinematics_3(model, q, body = :leg_3, mode = :com)
	J_leg_3 = jacobian_3(model, q, body = :leg_3, mode = :com)
	v_leg_3 = J_leg_3 * q̇

	L += 0.5 * model.m7 * v_leg_3' * v_leg_3
	L += 0.5 * model.J7 * q̇[9]^2.0
	L -= model.m7 * model.g * p_leg_3[2]

	# thigh 4
	p_thigh_4 = kinematics_2(model, q, body = :thigh_4, mode = :com)
	J_thigh_4 = jacobian_2(model, q, body = :thigh_4, mode = :com)
	v_thigh_4 = J_thigh_4 * q̇

	L += 0.5 * model.m8 * v_thigh_4' * v_thigh_4
	L += 0.5 * model.J8 * q̇[10]^2.0
	L -= model.m8 * model.g * p_thigh_4[2]

	# leg 4
	p_leg_4 = kinematics_3(model, q, body = :leg_4, mode = :com)
	J_leg_4 = jacobian_3(model, q, body = :leg_4, mode = :com)
	v_leg_4 = J_leg_4 * q̇

	L += 0.5 * model.m9 * v_leg_4' * v_leg_4
	L += 0.5 * model.J9 * q̇[11]^2.0
	L -= model.m9 * model.g * p_leg_4[2]

	return L
end

function dLdq(model::Quadruped, q, q̇)
	Lq(x) = lagrangian(model, x, q̇)
	ForwardDiff.gradient(Lq, q)
end

function dLdq̇(model::Quadruped, q, q̇)
	Lq̇(x) = lagrangian(model, q, x)
	ForwardDiff.gradient(Lq̇, q̇)
end

# q̇ = rand(nq)
# lagrangian(model, q, q̇)
# dLdq(model, q, q̇)
# dLdq̇(model, q, q̇)


# Methods
function M_func(model::Quadruped, q)
	M = Diagonal([0.0, 0.0, model.J1, model.J2, model.J3, model.J4, model.J5, model.J6, model.J7, model.J8, model.J9])

	# torso
	J_torso = jacobian_1(model, q, body = :torso, mode = :com)
	M += model.m1 * J_torso' * J_torso

	# thigh 1
	J_thigh_1 = jacobian_1(model, q, body = :thigh_1, mode = :com)
	M += model.m2 * J_thigh_1' * J_thigh_1

	# leg 1
	J_leg_1 = jacobian_2(model, q, body = :leg_1, mode = :com)
	M += model.m3 * J_leg_1' * J_leg_1

	# thigh 2
	J_thigh_2 = jacobian_1(model, q, body = :thigh_2, mode = :com)
	M += model.m4 * J_thigh_2' * J_thigh_2

	# leg 2
	J_leg_2 = jacobian_2(model, q, body = :leg_2, mode = :com)
	M += model.m5 * J_leg_2' * J_leg_2

	# thigh 3
	J_thigh_3 = jacobian_2(model, q, body = :thigh_3, mode = :com)
	M += model.m6 * J_thigh_3' * J_thigh_3

	# leg 3
	J_leg_3 = jacobian_3(model, q, body = :leg_3, mode = :com)
	M += model.m7 * J_leg_3' * J_leg_3

	# thigh 4
	J_thigh_4 = jacobian_2(model, q, body = :thigh_4, mode = :com)
	M += model.m8 * J_thigh_4' * J_thigh_4

	# leg 4
	J_leg_4 = jacobian_3(model, q, body = :leg_4, mode = :com)
	M += model.m9 * J_leg_4' * J_leg_4

	return M
end

# eigen(M_func(model, q))
#
# tmp_q(z) = dLdq̇(model, z, q̇)
# tmp_q̇(z) = dLdq̇(model, q, z)
# norm(ForwardDiff.jacobian(tmp_q̇,q̇) - M_func(model, q))

function C_func(model::Quadruped, q, q̇)
	tmp_q(z) = dLdq̇(model, z, q̇)
	tmp_q̇(z) = dLdq̇(model, q, z)

	ForwardDiff.jacobian(tmp_q, q) * q̇ - dLdq(model, q, q̇)
end

function ϕ_func(model::Quadruped, q)
	p_leg_1 = kinematics_2(model, q, body = :leg_1, mode = :ee)
	p_leg_2 = kinematics_2(model, q, body = :leg_2, mode = :ee)
	p_leg_3 = kinematics_3(model, q, body = :leg_3, mode = :ee)
	p_leg_4 = kinematics_3(model, q, body = :leg_4, mode = :ee)

	@SVector [p_leg_1[2], p_leg_2[2], p_leg_3[2], p_leg_4[2]]
end

function B_func(model::Quadruped, q)
	@SMatrix [0.0 0.0 0.0 1.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0;
			  0.0 0.0 0.0 0.0 1.0 0.0 0.0 0.0 0.0 0.0 0.0;
			  0.0 0.0 0.0 0.0 0.0 1.0 0.0 0.0 0.0 0.0 0.0;
			  0.0 0.0 0.0 0.0 0.0 0.0 1.0 0.0 0.0 0.0 0.0
			  0.0 0.0 0.0 0.0 0.0 0.0 0.0 1.0 0.0 0.0 0.0;
			  0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 1.0 0.0 0.0;
			  0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 1.0 0.0;
			  0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 1.0]
end

function N_func(model::Quadruped, q)
	J_leg_1 = jacobian_2(model, q, body = :leg_1, mode = :ee)
	J_leg_2 = jacobian_2(model, q, body = :leg_2, mode = :ee)
	J_leg_3 = jacobian_3(model, q, body = :leg_3, mode = :ee)
	J_leg_4 = jacobian_3(model, q, body = :leg_4, mode = :ee)

	return [view(J_leg_1, 2:2, :);
			view(J_leg_2, 2:2, :);
			view(J_leg_3, 2:2, :);
			view(J_leg_4, 2:2, :)]
end

function _P_func(model::Quadruped, q)
	J_leg_1 = jacobian_2(model, q, body = :leg_1, mode = :ee)
	J_leg_2 = jacobian_2(model, q, body = :leg_2, mode = :ee)
	J_leg_3 = jacobian_3(model, q, body = :leg_3, mode = :ee)
	J_leg_4 = jacobian_3(model, q, body = :leg_4, mode = :ee)

	return [view(J_leg_1, 1:1, :);
			view(J_leg_2, 1:1, :);
			view(J_leg_3, 1:1, :);
			view(J_leg_4, 1:1, :)]
end

function P_func(model::Quadruped, q)
	J_leg_1 = jacobian_2(model, q, body = :leg_1, mode = :ee)
	J_leg_2 = jacobian_2(model, q, body = :leg_2, mode = :ee)
	J_leg_3 = jacobian_3(model, q, body = :leg_3, mode = :ee)
	J_leg_4 = jacobian_3(model, q, body = :leg_4, mode = :ee)
	map = [1.0; -1.0]

	return [map * view(J_leg_1, 1:1, :);
			map * view(J_leg_2, 1:1, :);
			map * view(J_leg_3, 1:1, :);
			map * view(J_leg_4, 1:1, :)]
end

function friction_cone(model::Quadruped, u)
	λ = view(u, model.idx_λ)
	b = view(u, model.idx_b)
	return @SVector [model.μ * λ[1] - sum(view(b, 1:2)),
					 model.μ * λ[2] - sum(view(b, 3:4)),
					 model.μ * λ[3] - sum(view(b, 5:6)),
					 model.μ * λ[4] - sum(view(b, 7:8))]
end

function maximum_dissipation(model::Quadruped, x⁺, u, h)
	q3 = view(x⁺, model.nq .+ (1:model.nq))
	q2 = view(x⁺, 1:model.nq)
	ψ = view(u, model.idx_ψ)
	ψ_stack = [ψ[1] * ones(2); ψ[2] * ones(2); ψ[3] * ones(2); ψ[4] * ones(2)]
	η = view(u, model.idx_η)
	return P_func(model, q3) * (q3 - q2) / h + ψ_stack - η
end

function no_slip(model::Quadruped, x⁺, u, h)
	q3 = view(x⁺, model.nq .+ (1:model.nq))
	q2 = view(x⁺, 1:model.nq)
	λ = view(u, model.idx_λ)
	s = view(u, model.idx_s)
	λ_stack = [λ[1]; λ[2]; λ[3]; λ[4]]
	return s[1] - (λ_stack' * _P_func(model, q3) * (q3 - q2) / h)[1]
end

function fd(model::Quadruped, x⁺, x, u, w, h, t)
	q3 = view(x⁺, model.nq .+ (1:model.nq))
	q2⁺ = view(x⁺, 1:model.nq)
	q2⁻ = view(x, model.nq .+ (1:model.nq))
	q1 = view(x, 1:model.nq)
	u_ctrl = view(u, model.idx_u)
	λ = view(u, model.idx_λ)
	b = view(u, model.idx_b)

    [q2⁺ - q2⁻;
    ((1.0 / h) * (M_func(model, q1) * (SVector{11}(q2⁺) - SVector{11}(q1))
    - M_func(model, q2⁺) * (SVector{11}(q3) - SVector{11}(q2⁺)))
    + transpose(B_func(model, q3)) * SVector{8}(u_ctrl)
    + transpose(N_func(model, q3)) * SVector{4}(λ)
    + transpose(P_func(model, q3)) * SVector{8}(b)
    - h * C_func(model, q3, (q3 - q2⁺) / h))]
end

model = Quadruped(n, m, d,
			  g, μ,
			  l_torso, d_torso, m_torso, J_torso,
			  l_thigh, d_thigh, m_thigh, J_thigh,
			  l_leg, d_leg, m_leg, J_leg,
			  l_thigh, d_thigh, m_thigh, J_thigh,
			  l_leg, d_leg, m_leg, J_leg,
			  l_thigh, d_thigh, m_thigh, J_thigh,
			  l_leg, d_leg, m_leg, J_leg,
			  l_thigh, d_thigh, m_thigh, J_thigh,
			  l_leg, d_leg, m_leg, J_leg,
			  qL, qU,
			  uL, uU,
			  nq,
			  nu,
			  nc,
			  nf,
			  nb,
			  ns,
			  idx_u,
			  idx_λ,
			  idx_b,
			  idx_ψ,
			  idx_η,
			  idx_s)



# visualization
function visualize!(vis, model::Quadruped, q;
      r = 0.035, Δt = 0.1)


	torso = Cylinder(Point3f0(0.0, 0.0, 0.0), Point3f0(0.0, 0.0, model.l1),
		convert(Float32, 0.025))
	setobject!(vis["torso"], torso,
		MeshPhongMaterial(color = RGBA(0.0, 0.0, 0.0, 1.0)))

	thigh_1 = Cylinder(Point3f0(0.0,0.0,0.0), Point3f0(0.0, 0.0, model.l2),
		convert(Float32, 0.025))
	setobject!(vis["thigh1"], thigh_1,
		MeshPhongMaterial(color = RGBA(0.0, 0.0, 0.0, 1.0)))

	leg_1 = Cylinder(Point3f0(0.0,0.0,0.0), Point3f0(0.0, 0.0, model.l3),
		convert(Float32, 0.025))
	setobject!(vis["leg1"], leg_1,
		MeshPhongMaterial(color = RGBA(0.0, 0.0, 0.0, 1.0)))

	thigh_2 = Cylinder(Point3f0(0.0,0.0,0.0), Point3f0(0.0, 0.0, model.l4),
		convert(Float32, 0.025))
	setobject!(vis["thigh2"], thigh_2,
		MeshPhongMaterial(color = RGBA(0.0, 0.0, 0.0, 1.0)))

	leg_2 = Cylinder(Point3f0(0.0,0.0,0.0), Point3f0(0.0, 0.0, model.l5),
		convert(Float32, 0.025))
	setobject!(vis["leg2"], leg_2,
		MeshPhongMaterial(color = RGBA(0.0, 0.0, 0.0, 1.0)))

	thigh_3 = Cylinder(Point3f0(0.0,0.0,0.0), Point3f0(0.0, 0.0, model.l6),
		convert(Float32, 0.025))
	setobject!(vis["thigh3"], thigh_3,
		MeshPhongMaterial(color = RGBA(0.0, 0.0, 0.0, 1.0)))

	leg_3 = Cylinder(Point3f0(0.0,0.0,0.0), Point3f0(0.0, 0.0, model.l7),
		convert(Float32, 0.025))
	setobject!(vis["leg3"], leg_3,
		MeshPhongMaterial(color = RGBA(0.0, 0.0, 0.0, 1.0)))

	thigh_4 = Cylinder(Point3f0(0.0,0.0,0.0), Point3f0(0.0, 0.0, model.l8),
		convert(Float32, 0.025))
	setobject!(vis["thigh4"], thigh_4,
		MeshPhongMaterial(color = RGBA(0.0, 0.0, 0.0, 1.0)))

	leg_4 = Cylinder(Point3f0(0.0,0.0,0.0), Point3f0(0.0, 0.0, model.l9),
		convert(Float32, 0.025))
	setobject!(vis["leg4"], leg_4,
		MeshPhongMaterial(color = RGBA(0.0, 0.0, 0.0, 1.0)))

	anim = MeshCat.Animation(convert(Int, floor(1.0 / Δt)))

	hip1 = setobject!(vis["hip1"], GeometryTypes.Sphere(Point3f0(0),
        convert(Float32, 0.025)),
        MeshPhongMaterial(color = RGBA(0, 0, 0, 1.0)))

	hip2 = setobject!(vis["hip2"], GeometryTypes.Sphere(Point3f0(0),
        convert(Float32, 0.025)),
        MeshPhongMaterial(color = RGBA(0, 0, 0, 1.0)))

	knee1 = setobject!(vis["knee1"], GeometryTypes.Sphere(Point3f0(0),
        convert(Float32, 0.025)),
        MeshPhongMaterial(color = RGBA(0, 0, 0, 1.0)))

	knee2 = setobject!(vis["knee2"], GeometryTypes.Sphere(Point3f0(0),
		convert(Float32, 0.025)),
		MeshPhongMaterial(color = RGBA(0, 0, 0, 1.0)))

	knee3 = setobject!(vis["knee3"], GeometryTypes.Sphere(Point3f0(0),
		convert(Float32, 0.025)),
		MeshPhongMaterial(color = RGBA(0, 0, 0, 1.0)))

	knee4 = setobject!(vis["knee4"], GeometryTypes.Sphere(Point3f0(0),
        convert(Float32, 0.025)),
        MeshPhongMaterial(color = RGBA(0, 0, 0, 1.0)))

	feet1 = setobject!(vis["feet1"], GeometryTypes.Sphere(Point3f0(0),
        convert(Float32, r)),
        MeshPhongMaterial(color = RGBA(1.0, 165.0 / 255.0, 0, 1.0)))

	feet2 = setobject!(vis["feet2"], GeometryTypes.Sphere(Point3f0(0),
		convert(Float32, r)),
		MeshPhongMaterial(color = RGBA(1.0, 165.0 / 255.0, 0, 1.0)))

	feet3 = setobject!(vis["feet3"], GeometryTypes.Sphere(Point3f0(0),
		convert(Float32, r)),
		MeshPhongMaterial(color = RGBA(1.0, 165.0 / 255.0, 0, 1.0)))

	feet4 = setobject!(vis["feet4"], GeometryTypes.Sphere(Point3f0(0),
        convert(Float32, r)),
        MeshPhongMaterial(color = RGBA(1.0, 165.0 / 255.0, 0, 1.0)))

	T = length(q)
	p_shift = [0.0, 0.0, r]
	for t = 1:T
		MeshCat.atframe(anim, t) do
			p = [q[t][1]; 0.0; q[t][2]] + p_shift

			k_torso = kinematics_1(model, q[t], body = :torso, mode = :ee)
			p_torso = [k_torso[1], 0.0, k_torso[2]] + p_shift

			k_thigh_1 = kinematics_1(model, q[t], body = :thigh_1, mode = :ee)
			p_thigh_1 = [k_thigh_1[1], 0.0, k_thigh_1[2]] + p_shift

			k_leg_1 = kinematics_2(model, q[t], body = :leg_1, mode = :ee)
			p_leg_1 = [k_leg_1[1], 0.0, k_leg_1[2]] + p_shift

			k_thigh_2 = kinematics_1(model, q[t], body = :thigh_2, mode = :ee)
			p_thigh_2 = [k_thigh_2[1], 0.0, k_thigh_2[2]] + p_shift

			k_leg_2 = kinematics_2(model, q[t], body = :leg_2, mode = :ee)
			p_leg_2 = [k_leg_2[1], 0.0, k_leg_2[2]] + p_shift


			k_thigh_3 = kinematics_2(model, q[t], body = :thigh_3, mode = :ee)
			p_thigh_3 = [k_thigh_3[1], 0.0, k_thigh_3[2]] + p_shift

			k_leg_3 = kinematics_3(model, q[t], body = :leg_3, mode = :ee)
			p_leg_3 = [k_leg_3[1], 0.0, k_leg_3[2]] + p_shift

			k_thigh_4 = kinematics_2(model, q[t], body = :thigh_4, mode = :ee)
			p_thigh_4 = [k_thigh_4[1], 0.0, k_thigh_4[2]] + p_shift

			k_leg_4 = kinematics_3(model, q[t], body = :leg_4, mode = :ee)
			p_leg_4 = [k_leg_4[1], 0.0, k_leg_4[2]] + p_shift

			settransform!(vis["thigh1"], cable_transform(p, p_thigh_1))
			settransform!(vis["leg1"], cable_transform(p_thigh_1, p_leg_1))
			settransform!(vis["thigh2"], cable_transform(p, p_thigh_2))
			settransform!(vis["leg2"], cable_transform(p_thigh_2, p_leg_2))
			settransform!(vis["thigh3"], cable_transform(p_torso, p_thigh_3))
			settransform!(vis["leg3"], cable_transform(p_thigh_3, p_leg_3))
			settransform!(vis["thigh4"], cable_transform(p_torso, p_thigh_4))
			settransform!(vis["leg4"], cable_transform(p_thigh_4, p_leg_4))
			settransform!(vis["torso"], cable_transform(p, p_torso))
			settransform!(vis["hip1"], Translation(p))
			settransform!(vis["hip2"], Translation(p_torso))
			settransform!(vis["knee1"], Translation(p_thigh_1))
			settransform!(vis["knee2"], Translation(p_thigh_2))
			settransform!(vis["knee3"], Translation(p_thigh_3))
			settransform!(vis["knee4"], Translation(p_thigh_4))
			settransform!(vis["feet1"], Translation(p_leg_1))
			settransform!(vis["feet2"], Translation(p_leg_2))
			settransform!(vis["feet3"], Translation(p_leg_3))
			settransform!(vis["feet4"], Translation(p_leg_4))
		end
	end

	MeshCat.setanimation!(vis, anim)
end

# urdf = joinpath(pwd(), "models/Quadruped/urdf/Quadruped_float.urdf")
# mechanism = parse_urdf(urdf, floating=false)
# state = MechanismState(mechanism)
# state_cache = StateCache(mechanism)
# result = DynamicsResult(mechanism)
# result_cache = DynamicsResultCache(mechanism)
#
# q = zeros(model.nq)
# q_f = copy(q)
# q_f[3] = - pi
# q_f[4] = - pi / 2.0
# q_f[5] = - pi / 2.0
# set_configuration!(state, q)
# mass_matrix(state)
# M_func(model, q)
function initial_configuration(model::Quadruped, θ)
	q1 = zeros(model.nq)
	q1[3] = pi / 2.0
	q1[4] = θ #+ pi / 20.0
	q1[5] = -2.0 * θ
	q1[6] = θ #- pi / 20.0
	q1[7] = -2.0 * θ
	q1[2] = model.l2 * cos(θ) + model.l3 * cos(θ)

	q1[8] = -pi / 2.0 + θ
	q1[10] = -pi / 2.0 + θ

	q1[9] = -2.0 * θ
	q1[11] = -2.0 * θ

	return q1
end

include(joinpath(pwd(),"models/visualize.jl"))
vis = Visualizer()
render(vis)
q0 = initial_configuration(model, pi / 7.5)
visualize!(vis, model, [q0], Δt = 1.0)
