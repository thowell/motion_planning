using Symbolics
using LinearAlgebra
using StaticArrays

function L_multiply(q)
	s = q[1]
	v = q[2:4]

	SMatrix{4,4}([s -transpose(v);
	              v s * I + skew(v)])
end

skew(x) = [0 -x[3] x[2]; x[3] 0 -x[1]; -x[2] x[1] 0]
function R_multiply(q)
	s = q[1]
	v = q[2:4]

	SMatrix{4,4}([s -transpose(v);
	              v s * I - skew(v)])
end

function V_func() 
    [zeros(3) Diagonal(ones(3))]
end

function quaternion_rotation_matrix(q)
	r, i, j, k  = q

	r11 = 1.0 - 2.0 * (j^2.0 + k^2.0)
	r12 = 2.0 * (i * j - k * r)
	r13 = 2.0 * (i * k + j * r)

	r21 = 2.0 * (i * j + k * r)
	r22 = 1.0 - 2.0 * (i^2.0 + k^2.0)
	r23 = 2.0 * (j * k - i * r)

	r31 = 2.0 * (i * k - j * r)
	r32 = 2.0 * (j * k + i * r)
	r33 = 1.0 - 2.0 * (i^2.0 + j^2.0)

	SMatrix{3,3}([r11 r12 r13;
	              r21 r22 r23;
				  r31 r32 r33])
end

q0 = rand(4) 
q0 ./= norm(q0)
r0 = ones(3)

quaternion_rotation_matrix(q0) - V_func() * R_multiply(q0)' * L_multiply(q0) * V_func()'
@variables q[1:4], r[1:3]

y = quaternion_rotation_matrix(q) * r
y = V_func() * R_multiply(q)' * L_multiply(q) * V_func()' * r

y_jac = Symbolics.jacobian(y, q)


string(y_jac[1, 1])
y_jac[2, 1]
y_jac[3, 1]

y_jac[1, 2]
y_jac[2, 2]
y_jac[3, 2]

y_jac[1, 3]
y_jac[2, 3]
y_jac[3, 3]

string(y_jac[1, 4])
y_jac[2, 4]
y_jac[3, 4]

function jacobian(q, r) 
    q1, q2, q3, q4 = q
    r1, r2, r3 = r
    "2q₂*r₃ - (2q₁*r₂) - (2q₄*r₁)"


    



y_jac_func = eval(Symbolics.build_function(y_jac, q, r)[1])

y_jac_func(q0, r0) 


p0 = [1.0, 2.0, 4.0] 
q0 = [0.0, 1.0, 0.0, 0.0] 
q0 ./= norm(q0)
y_jac_func(q0, p0)

UnitQuaternion(q0) * p0 - quaternion_rotation_matrix(q0) * p0
q0 ./= norm(q0)

y_jac_func(q0, p0)


using Rotations

q = UnitQuaternion(RotX(0.5 * π))
p = [0.0, 0.0, 1.0]

q * p
q