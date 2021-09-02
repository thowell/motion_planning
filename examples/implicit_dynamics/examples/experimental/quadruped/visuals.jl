function visualize!(vis, model,
    q_body, q_f1, q_f2, q_f3, q_f4;
    q_body_ref = q_body,
    ref_tl = 0.5,
    r = 0.025,
    l_torso = 0.19,
    w_torso = 0.1025,
    Δt = 0.1)

	default_background!(vis)

	setobject!(vis["torso"],
    	Rect(Vec(-l_torso, -w_torso, -0.025),Vec(2.0 * l_torso, 2.0 * w_torso, 0.05)),
    	MeshPhongMaterial(color = RGBA(0.0, 0.0, 0.0, 1.0)))

    setobject!(vis["torso_ref"],
    	Rect(Vec(-l_torso, -w_torso, -0.025),Vec(2.0 * l_torso, 2.0 * w_torso, 0.05)),
    	MeshPhongMaterial(color = RGBA(0.0, 0.0, 0.0, ref_tl)))

	feet1 = setobject!(vis["feet1"], Sphere(Point3f0(0),
        convert(Float32, r)),
        MeshPhongMaterial(color = RGBA(1.0, 165.0 / 255.0, 0, 1.0)))

	feet2 = setobject!(vis["feet2"], Sphere(Point3f0(0),
		convert(Float32, r)),
		MeshPhongMaterial(color = RGBA(1.0, 165.0 / 255.0, 0, 1.0)))

	feet3 = setobject!(vis["feet3"], Sphere(Point3f0(0),
		convert(Float32, r)),
		MeshPhongMaterial(color = RGBA(1.0, 165.0 / 255.0, 0, 1.0)))

	feet4 = setobject!(vis["feet4"], Sphere(Point3f0(0),
        convert(Float32, r)),
        MeshPhongMaterial(color = RGBA(1.0, 165.0 / 255.0, 0, 1.0)))

	anim = MeshCat.Animation(convert(Int, floor(1.0 / Δt)))

	T = length(q_body)
	p_shift = [0.0, 0.0, r]

	for t = 1:T
		MeshCat.atframe(anim, t) do
			rot = MRP(q_body[t][4:6]...)
            rot_ref = MRP(q_body_ref[t][4:6]...)


			p_torso = q_body[t][1:3] + p_shift
            p_torso_ref = q_body_ref[t][1:3] + p_shift
			p_foot1 = q_f1[t] + p_shift
			p_foot2 = q_f2[t] + p_shift
			p_foot3 = q_f3[t] + p_shift
			p_foot4 = q_f4[t] + p_shift

			settransform!(vis["torso"], compose(Translation(p_torso), LinearMap(rot)))
            settransform!(vis["torso_ref"], compose(Translation(p_torso_ref), LinearMap(rot_ref)))
			settransform!(vis["feet1"], Translation(p_foot1))
			settransform!(vis["feet2"], Translation(p_foot2))
			settransform!(vis["feet3"], Translation(p_foot3))
			settransform!(vis["feet4"], Translation(p_foot4))
		end
	end

	# settransform!(vis["/Cameras/default"],
	#     compose(Translation(0.0, 0.0, -1.0), LinearMap(RotZ(-pi / 2.0))))

	MeshCat.setanimation!(vis, anim)
end
