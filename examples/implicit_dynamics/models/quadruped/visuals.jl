include(joinpath(pwd(), "models/visualize.jl"))

function visualize!(vis, model,
    q_body, q_f1, q_f2, q_f3, q_f4;
    q_body_ref = q_body,
    ref_tl = 0.5,
    r = 0.025,
    l_torso = 0.367,
    w_torso = 0.267,
    Δt = 0.1)

	default_background!(vis)

	setobject!(vis["torso"],
    	Rect(Vec(-0.5 * l_torso, -0.5 * w_torso, -0.025),Vec(l_torso, w_torso, 0.05)),
    	MeshPhongMaterial(color = RGBA(0.0, 0.0, 0.0, 1.0)))

    setobject!(vis["torso_ref"],
    	Rect(Vec(-0.5 * l_torso, -0.5 * w_torso, -0.025),Vec(l_torso, w_torso, 0.05)),
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

function foot_vis!(vis, i, t, contact_modes;
    h = 0.005,
    r = 0.035,
    tl = 0.5)

    if convert(Bool, contact_modes[t][i])
        setobject!(vis["f$(i)_$(t)"],
            Cylinder(Point(0.0, 0.0, 0.0), Point(0.0, 0.0, h), r),
            MeshPhongMaterial(color = RGBA(0.0, 1.0, 0.0, tl)))
        settransform!(vis["f$(i)_$(t)"], Translation(eval(Symbol("pf$(i)_ref"))[t]))
    end
    # setvisible!(vis["f$(i)_$(j)_$(t)"], convert(Bool, contact_modes[t][i]))
end

function cone_vis!(vis, i, t, contact_modes;
    h = 0.05,
    l = h * 1.0,
    w = h * 1.45,
    tl = 0.5,
    n = 10)

    pyramid = Pyramid(Point3(0.0, 0.0, 0.0), l, w)

    if convert(Bool, contact_modes[t][i])
        for p = 1:n
            setobject!(vis["pyramid$(i)_$(p)_$(t)"], pyramid,
                MeshPhongMaterial(
                # color = RGBA(1,153/255,51/255, 1.0))
                color = RGBA(51/255,1,1, tl))
                )
            settransform!(vis["pyramid$(i)_$(p)_$(t)"],
                compose(Translation(eval(Symbol("pf$(i)_ref"))[t][1], eval(Symbol("pf$(i)_ref"))[t][2], l),
                    LinearMap(RotX(π) * RotZ(π * p / n))),
                    )
            setvisible!(vis["pyramid$(i)_$(p)_$(t)"], convert(Bool, contact_modes[t][i]))
        end
    end
end

function force_vis!(vis, u, i, t;
    μ_friction = 1.0,
    f_max = 1.0,
    cone_vis_scale = 0.1)

    u_idx = (i - 1) * 3 .+ (1:3)
    ui = u[t][u_idx]
    us = cone_vis_scale * ui / f_max
    us[3] *= μ_friction
    p1 = eval(Symbol("pf$(i)_ref"))[t]

    force_vis = ArrowVisualizer(vis["force_$(i)_$(t)"])
    setobject!(force_vis, MeshPhongMaterial(color=RGBA(1.0, 0.0, 0.0, 1.0)))

    settransform!(force_vis,
    			Point(p1[1] - 0.0 * us[1], p1[2] - 0.0 * us[2], p1[3] - 0.0 * us[3]),
    			Vec(us[1], us[2], us[3]),
    			shaft_radius=0.01,
    			max_head_radius=0.025)
end
