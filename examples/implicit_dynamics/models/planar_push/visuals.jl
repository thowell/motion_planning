function visualize!(vis, model::PlanarPush, q, u; r = r,
        Δt = 0.1,
        contact_points = false)

	default_background!(vis)

    setobject!(vis["box"], GeometryBasics.Rect(Vec(-1.0 * r,
		-1.0 * r,
		-1.0 * r),
		Vec(2.0 * r, 2.0 * r, 2.0 * r)),
		MeshPhongMaterial(color = RGBA(0.0, 0.0, 0.0, 1.0)))

    if contact_points
        for i = 1:4
            setobject!(vis["contact$i"], GeometryBasics.Sphere(Point3f0(0),
                convert(Float32, 0.02)),
                MeshPhongMaterial(color = RGBA(1.0, 165.0 / 255.0, 0.0, 1.0)))
        end
    end

	force_vis = ArrowVisualizer(vis[:force])
	setobject!(force_vis, MeshPhongMaterial(color=RGBA(1.0, 0.0, 0.0, 1.0)))

	us = u[1] / 10.0

	settransform!(force_vis,
				Point(q[1][4] - us[1], q[1][5] - us[1], 0.0),
				Vec(us[1], us[1], 0.0),
				shaft_radius=0.01,
				max_head_radius=0.025)


    anim = MeshCat.Animation(convert(Int, floor(1.0 / Δt)))

	T = length(q)
    for t = 1:T-1
        MeshCat.atframe(anim, t) do
			if t < T-1

				if norm(u[t]) < 1.0e-6
					setvisible!(vis[:force], false)
				else
					setvisible!(vis[:force], true)

					us = u[t] / 10.0
					settransform!(force_vis,
								Point(q[t+1][4] - us[1], q[t+1][5] - us[2], 0.0),
								Vec(us[1], us[2], 0.0),
								shaft_radius=0.01,
								max_head_radius=0.025)
				end
			end

            settransform!(vis["box"],
				compose(Translation(q[t+1][1], q[t+1][2], r), LinearMap(RotZ(q[t+1][3]))))

            if contact_points
                for i = 1:4
                    settransform!(vis["contact$i"],
                        Translation(([q[t+1][1:2]; 0.0] + RotZ(q[t+1][3]) * [contact_corner_offset[i]; 0.0])...))
                end
            end
        end
    end

	settransform!(vis["/Cameras/default"],
		compose(Translation(0.0, 0.0, 50.0), LinearMap(RotZ(0.5 * pi) * RotY(-pi/2.5))))
	setprop!(vis["/Cameras/default/rotated/<object>"], "zoom", 25)


    MeshCat.setanimation!(vis, anim)
end
