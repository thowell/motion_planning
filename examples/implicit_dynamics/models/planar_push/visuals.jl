# function visualize!(vis, model::PlanarPush, q, u;
#         r = 1.0,
#         r_pusher = 0.1,
#         Δt = 0.1,
#         contact_points = false,
#         force_vis = false)
#
# 	default_background!(vis)
#
#     r_box = r - r_pusher
#
#     setobject!(vis["box"], GeometryBasics.Rect(Vec(-1.0 * r_box,
# 		-1.0 * r_box,
# 		-1.0 * r_box),
# 		Vec(2.0 * r_box, 2.0 * r_box, 2.0 * r_box)),
# 		MeshPhongMaterial(color = RGBA(0.0, 0.0, 0.0, 1.0)))
#
#     setobject!(vis["pusher"],
#         Cylinder(Point(0.0, 0.0, 0.0), Point(0.0, 0.0, r_box), r_pusher),
#         MeshPhongMaterial(color = RGBA(1.0, 153.0 / 255.0, 51.0 / 255.0, 1.0)))
#
#     if contact_points
#         for i = 1:4
#             setobject!(vis["contact$i"], GeometryBasics.Sphere(Point3f0(0),
#                 convert(Float32, 0.02)),
#                 MeshPhongMaterial(color = RGBA(1.0, 165.0 / 255.0, 0.0, 1.0)))
#         end
#     end
#
#     if force_vis
# 	       force_vis = ArrowVisualizer(vis[:force])
#     	setobject!(force_vis, MeshPhongMaterial(color=RGBA(1.0, 0.0, 0.0, 1.0)))
#
#         unorm = maximum(abs.(hcat(u...)))
#
#     	us = u[1] / unorm
#
#     	settransform!(force_vis,
#     				Point(q[1][4] - us[1], q[1][5] - us[1], 0.0),
#     				Vec(us[1], us[1], 0.0),
#     				shaft_radius=0.025,
#     				max_head_radius=0.05)
#     end
#
#     anim = MeshCat.Animation(convert(Int, floor(1.0 / Δt)))
#
# 	T = length(q)
#     for t = 1:T-1
#         MeshCat.atframe(anim, t) do
#
#             if force_vis
#     			if t < T-1
#     				if norm(u[t]) < 1.0e-6
#     					setvisible!(vis[:force], false)
#     				else
#     					setvisible!(vis[:force], true)
#
#     					us = u[t] / unorm
#     					settransform!(force_vis,
#     								Point(q[t+1][4] - us[1], q[t+1][5] - us[2], 0.0),
#     								Vec(us[1], us[2], 0.0),
#                                     shaft_radius=0.025,
#                     				max_head_radius=0.05)
#     				end
#     			end
#             end
#
#             settransform!(vis["box"],
# 				compose(Translation(q[t+1][1], q[t+1][2], r), LinearMap(RotZ(q[t+1][3]))))
#
#             settransform!(vis["pusher"], Translation(q[t+1][4], q[t+1][5], r))
#
#             if contact_points
#                 for i = 1:4
#                     settransform!(vis["contact$i"],
#                         Translation(([q[t+1][1:2]; 0.0] + RotZ(q[t+1][3]) * [contact_corner_offset[i]; 0.0])...))
#                 end
#             end
#         end
#     end
#
# 	settransform!(vis["/Cameras/default"],
# 		compose(Translation(0.0, 0.0, 50.0), LinearMap(RotZ(0.5 * pi) * RotY(-pi/2.5))))
# 	setprop!(vis["/Cameras/default/rotated/<object>"], "zoom", 50)
#
#
#     MeshCat.setanimation!(vis, anim)
# end

function _create_planar_push!(vis, model::PlanarPush;
        i = 1,
        r = 1.0,
        r_pusher = 0.1,
        tl = 1.0,
        box_color = RGBA(0.0, 0.0, 0.0, tl),
        pusher_color = RGBA(0.5, 0.5, 0.5, tl))

    r_box = r - r_pusher

    setobject!(vis["box_$i"], GeometryBasics.Rect(Vec(-1.0 * r_box,
		-1.0 * r_box,
		-1.0 * r_box),
		Vec(2.0 * r_box, 2.0 * r_box, 2.0 * r_box)),
		MeshPhongMaterial(color = box_color))

    setobject!(vis["pusher_$i"],
        Cylinder(Point(0.0, 0.0, 0.0), Point(0.0, 0.0, r_box), r_pusher),
        MeshPhongMaterial(color = pusher_color))
end

function _set_planar_push!(vis, model::PlanarPush, q;
    i = 1)
    settransform!(vis["box_$i"],
		compose(Translation(q[1], q[2], 0.01 * i), LinearMap(RotZ(q[3]))))
    settransform!(vis["pusher_$i"], Translation(q[4], q[5], 0.01 * i))
end

function visualize!(vis, model::PlanarPush, q;
        i = 1,
        r = 1.0,
        r_pusher = 0.1,
        tl = 1.0,
        box_color = RGBA(0.0, 0.0, 0.0, tl),
        pusher_color = RGBA(0.5, 0.5, 0.5, tl),
        Δt = 0.1)

	default_background!(vis)

    _create_planar_push!(vis, model,
        i = i,
        r = r,
        r_pusher = r_pusher,
        tl = tl,
        box_color = box_color,
        pusher_color = pusher_color)

    anim = MeshCat.Animation(convert(Int, floor(1.0 / Δt)))

	T = length(q)
    for t = 1:T-1
        MeshCat.atframe(anim, t) do
            _set_planar_push!(vis, model, q[t])
        end
    end

	settransform!(vis["/Cameras/default"],
		compose(Translation(0.0, 0.0, 50.0), LinearMap(RotZ(0.5 * pi) * RotY(-pi/2.5))))
	setprop!(vis["/Cameras/default/rotated/<object>"], "zoom", 50)


    MeshCat.setanimation!(vis, anim)
end
