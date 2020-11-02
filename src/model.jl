abstract type Model end

struct TemplateModel <: Model
	n::Int # state dimension
	m::Int # control dimension
	d::Int # disturbance dimension

	# additional model parameters ...
end

function f(model::TemplateModel, x, u, w)
	# continuous-time dynamics
	nothing
end

function k(model::TemplateModel, x)
	# kinematics
	nothing
end

model = TemplateModel(0, 0, 0)

state_output(model, x) = x
state_output_idx(model, idx) = idx
control_output(model, u) = u
