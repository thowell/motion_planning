struct Disturbances
	W
	w
end

function disturbances(W)
	T = length(W)
	d = size(W[1], 1)
	w = []
	for t = 1:T
		_w = sqrt(W[t])
		tmp = []
		for i = 1:d
			push!(tmp, _w[:, i])
			push!(tmp, -1.0 * _w[:, i])
		end

		push!(w, tmp)
	end

	return Disturbances(W, w)
end
