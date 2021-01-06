struct Disturbances
	D # covariance matrix
	w # deterministically sample disturbance input
end

function disturbances(D)
	T = length(D)
	d = size(D[1], 1)
	w = []
	for t = 1:T
		_w = Array(sqrt(D[t]))
		tmp = []
		for i = 1:d
			push!(tmp, _w[:, i])
			push!(tmp, -1.0 * _w[:, i])
		end

		push!(w, tmp)
	end

	return Disturbances(D, w)
end
