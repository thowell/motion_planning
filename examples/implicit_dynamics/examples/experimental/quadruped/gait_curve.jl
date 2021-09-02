T = 101
x = range(0, stop = 1.0, length = T)
y = 0.25 * sin.((2.0 * π) .* x)

plot(x, y, ratio = :equal)
