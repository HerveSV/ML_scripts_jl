using Plots

#pyplot()

#=
X = LinRange(0, 1, 100)
Y = LinRange(0, 1, 100)
data = zeros(10000, 3)
for i in 1:100
    for j in 1:100
        data[i*j, 1] = X[i]
        data[i*j, 2] = X[j]
        data[i*j, 3] = (X[j] > X[i]) * 5
    end
end

#println(data)

contour(data[:, 1], data[:, 2], data[:, 3])
=#


x = LinRange(0, 1, 100)
y = LinRange(0, 1, 100)

stuff = rand(100, 100)
f(x, y) = begin
        return y > x
    end
X = repeat(reshape(x, 1, :), length(y), 1)
Y = repeat(y, 1, length(x))
Z = map(f, X, Y)

p1 = contour(x, y, f, fill = true)
scatter!(p1, stuff[:, 1], stuff[:, 2])
p2 = contour(x, y, Z)
plot(p1, p2)
