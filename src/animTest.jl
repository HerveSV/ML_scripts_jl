using Plots

@userplot CirclePlot
@recipe function f(cp::CirclePlot)
    x, y, i = cp.args
    n = length(x)
    inds = circshift(1:n, 1 - i)
    linewidth --> range(0, 10, length = n)
    seriesalpha --> range(0, 1, length = n)
    aspect_ratio --> 1
    label --> false
    x[inds], y[inds]
end

n = 150

i = range(0, 2π, length = n)
x = sin.(t)
y = cos.(t)

X = 2*rand(100, 2) .- 1
for i in 1:50
    X[i, 1] += 3
    X[i, 2] += 6
    y[i] = 1
end
for i in 51:100
    X[i, 1] += 4
    X[i, 2] += 4
end
w₁ = -4.274041596079461
w₂ = 3.246711446845003
b = -0.9403400983130439
x₂ = -(X[:, 1].*W[1] .+ b) ./ W[2]

anim = @animate for i ∈ 1:n
    circleplot(x, y, i)
end
gif(anim, "anim_fps15.gif", fps = 15)