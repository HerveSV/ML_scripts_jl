using Plots
using HDF5

f(x) = sqrt(9 - x^2)
g(x) = sqrt(16 - x^2)
function generate_data()
    #=
    X = rand(200, 2)
    y = zeros(200)

    for i in 1:100
        X[i, 1] = X[i, 1]*8 - 4
        X[i, 2] = g(X[i, 1])/4 + (rand()/6)
        X[i, 1] = X[i, 1]/8 + 0.5 - (rand()/6)
        #X[i, 1] = (X[i, 1] - rand()*2-1)
        y[i] = 1
    end
    for i in 101:200
        X[i, 1] = X[i, 1]*6 - 3
        X[i, 2] = f(X[i, 1])/4 + (rand()/6)
        X[i, 1] = X[i, 1]/8 + 0.5 - (rand()/6)
        #X[i, 1] = (X[i, 1] - rand()*2-1)
    end
    =#
    #=
    X = 2*rand(200, 2) .- 1
	y = zeros(200)

	for i in 1:100
		X[i, 1] += 3
		X[i, 2] += 6
		y[i] = 1
	end
	for i in 101:200
		X[i, 1] += 4
		X[i, 2] += 4
	end
    =#

    X = zeros(200, 2)
    y = zeros(200)

    for i in 1:100
        θ = rand()*2π
        X[i, 1] = 3*cos(θ)
        X[i, 2] = 3*sin(θ)
        y[i] = 1
    end

    for i in 101:200
        θ = rand()*2π
        X[i, 1] = cos(θ)
        X[i, 2] = sin(θ)
    end

    return X', y' # we transpose both matrixes to swap dimensions, for visual clarity

end

function load_data()
    X = zeros(1000, 64*64)
    y = zeros(1000)
    file = h5open("/Users/hervesv/Desktop/Stuff/Projects/julia_scripts/ML/src/trainset.hdf5", "r") do file
        tempX = read(file, "X_train")
        tempY = read(file, "Y_train")
        println(size(tempX[:, :, 1]))
        
        # flatten and normalise images
        for i in 1:1000
            for j in 1:64
                for k in 1:64
                    X[i, j*k] = tempX[j, k, i]/255
                end
            end
            y[i] = tempY[i]

        end
        #return X, y
        #println(X[1, :])
    end
    return X', y'
    

end

function init(n0, n1, n2)

    W1 = rand(n1, n0)
    b1 = zeros(n1, 1)
    W2 = rand(n2, n1)
    b2 = zeros(n2, 1)

    params = Dict("W1" => W1,
                        "b1" => b1,
                        "W2" => W2,
                        "b2" => b2)
    return params

end

function forward_prop(X, params)

    W1 = params["W1"]
    b1 = params["b1"]
    W2 = params["W2"]
    b2 = params["b2"]

    Z1 = W1 * X .+ b1
    A1 = 1 ./ (1 .+ ℯ.^(.-Z1))

    Z2 = W2 * A1 .+ b2
    A2 = 1 ./ (1 .+ ℯ.^(.-Z2))

    activations = Dict("A1" => A1,
                        "A2" => A2)
    return activations

end

function back_prop(X, y, params, activations)
    
    A1 = activations["A1"]
    A2 = activations["A2"]
    W2 = params["W2"]

    m = size(y)[1]

    dZ2 = A2 .- y
    dW2 = (1/m) .* dZ2 * A1'
    db2 = (1/m) .* sum(dZ2, dims=2)

    dZ1 = W2' * dZ2 .* A1 .* (1 .- A1)
    dW1 = (1/m) .* dZ1 * X'
    db1 = (1/m) .* sum(dZ1, dims=2)

    gradients = Dict("dW1" => dW1,
                        "db1" => db1,
                        "dW2" => dW2,
                        "db2" => db2)

    return gradients


end

function log_loss(A2, y)
    m = size(y)[1]
    ϵ = 1*10^(-15)
    loss = -(1/m) * sum((y .* log.(A2.+ϵ)) .+ ((1 .- y) .* log.(ℯ, 1 .- A2 .+ ϵ)))
end

function update(gradients, params, learn_rate)

    W1 = params["W1"]
    b1 = params["b1"]
    W2 = params["W2"]
    b2 = params["b2"]

    dW1 = gradients["dW1"]
    db1 = gradients["db1"]
    dW2 = gradients["dW2"]
    db2 = gradients["db2"]

    W1 = W1 - learn_rate .* dW1
    b1 = b1 - learn_rate .* db1
    W2 = W2 - learn_rate .* dW2
    b2 = b2 - learn_rate .* db2

    params = Dict("W1" => W1,
                        "b1" => b1,
                        "W2" => W2,
                        "b2" => b2)
    return params

end

function predict(X, params)

    activations = forward_prop(X, params)
    A2 = activations["A2"]
    return A2 .>= 0.5

end

function accuracy(pred, actual)
    correct = 0
    for i in 1:size(pred)[2]
        if pred[i] == actual[i]
            correct += 1
        end
    end
    return correct/size(pred)[2]
end

function neural_network(X, y, n1=3, n_iter=100, learn_rate=0.1)

    n0 = size(X)[1]
    n2 = size(X)[1]
    params = init(n0, n1, n2)

    train_loss = []
    train_acc = []
    #history = []

    for i in 1:n_iter
        activations = forward_prop(X, params)
        A2 = activations["A2"]

        #push!(train_loss, log_loss(A2, y))
        if i%10 == 0
            y_pred = predict(X, params)
            push!(train_acc, accuracy(y_pred, y))
        end

        gradients = back_prop(X, y, params, activations)
        params = update(gradients, params, learn_rate)

    end

    return params, train_loss, train_acc

end

n_iter = 1000

X, y = load_data()#generate_data()
params, train_loss, train_acc = neural_network(X, y, 3, n_iter, 0.01)

#p1 = plot(1:n_iter, train_loss)
p2 = plot(1:n_iter, train_acc)
#p3 = scatter(X[1, 1:100], X[2, 1:100])
#scatter!(p3, X[1, 101:200], X[2, 101:200])
#plot(p1, p2, p3)
#plot(p1, p2)
