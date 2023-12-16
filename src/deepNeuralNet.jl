using Plots, HDF5

function load_data()
    X = zeros(1000, 64*64)
    y = zeros(1000)
    file = h5open("/Users/hervesv/Desktop/Stuff/Projects/julia_scripts/ML/src/trainset.hdf5", "r") do file
        tempX = read(file, "X_train")
        y = read(file, "Y_train")
        println(size(tempX[:, :, 1]))
        
        # flatten and normalise images
        for i in 1:1000
            for j in 1:64
                for k in 1:64
                    X[i, j*k] = tempX[j, k, i]/255
                end
            end
            

        end
        #return X, y
        #println(X[1, :])
    end
    return X', y
    

end

function generate_data()
    
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

#X, y = load_data()
#println(size(X), size(y))
#@show y

function init(n0::Int, layers::Int, n::Array{Int})
    W = [rand(n[i], (i==1) ? n0 : n[i-1]) for i=1:layers] 
    b = [zeros(n[i], 1) for i=1:layers]

    return W, b
end


#W, b = init(size(X)[1], 4, [3, 4, 2, 1])


function forward_prop(X, W, b)
    σ(z) = 1/(1+ℯ^(-z))

    layers = length(W)
    #println(layers, " layers")
    Z = [zeros(1, 1) for i=1:layers]
    A = [zeros(1, 1) for i=1:layers]

    for i in 1:layers
        if i == 1
            Z[i] = (W[i] * X) .+ b[i]
            A[i] = σ.(Z[i])
            continue
        end

        Z[i] = (W[i] * A[i-1]) .+ b[i]
        A[i] = σ.(Z[i])
    end

    return A
end

#A = forward_prop(X, W, b)
#println(size(A), size(A[1]), size(A[2]), size(A[3]), size(A[4]))

function log_loss(Aₗ, y)
    m = length(y)
    ϵ = 1*10^(-15)
    loss = -(1/m) * sum((y .* log.(Aₗ .+ ϵ)) .+ ((1 .- y) .* log.(ℯ, 1 .- Aₗ .+ ϵ)))
end

function back_prop(A, W, X, y)

    m = length(y)
    layers = length(W)

    # initialising value for the last layer
    dZi = A[layers] .- y

    dW = [zeros(1, 1) for i=1:layers]
    db = [zeros(1, 1) for i=1:layers]
    
    for i in layers:-1:1
        if i == 1
            dW[i] = (1/m) .* dZi * X'
            db[i] = (1/m) .* sum(dZi, dims=2)
            continue
        end

        dW[i] = (1/m) .* dZi * A[i-1]'
        db[i] = (1/m) .* sum(dZi, dims=2)

        # update dZi value for next layer to the left (the (i-1)th layer)
        dZi = W[i]' * dZi .* A[i-1] .* (1 .- A[i-1])

    end

    return dW, db
    
end

#dW, db = back_prop(A, W, X, y)
#@show size(dW), size(db)

function update(W, b, dW, db, learn_rate)

    layers = length(W)

    for i in 1:layers

        W[i] = W[i] .- learn_rate .* dW[i]
        b[i] = b[i] .- learn_rate .* db[i]

    end

    return W, b

end

#W, b = update(W, b, dW, db, 0.01)
#@show size(W), size(b)

function predict(X, W, b)
    layers = length(W)
    Aₗ = forward_prop(X, W, b)[layers]
    return Aₗ .>= 0.5
end

function accuracy(y_pred, y)

    m = size(y)[2]
    #@show m
    correct = 0
    for i in 1:m
        if y_pred[i] == y[i]
            #println(i)
            correct += 1
        end
    end
    #@show correct
    return correct / m

end

#y_pred= predict(X, W, b)
#@show accuracy(y_pred, y)
#@show size(X), size(y)

function neural_net(X, y, layers, n ; n_iter=100, learn_rate=0.1)

    W, b = init(size(X)[1], layers, n)
    
    train_loss = []
    train_accuracy = []

    for i in 1:n_iter
        A = forward_prop(X, W, b)

        if i%10 == 0
            push!(train_loss, log_loss(A[2], y))
            y_pred = predict(X, W, b)
            push!(train_accuracy, accuracy(y_pred, y))
        end

        dW, db = back_prop(A, W, X, y)
        W, b = update(W, b, dW, db, learn_rate)
    end
    println("Done training")
    return W, b, train_loss, train_accuracy
end


X, y = load_data()#generate_data()#

n_iter = 1000
W, b, loss, t_accuracy = neural_net(X, y, 4, [32, 32, 16, 1], n_iter=n_iter, learn_rate=0.01)

p1 = plot(1:n_iter/10, loss)
p2 = plot(1:n_iter/10, t_accuracy)
p3 = plot(X[1, :], X[2, :])
plot(p1, p2, p3)
#p3 = plot(X[1, :], X[2, :])
#@show X[:, 1]
