using Plots
using HDF5

f(x) = sqrt(9 - x^2)
g(x) = sqrt(16 - x^2)
function generate_data()
    X = rand(200, 2)
    y = zeros(200)

    for i in 1:100
        X[i, 1] = X[i, 1]*8 - 4
        X[i, 2] = g(X[i, 1])/4# + (rand()-0.5)
        X[i, 1] = X[i, 1]/8 + 0.5
        #X[i, 1] = (X[i, 1] - rand()*2-1)
        y[i] = 1
    end
    for i in 101:200
        X[i, 1] = X[i, 1]*6 - 3
        X[i, 2] = f(X[i, 1])/4#+ (rand()-0.5)
        X[i, 1] = X[i, 1]/8 + 0.5
        #X[i, 1] = (X[i, 1] - rand()*2-1)
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

#=
layers{int}: number of layers of neurons
n0{int}: number of variables
n{array{int, layers}}: number of neurons in each layer
=#
function init(n0 = 2, layers = 2, n = [3, 1])
    W = [zeros(1,1) for i=1:layers]
    b = [zeros(1) for i=1:layers]
    for i = 1:layers
        W[i] = rand(n[i], (i==1) ? n0 : n[i-1])
        b[i] = rand(n[i])
    end
    
    return W, b

end


σ(z) = 1/(1+ℯ^(-z))
# only  works with 2 layers of dimension [3, 1] with 2 initial inputs
function forward_prop(X, W, b)
    Z1 = ((W[1] * X) .+ b[1])
    A1 = σ.(Z1)
    Z2 = ((W[2] * A1) .+ b[2])
    A2 = σ.(Z2)

    return [A1, A2]

end

function cost(A2, y)
    m = size(y)[1]
    ϵ = 1*10^(-15)
    loss = -(1/m) * sum((y .* log.(A2.+ϵ)) .+ ((1 .- y) .* log.(ℯ, 1 .- A2 .+ ϵ)))
    return loss
end

function back_prop(A, W, X, y)
    m = size(y)[1]
    dZ2 = A[2] .- y
    dW2 = (1/m) .* dZ2 * A[1]'
    db2 = (1/m) .* sum(dZ2, dims=2) # we want to sum all columns into one, which is defined as dimension 2

    #=println("dZ2: ", size(dZ2))
    println("dW2: ", size(dW2))
    println("db2: ", size(db2))
    println("X: ", size(X))
    println("A2: ", size(A[2]))=#

    dZ1 = (W[2]' * dZ2) .* (A[1] .* (1 .- A[1])) # note that element-wise matrix mult must be broadcasted
    dW1 = (1/m) .* dZ1 * transpose(X)
    db1 = (1/m) .* sum(dZ1, dims=2)

    #=println("dZ1: ", size(dZ1))
    println("dW1: ", size(dW1))
    println("db1: ", size(db1))
    println("X: ", size(X))
    println("A1: ", size(A[1]))=#

    return [dW1, dW2], [db1, db2]
end

# only  works with 2 layers of dimension [3, 1] with 2 initial inputs
function update(W, b, dW, db, learn_rate)
    #println(size(W[1]), size(dW[1]))
    W1 = W[1] .- learn_rate .* dW[1]
    b1 = b[1] .- learn_rate.*db[1]

    W2 = W[2] .- learn_rate.*dW[2]
    b2 = b[2] .- learn_rate.*db[2]

    return [W1, W2], [b1, b2]
end

function predict(X, W, b)
    A = forward_prop(X, W, b)
    A2 = A[2]
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


function neural_net(X, y, n_iter=100, learn_rate=0.1)
    W, b = init(size(X)[1])

    train_loss = zeros(n_iter)
    train_accuracy = []
    for i in 1:n_iter
        A = forward_prop(X, W, b)
        loss = cost(A[2], y)
        dW, db = back_prop(A, W, X, y)
        W, b = update(W, b, dW, db, learn_rate)

        train_loss[i] = loss

        if (i%10 == 0)
            pred_y = predict(X, W, b)
            push!(train_accuracy, accuracy(pred_y, y))
            #println("Accuracy of ", accuracy(pred_y, y))
        end

    end

    

    pred_y = predict(X, W, b)
    println("Accuracy of ", accuracy(pred_y, y))

    return W, b, train_loss, train_accuracy
end


#X, y = generate_data()
X, y = load_data()

#println(y)


n_iter = 1000
W, b, train_loss, train_accuracy = neural_net(X, y, n_iter, 0.001)
p1 = plot(1:n_iter, train_loss)
p2 = plot(1:length(train_accuracy), train_accuracy)
plot(p1, p2)


#p1 = scatter(X[1, 1:100], X[2, 1:100])
#scatter!(p1, X[1, 101:200], X[2, 101:200])
#=nx = LinRange(0, 1, 100)
ny = LinRange(0, 1, 100)
data = zeros(10000, 2)
for i in 1:100
    for j in 1:100
        data[i*j, 1] = nx[i]
        data[i*j, 2] = ny[j]
    end
end
#nX = repeat(reshape(nx, 1, :), length(ny), 1)
#nY = repeat(ny, 1, length(nx))
#data = [nX; nY]
data = data'
#println(size(data))
#println(size(X))
A2 = forward_prop(data, W, b)[2]'
#println(size(A2))
#println("X1:", size(X[1]))
#println("X2:", size(X[2]))

gfx(x) = x > 0.5
A2 = gfx.(A2)
for i in 1:length(A2)
    
    println(A2[i])
   
end
=#

#gs = rand(10000)
#Z = map(A2, data[1, :], data[2, :])
#data = data'

#=
nX = []
nY = []
nA = []

for i in 1:length(nX)
    if(A[i])
        println("yeet")
        push!(nX, data[i, 1])
        push!(nY, data[i, 2])
        push!(nA, A[i])
    end
end
=#

#println(size(data[:, 1]), size(A2[:, 1]))

#p1 = contour(data[:, 1], data[:, 2], A2[:, 1], fill=true)
#scatter!(p1, X[1, 1:100], X[2, 1:100])
#scatter!(p1, X[1, 101:200], X[2, 101:200])
#println(nX, nY)
#scatter(nX, nY)