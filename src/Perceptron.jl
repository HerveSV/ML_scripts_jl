using Plots
using HDF5

# initialise and return set of weights and bias variable
function init(X)
    W = rand(size(X)[2])
    b = rand()
    return W, b
end

σ(z) = 1/(1+ℯ^(-z))
	
function model(X, W, b)
    #println(size(X), size(W))
    Z = ((X * W) .+ b)
    A = σ.(Z) # broadcast all Z elements through sigmoid activation function
    return A
end

# returns log loss
function cost(A, y)
    m = size(y)[1]
    ϵ = 1*10^(-15)
    loss = -(1/m) * sum((y .* log.(A.+ϵ)) .+ ((1 .- y) .* log.(ℯ, 1 .- A .+ ϵ)))
    return loss
end

# returns gradient of log loss with respect to weights and bias
function gradients(A, X, y)
    m = size(y)[1]
    dW = (1/m) .* transpose(X)*(A .- y)
    db = (1/m) * sum(A .- y)
    return dW, db
end

function update(W, b, dW, db, α)
    W = W .- α.*dW
    b = b .- α.*db
    return W, b
end

function predict(X, W, b)
    A = model(X, W, b)
    return A .>= 0.5
end

function accuracy(pred, actual)
    correct = 0
    for i in 1:size(pred)[1]
        if pred[i] == actual[i]
            correct += 1
        end
    end
    return correct/size(pred)[1]
end

function perceptron(X, y, α)
    W, b = init(X)
    
    
    
    
    
    cycles = 1000
    all_loss = zeros(cycles)

    
    
    for i in 1:cycles
        A = model(X, W, b)
        loss = cost(A, y)
        dW, db = gradients(A, X, y)
        W, b = update(W, b, dW, db, α)
        
        all_loss[i] = loss
        
        #=
        scatter(X[1:50, 1], X[1:50, 2], xlim = (1.5, 5.5), ylim = (2.5, 7.5))
        scatter!(X[51:100, 1], X[51:100, 2])
        x₂ = -(X[:, 1].*W[1] .+ b) ./ W[2]
        plot!(X[:, 1], x₂, color="purple")
        =#
        
        
        
        
    end

    plot(1:cycles, all_loss)
    pred_y = predict(X, W, b)
    println("Accuracy of ", accuracy(pred_y, y))
    return W, b
end

function loaddata()
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
    return X, y
    

end

function loadtest(path, m)
    X = zeros(m, 64*64)
    y = zeros(m)
    file = 	h5open(path, "r") do file
        tempX = read(file, "X_test")
        y = read(file, "Y_test")
        println(size(tempX[:, :, 1]))
        
        # flatten and normalise images
        for i in 1:m
            for j in 1:64
                for k in 1:64
                    X[i, j*k] = tempX[j, k, i]/255
                end
            end

        end
        #return X, y
        #println(X[1, :])
    end
    return X, y


end


σ(z) = 1/(1+ℯ^(-z))

function main()
    #=
    X = 2*rand(100, 2) .- 1
	y = zeros(100)

	for i in 1:50
		X[i, 1] += 3
		X[i, 2] += 6
		y[i] = 1
	end
	for i in 51:100
		X[i, 1] += 4
		X[i, 2] += 4
	end

	#println(X)
	=#
    X, y = loaddata()
    Xtest, ytest = loadtest("/Users/hervesv/Desktop/Stuff/Projects/julia_scripts/ML/src/testset.hdf5", 200) 

    # Start

    W, b = perceptron(X, y, 0.2)
    #println(W, b)
	#p1 = [3.1 5;]
	#predict(p1, W, b)

    #=
    scatter(X[1:50, 1], X[1:50, 2])
    scatter!(X[51:100, 1], X[51:100, 2])
    x₂ = -(X[:, 1].*W[1] .+ b) ./ W[2]
    plot!(X[:, 1], x₂, color="purple")
    =#
    


	

end


main()



#loaddata()