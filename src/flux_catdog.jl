using Flux, Plots, HDF5, Images

using Flux: crossentropy, train!, params, onehotbatch

function load_data()
    
    file = h5open("/Users/hervesv/Desktop/Stuff/Projects/julia_scripts/ML/src/trainset.hdf5", "r") do file
        X = read(file, "X_train")
        y = read(file, "Y_train")
        #println(size(tempX[:, :, 1]))
        
        return Float32.(X ./ 255), Float32.(y)
        #
        #return X, y
        #println(X[1, :])
    end
    
    

end

X_raw, y_raw = load_data()

y_train = Flux.flatten(onehotbatch(y_raw, 0:1))

#img = float(X_raw[:, :, 1])
#colorview(Gray, img')

X_train = Flux.flatten(X_raw)

model = Chain(
    Dense(64 * 64, 2000, relu), # first layer where output is first passed through ReLU
    Dense(2000, 150, relu),
    Dense(150, 64, relu),
    Dense(64, 32, sigmoid_fast),
    Dense(32, 10, sigmoid_fast),
    Dense(10, 2), # second layer
    softmax # softmax is the logistic curve, but for more than 2 classes (there are 9 here)
)

loss(x, y) = crossentropy(model(x), y)

ps = params(model)

η = 0.01

opt = Adam(η)

loss_history = []

epochs = 500

for e in 1:epochs
    train!(loss, ps, [(X_train, y_train)], opt)

    if e%10 == 0
        train_loss = loss(X_train, y_train)
        push!(loss_history, train_loss)
        println("Epoch: $e, Loss: $train_loss")
    end

end

plot(1:epochs/10, loss_history)


#=
#@show size(X_raw)
img = float(X_raw[:, :, 1]) #first image in MNIST dataset
#@show typeof img
colorview(Gray, img')
#colorview(Gray, img')

#colorview(RGB, img)

X = Flux.flatten(X_raw)

model = Chain(
        Dense(64*64, 64, relu),
        Dense(64, 32, relu),
        Dense(32, 10, relu),
        Dense(10, 1),
        sigmoid_fast
)

loss(x, y) = mse(model(x), y)

ps = params(model)

η = 0.01 # learning rate
opt = Descent(η)#Adam(η) # gradient descent is expensive to compute

# train model

loss_history = []

epochs = 500

for e in 1:epochs
    train!(loss, ps, [(X, y)], opt)

    if e%10 == 0
        train_loss = loss(X, y)
        push!(loss_history, train_loss)
        println("Epoch: $e, Loss: $train_loss")
    end

end

plot(1:epochs/10, loss_history)


=#