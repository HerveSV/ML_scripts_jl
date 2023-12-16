using Flux, Images, MLDatasets, Plots

using Flux: crossentropy, onecold, onehotbatch, train!, params

# load data

X_train_raw, y_train_raw = MLDatasets.MNIST(:train)[:]

X_test_raw, y_test_raw = MLDatasets.MNIST(:test)[:]


img = X_train_raw[:, :, 1] #first image in MNIST dataset
@show typeof img
#colorview(Gray, img')


# flatten input data

X_train = Flux.flatten(X_train_raw)
X_test = Flux.flatten(X_test_raw)

# one-hot label encoding

y_train = onehotbatch(y_train_raw, 0:9)
y_test = onehotbatch(y_test_raw, 0:9)


# define model

# fancier way to store the W and b parameters
model = Chain(
    Dense(28 * 28, 32, relu), # first layer where output is first passed through ReLU
    Dense(32, 10), # second layer
    softmax # softmax is the logistic curve, but for more than 2 classes (there are 9 here)
)

# loss function

# similar to the logloss algo, but this works well with multi-class classification
loss(x, y) = crossentropy(model(x), y)

# fetch parameters
ps = params(model) # returns array where each element is a matrix containing parameters, W before b, and sorted by layer


η = 0.01 # learning rate
opt = Adam(η) # gradient descent is expensive to compute

# train model

loss_history = []

epochs = 500

for epoch in 1:epochs
    train!(loss, ps, [(X_train, y_train)], opt)

    train_loss = loss(X_train, y_train)
    push!(loss_history, train_loss)
    println("Epoch: $epoch, Loss: $train_loss")
end

plot(1:epochs, loss_history)