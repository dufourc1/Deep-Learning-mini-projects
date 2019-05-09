# RUN :
input_train, target_train, classes_train, input_test, target_test, classes_test = generate_pair_sets(1000)

lr, nb_epochs, batch_size = 10e-3, 25, 250

#model = ResNet(nb_channels=50, kernel_size=5, nb_blocks=7)
model = Net2(nodes_in=2*14**2, nodes_hidden=1000, nodes_out=2, n_hidden=2)
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(),lr)

device = 'cpu'
model = model.to(device)
criterion = criterion.to(device)
input_train, target_train = input_train.to(device), target_train.to(device)
input_test, target_test = input_test.to(device), target_test.to(device)

for e in range(nb_epochs):
    for input, targets in zip(input_train.split(batch_size), target_train.split(batch_size)):
        optimizer.zero_grad()
        output = model(input)
        loss = criterion(output, targets)
        loss.backward()
        optimizer.step()
    with torch.no_grad():
        the_loss = criterion(model(input_train),target_train).item()
        print('Iteration {:>3}:\t'.format(e), 'Loss on train:\t', the_loss, end='\r')

with torch.no_grad():
    train_loss = criterion(model(input_train),target_train).item()
    test_loss  = criterion(model(input_test),target_test).item()
    train_accuracy = (model(input_train).argmax(dim=1) == target_train).sum().item() / len(target_train)
    test_accuracy  = (model(input_test).argmax(dim=1) == target_test).sum().item() / len(target_test)
    print('\n Loss on TRAIN :\t', train_loss,
          '\n Loss on TEST :\t', test_loss,
          '\n Accuracy on TRAIN :\t', train_accuracy,
          '\n Accuracy on TEST :\t', test_accuracy)
    
