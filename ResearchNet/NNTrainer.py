class NNTrainer:

    def __init__(self, net, optimizer, criterion):
        self.net = net
        self.optim = optimizer
        self.criterion = criterion


    def train_for_iterations(self, data, iterations=100, batch_size=100):
        images = data['images']
        labels = data['labels']
        for iteration in range(iterations):
            loss_acum = 0
            for i in range(len(images)//batch_size+1):
                local_images = images[i*batch_size:(i+1)*batch_size]
                local_labels = labels[i*batch_size:(i+1)*batch_size]
                if len(local_images) < 1:
                    print('Attempted to use images from {} to {}'.format(i*batch_size, (i+1)*batch_size))
                    print('But images end at {}'.format(len(images)))
                    continue
                self.optim.zero_grad()
                output = self.net(local_images)
                loss = self.criterion(output, local_labels)
                loss_acum += loss.item()
                loss.backward()
                self.optim.step()
            print('Finished iteraion {}'.format(iteration))
            print('Had loss of {}'.format(loss_acum/(i+1)))



