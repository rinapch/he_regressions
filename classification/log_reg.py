import torch
import time 


class LogReg(torch.nn.Module):

    def __init__(self, n_features):
        super(LogReg, self).__init__()
        self.lr = torch.nn.Linear(n_features, 1)

    def forward(self, x):
        out = torch.sigmoid(self.lr(x))
        return out
    
    def accuracy(self, x, y):
        out = self.forward(x)
        correct = torch.abs(y - out) < 0.5
        return correct.float().mean()


def train_log_reg(model, x_train, y_train, x_test, y_test, epochs=5, lr=0.01):
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)

    times = []
    for epoch in range(epochs):
        t_start = time.time()
        optimizer.zero_grad()
        out = model(x_train)
        loss = torch.nn.functional.binary_cross_entropy(out, y_train)
        loss.backward()
        optimizer.step()
        t_end = time.time()
        times.append(t_end - t_start)

        accuracy = model.accuracy(x_test, y_test)
        print(f"Accuracy at epoch #{epoch + 1} is {accuracy}")

    print(f"\nAverage time per epoch: {int(sum(times) / len(times))} seconds")
    print(f"Final accuracy is {accuracy}")
