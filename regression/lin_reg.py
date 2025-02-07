import time

import torch


class LinReg(torch.nn.Module):

    def __init__(self, n_features):
        super(LinReg, self).__init__()
        self.lr = torch.nn.Linear(n_features, 1)

    def forward(self, x):
        out = self.lr(x)
        return out


def train_linear_reg(model, x_train, y_train, x_test, y_test, lr=0.00001, epochs=5):
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    loss_fn = torch.nn.MSELoss()
    times = []
    for epoch in range(epochs):
        t_start = time.time()
        optimizer.zero_grad()
        preds = model(x_train)
        loss = loss_fn(preds, y_train)
        loss.backward()
        optimizer.step()
        t_end = time.time()
        times.append(t_end - t_start)
        # print(x_train)

        # print("PREDICTED")
        # print(model(x_test))
        # print("TRUE")
        # print(y_test)
        print(f"MSE at epoch #{epoch + 1} is {loss_fn(model(x_test), y_test).item()}")

    print(f"\nAverage time per epoch: {int(sum(times) / len(times))} seconds")
    print(f"Final MSE is {loss_fn(model(x_test), y_test).item()}")
