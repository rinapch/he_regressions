import torch 
import tenseal as ts
import numpy as np
import time 
 
class EncryptedLinearRegression:
    def __init__(self, n_features):
        """
        Initialize a plain untrained linear regression model.
        Creates zero weight and bias as Python lists (for later encryption).
        """
        # Get number of features from the model shape
        
        # Initialize zero weights and bias
        self.weight = [0.0] * n_features
        self.bias = [0.0]

        # Gradient accumulators
        self._delta_w = 0
        self._delta_b = 0
        self._count = 0

    def forward(self, enc_x):
        """
        Compute the encrypted forward pass:
            enc_out = enc_x.dot(weight) + bias
        """
        return enc_x.dot(self.weight) + self.bias

    def backward(self, enc_x, enc_out, enc_y):
        """
        Compute (and accumulate) gradients:
            error = enc_out - enc_y
            dW += X^T * error
            dB += error
        """
        error = enc_out - enc_y
        self._delta_w += enc_x * error
        self._delta_b += error
        self._count += 1

    def update_parameters(self, lr=0.001, l2_reg=0.01):
        """
        Update parameters using the accumulated gradients.
        lr: Learning rate
        l2_reg: L2 regularization (applied directly to the weight)
        """
        if self._count == 0:
            raise RuntimeError("No forward/backward iterations have been performed.")

        # Average the accumulated gradients
        avg_dW = self._delta_w * (1 / self._count)
        avg_dB = self._delta_b * (1 / self._count)

        # Apply L2 regularization to weights
        # E.g., dW = dW + lambda * W
        if l2_reg > 0.0:
            avg_dW += [w * l2_reg for w in self.weight] if not isinstance(self.weight, ts.CKKSVector) else self.weight * l2_reg

        # Perform the gradient update step
        # If self.weight is encrypted, we do homomorphic updates; 
        # if self.weight is plain, these operations will be normal float ops.
        self.weight -= avg_dW * lr
        self.bias -= avg_dB * lr

        # Reset accumulators
        self._delta_w = 0
        self._delta_b = 0
        self._count = 0

    def plain_mse(self, x_test, y_test):
        """
        Evaluate the model on plain data to compute MSE (mean squared error).
        This is useful for checking accuracy locally (in the clear).
        """
        w = torch.tensor(self.weight)
        b = torch.tensor(self.bias)
        # predictions = XW + b
        preds = np.matmul(x_test, w) + b
        mse = ((preds - y_test) ** 2).mean()
        return mse

    def encrypt(self, context):
        """
        Encrypt the weight and bias using CKKS (TenSEAL).
        After encryption, weight and bias are no longer Python lists but CKKSVectors.
        """
        self.weight = ts.ckks_vector(context, self.weight)
        self.bias = ts.ckks_vector(context, self.bias)

    def decrypt(self):
        """
        Decrypt the weight and bias (if needed).
        """
        self.weight = self.weight.decrypt()
        self.bias = self.bias.decrypt()

    def __call__(self, *args, **kwargs):
        """
        Allows the object to be used like a function:
            model(enc_x)
        """
        return self.forward(*args, **kwargs)
    

def train_encrypted_linear_reg(
    model,
    ctx_training,
    enc_x_train,
    enc_y_train,
    x_test,
    y_test,
    lr=0.001,
    epochs=5,
    batch_size=100
):
    times = []
    n_samples = len(enc_x_train)

    for epoch in range(epochs):
        # Encrypt fresh each epoch if needed 
        # (often you might just keep your model encrypted throughout,
        #  but here's a place to ensure encryption is set)
        model.encrypt(ctx_training)

        # Time the epoch
        t_start = time.time()
        
        # Shuffle data if you like (optional)
        # (requires storing them together or using the same shuffled indices)
        # For simplicity here, we won't shuffle

        # Process each mini-batch
        for i in range(0, n_samples, batch_size):
            x_batch = enc_x_train[i : i + batch_size]
            y_batch = enc_y_train[i : i + batch_size]

            # 1) Accumulate gradients over the batch
            for enc_x, enc_y in zip(x_batch, y_batch):
                enc_out = model.forward(enc_x)
                model.backward(enc_x, enc_out, enc_y)

            # 2) Now update once using the accumulated gradients
            model.update_parameters(lr=lr, l2_reg=0.001)

        # End-of-epoch time
        t_end = time.time()
        epoch_time = t_end - t_start
        times.append(epoch_time)

        # Decrypt temporarily to compute plain MSE (or you could keep it encrypted if you have a different approach)
        model.decrypt()
        current_mse = model.plain_mse(x_test, y_test)
        print(f"[Epoch {epoch + 1}] MSE: {current_mse:.4f} | Time: {epoch_time:.2f} s")

    average_time_per_epoch = sum(times) / len(times)
    print(f"\nAverage time per epoch: {average_time_per_epoch:.2f} seconds")
    final_mse = model.plain_mse(x_test, y_test)
    print(f"Final MSE after training: {final_mse:.4f}")