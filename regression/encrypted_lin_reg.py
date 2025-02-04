import torch 
import tenseal as ts
import numpy as np
 
 
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

    def update_parameters(self, lr=0.01, l2_reg=0.0):
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