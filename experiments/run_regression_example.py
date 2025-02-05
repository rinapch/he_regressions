import time

import tenseal as ts
from data_processing import get_financial_data

from regression import (EncryptedLinearRegression, LinReg,
                        train_encrypted_linear_reg, train_linear_reg)



def run_experiment(normalization_type):
    x_train, y_train, x_test, y_test = get_financial_data(
        standardization=normalization_type
    )
    print(f"Normalization type: {normalization_type}")
    print("Standard Linear Regression")
    model = LinReg(x_train.shape[1])
    train_linear_reg(model, x_train, y_train, x_test, y_test)

    print("--------------------------------")
    encrypted_model = EncryptedLinearRegression(x_train.shape[1])
    print("Encrypted Linear Regression")

    poly_mod_degree = 16384
    coeff_mod_bit_sizes = [60, 40, 40, 60]
    # create TenSEALContext
    ctx_training = ts.context(
        ts.SCHEME_TYPE.CKKS, poly_mod_degree, -1, coeff_mod_bit_sizes
    )
    ctx_training.global_scale = 2**8
    ctx_training.generate_galois_keys()
    ctx_training.generate_relin_keys()

    t_start = time.time()
    enc_x_train = [ts.ckks_vector(ctx_training, x.tolist()) for x in x_train]
    enc_y_train = [ts.ckks_vector(ctx_training, y.tolist()) for y in y_train]
    t_end = time.time()
    print(f"Encryption of the training_set took {int(t_end - t_start)} seconds")

    train_encrypted_linear_reg(
        encrypted_model, ctx_training, enc_x_train, enc_y_train, x_test, y_test
    )


if __name__ == "__main__":
    for normalization_type in [None, "z-score", "min-max"]:
        run_experiment(normalization_type)
