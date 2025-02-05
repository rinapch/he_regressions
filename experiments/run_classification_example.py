import time

import tenseal as ts
from data_processing import get_hospital_data

from classification import (EncryptedLogReg, LogReg, train_encrypted_log_reg,
                            train_log_reg)


def run_experiment(normalization_type):
    x_train, y_train, x_test, y_test = get_hospital_data(normalization_type)
    print(f"Normalization type: {normalization_type}")
    print("Standard Logistic Regression")
    n_features = x_train.shape[1]
    model = LogReg(n_features)
    train_log_reg(model, x_train, y_train, x_test, y_test)

    print("--------------------------------")

    encrypted_model = EncryptedLogReg(n_features)
    print("Encrypted Logistic Regression")

    poly_mod_degree = 8192
    coeff_mod_bit_sizes = [40, 21, 21, 21, 21, 21, 21, 40]
    # create TenSEALContext
    ctx_training = ts.context(
        ts.SCHEME_TYPE.CKKS, poly_mod_degree, -1, coeff_mod_bit_sizes
    )
    ctx_training.global_scale = 2**21
    ctx_training.generate_galois_keys()

    t_start = time.time()
    enc_x_train = [ts.ckks_vector(ctx_training, x.tolist()) for x in x_train]
    enc_y_train = [ts.ckks_vector(ctx_training, y.tolist()) for y in y_train]
    t_end = time.time()
    print(f"Encryption of the training_set took {int(t_end - t_start)} seconds")

    train_encrypted_log_reg(
        encrypted_model, ctx_training, enc_x_train, enc_y_train, x_test, y_test
    )
    print("--------------------------------")


if __name__ == "__main__":
    for normalization_type in ["min-max"]:
        run_experiment(normalization_type)
