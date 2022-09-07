import numpy as onp
import jax
import jax.numpy as np
from jax.experimental import optimizers, stax
from jax.experimental.stax import Dense, Relu, Sigmoid, Selu, Tanh, Softplus, Identity
import time
import os
import pickle
import glob
from torch.utils.data import Dataset, DataLoader

from src.fem.apps.multi_scale.arguments import args
from src.fem.apps.multi_scale.utils import flat_to_tensor, tensor_to_flat

import matplotlib.pyplot as plt

# Latex style plot
plt.rcParams.update({
    "text.latex.preamble": r"\usepackage{amsmath}",
    "text.usetex": True,
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica"]})


def H_to_C(H_flat):
    H = flat_to_tensor(H_flat)
    F = H + np.eye(3)
    C = F.T @ F
    C_flat = tensor_to_flat(C)
    return C_flat, C


def load_data():
    file_path = f"src/fem/apps/multi_scale/data/numpy/training"
    xy_file = os.path.join(file_path, "data_xy.npy")

    if os.path.isfile(xy_file) and False:
        data_xy = onp.load(xy_file)
    else:
        data_files = glob.glob(f"{file_path}/09052022/*.npy")
        assert len(data_files) > 0, f"No data file found in {file_path}!"
        data_xy = onp.stack([onp.load(f) for f in data_files])
        onp.save(xy_file, data_xy)

    print(f"data_xy.shape = {data_xy.shape}")

    H = data_xy[:, :-1]
    energy_density = data_xy[:, -1:]/(args.L**3)

    return H, energy_density




class EnergyDataset(Dataset):
    def __init__(self, data):
        self.data = data
        
    def __getitem__(self, index):
        return (self.data[index, :-1], self.data[index, -1])

    def __len__(self):
        return len(self.data)


def get_mlp():
    if args.activation == 'selu':
        act_fun = Selu
    elif args.activation == 'tanh':
        act_fun = Tanh
    elif args.activation == 'relu':
        act_fun = Relu
    elif args.activation == 'sigmoid':
        act_fun = Sigmoid
    elif args.activation == 'softplus':
        act_fun = Softplus
    else:
        raise ValueError(f"Invalid activation function {args.activation}.")

    layers_hidden = []
    for _ in range(args.n_hidden):
        layers_hidden.extend([Dense(args.width_hidden), act_fun])

    layers_hidden.append(Dense(1))
    mlp = stax.serial(*layers_hidden)
    return mlp


def shuffle_data(data):
    train_validation_cut = 0.8
    validation_test_cut = 0.9
    n_samps = len(data)
    n_train_validation = int(train_validation_cut * n_samps)
    n_validation_test = int(validation_test_cut * n_samps)
    inds = jax.random.permutation(jax.random.PRNGKey(0), n_samps).reshape(-1)
    inds_train = inds[:n_train_validation]
    inds_validation = inds[n_train_validation:n_validation_test]
    inds_test = inds[n_validation_test:]
    # train_data = data[inds_train]?
    train_data = onp.take(data, inds_train, axis=0)
    validation_data = onp.take(data, inds_validation, axis=0)
    test_data = onp.take(data, inds_test, axis=0)
    train_loader = DataLoader(EnergyDataset(train_data), batch_size=args.batch_size, shuffle=False) # For training, shuffle can be True
    validation_loader = DataLoader(EnergyDataset(validation_data), batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(EnergyDataset(test_data), batch_size=args.batch_size, shuffle=False)
    return train_data, validation_data, test_data, train_loader, validation_loader, test_loader


def min_max_scale(arr1, train_y):
    return (arr1 - np.min(train_y)) / (np.max(train_y) - np.min(train_y))


def evaluate_errors(partial_data, train_data, batch_forward):
    x = partial_data[:, :-1]
    true_vals = partial_data[:, -1]
    train_y = train_data[:, -1]
    preds = batch_forward(x).reshape(-1)
    scaled_true_vals = min_max_scale(true_vals, train_y)
    scaled_preds = min_max_scale(preds, train_y)
    compare = np.stack((scaled_true_vals, scaled_preds)).T
    absolute_error = np.absolute(compare[:, 0] - compare[:, 1])
    percent_error = np.absolute(absolute_error / compare[:, 0])
    scaled_MSE = np.sum((compare[:, 0] - compare[:, 1])**2) / len(compare)

    compare_full = np.hstack((np.stack((true_vals, preds)).T, compare))
    print(compare_full[:10])
    print(f"max percent error is {100*np.max(percent_error):03f}%")
    print(f"median percent error is {100*np.median(percent_error):03f}%")
    print(f"scaled MSE = {scaled_MSE}")

    return scaled_MSE, scaled_true_vals, scaled_preds




def polynomial_hyperelastic():
    H, y_true = load_data()
    C = jax.vmap(H_to_C)(H)[1]

    def I1_fn(C):
        return np.trace(C)

    def I2_fn(C):
        return 0.5*(np.trace(C)**2 - np.trace(C@C))

    def I3_fn(C):
        return np.linalg.det(C)

    def I1_bar_fn(C):
        return I3_fn(C)**(-1./3.) * I1_fn(C)

    def I2_bar_fn(C):
        return I3_fn(C)**(-2./3.) * I2_fn(C)

    def poly_psi(C):
        terms = []
        n = 3
        for i in range(n):
            for j in range(3):
                term = (I2_bar_fn(C) - 3.)**i * (I1_bar_fn(C) - 3.)**j
                terms.append(term)
        m = 3
        for k in range(1, m):
            term =  (np.sqrt(I3_fn(C)) - 3.)**(2*k)
            terms.append(term)

        return terms[1:]

    X = np.stack(jax.vmap(poly_psi)(C)).T

    print(X.shape)

    y_pred = X @ (np.linalg.inv(X.T @ X) @ X.T @ y_true)

    print(np.hstack((y_true, y_pred))[:10])

    # I1 = jax.vmap(I1_fn)(C)
    # plt.plot(I1 - 3., y_true.reshape(-1), color='black', marker='o', markersize=4, linestyle='None')  
    # plt.show()


    ref_vals = np.linspace(0., 40., 100)
    plt.plot(ref_vals, ref_vals, '--', linewidth=2, color='black')
    plt.plot(y_true, y_pred, color='red', marker='o', markersize=4, linestyle='None')  
    plt.axis('equal')
    plt.show()


def get_pickle_path():
    root_pickle = f"src/fem/apps/multi_scale/data/pickle"
    if not os.path.exists(root_pickle):
        os.makedirs(root_pickle)
    pickle_path = os.path.join(root_pickle, 'mlp_weights.pkl')
    return pickle_path


def get_nn_batch_forward():
    pickle_path = get_pickle_path()
    with open(pickle_path, 'rb') as handle:
        params = pickle.load(handle)  
    init_random_params, nn_batch_forward = get_mlp()
    batch_forward = lambda x_new: nn_batch_forward(params, x_new).reshape(-1)
    return batch_forward


def mlp_surrogate(train_data, train_loader, validation_data=None): 
    opt_init, opt_update, get_params = optimizers.adam(step_size=args.lr)
    init_random_params, nn_batch_forward = get_mlp()
    output_shape, params = init_random_params(jax.random.PRNGKey(0), (-1, args.input_size))
    opt_state = opt_init(params)
    
    batch_forward = lambda x_new: nn_batch_forward(params, x_new).reshape(-1)

    def loss_fn(params, x, y):
        preds = nn_batch_forward(params, x)
        y = y[:, None]
        assert preds.shape == y.shape, f"preds.shape = {preds.shape}, while y.shape = {y.shape}"
        return np.sum((preds - y)**2)

    @jax.jit
    def update(params, x, y, opt_state):
        """ Compute the gradient for a batch and update the parameters """
        value, grads = jax.value_and_grad(loss_fn)(params, x, y)
        opt_state = opt_update(0, grads, opt_state)
        return get_params(opt_state), opt_state, value

   
    num_epochs = 20000
    for epoch in range(num_epochs):
        # training_loss = 0.
        # validatin_loss = 0.
        for batch_idx, (x, y) in enumerate(train_loader):
            params, opt_state, loss = update(params, np.array(x), np.array(y), opt_state)
            # training_loss = training_loss + loss

        if epoch % 100 == 0:
            training_smse, _, _ = evaluate_errors(train_data, train_data, batch_forward)
            if validation_data is not None:
                validatin_smse, _, _ = evaluate_errors(validation_data, train_data, batch_forward)
                print(f"Epoch {epoch} training_smse = {training_smse}, Epoch {epoch} validatin_smse = {validatin_smse}")
            else:
                print(f"Epoch {epoch} training_smse = {training_smse}")                    
    
    pickle_path = get_pickle_path()
    with open(pickle_path, 'wb') as handle:
        pickle.dump(params, handle, protocol=pickle.HIGHEST_PROTOCOL)  

    return  batch_forward




def show_yy_plot(validation_data, train_data):
    batch_forward = get_nn_nn_batch_forward()
    evaluate_errors(validation_data, train_data, batch_forward)
    y_pred = batch_forward(validation_data[:, :-1]).reshape(-1)
    y_true = validation_data[:, -1]
    ref_vals = np.linspace(0., 80., 100)
    plt.plot(ref_vals, ref_vals, '--', linewidth=2, color='black')
    plt.plot(y_true, y_pred, color='red', marker='o', markersize=4, linestyle='None')  
    plt.xlabel(f"True Energy", fontsize=20)
    plt.ylabel(f"Predicted Energy", fontsize=20)
    plt.tick_params(labelsize=18)
    plt.axis('equal')
    pdf_root = f"src/fem/apps/multi_scale/data/pdf"
    plt.savefig(os.path.join(pdf_root, 'pred_true.pdf'), bbox_inches='tight')
    plt.show()


def main():
    H, energy_density = load_data()
    data = onp.array(np.hstack((jax.vmap(H_to_C)(H)[0], energy_density))) 
    print(f"data.shape = {data.shape}")
    train_data, validation_data, test_data, train_loader, validation_loader, test_loader = shuffle_data(data) 
    # batch_forward = mlp_surrogate(train_data, train_loader, validation_data)
    show_yy_plot(validation_data, train_data)
 

if __name__ == '__main__':
    main()
    # polynomial_hyperelastic()
