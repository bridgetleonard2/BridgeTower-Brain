import numpy as np
from sklearn.model_selection import check_cv
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn import set_config
from functions import remove_nan
from functions import generate_leave_one_run_out
from functions import Delayer
from himalaya.ridge import RidgeCV
from himalaya.backend import set_backend

# CHOOSE SUBJECT
subject = 'S1'

# Load data
print("Load movie data")
# Load fMRI data
# Using all data for cross-modality encoding model
fmri_train = np.load("data/moviedata/S1/train.npy")
fmri_test = np.load("data/moviedata/S1/test.npy")

print("Load movie features")
# Load in movie feature vectors
train00 = np.load("data/feature_vectors/movie/train_00_data.npy")
train01 = np.load("data/feature_vectors/movie/train_01_data.npy")
train02 = np.load("data/feature_vectors/movie/train_02_data.npy")
train03 = np.load("data/feature_vectors/movie/train_03_data.npy")
train04 = np.load("data/feature_vectors/movie/train_04_data.npy")
train05 = np.load("data/feature_vectors/movie/train_05_data.npy")
train06 = np.load("data/feature_vectors/movie/train_06_data.npy")
train07 = np.load("data/feature_vectors/movie/train_07_data.npy")
train08 = np.load("data/feature_vectors/movie/train_08_data.npy")
train09 = np.load("data/feature_vectors/movie/train_09_data.npy")
train10 = np.load("data/feature_vectors/movie/train_10_data.npy")
train11 = np.load("data/feature_vectors/movie/train_11_data.npy")
test = np.load("data/feature_vectors/movie/test_data.npy")

# Prep data
train_fmri = remove_nan(fmri_train)
test_fmri = remove_nan(fmri_test)

fmri_arrays = [train_fmri, test_fmri]
feature_arrays = [train00, train01, train02, train03, train04,
                  train05, train06, train07, train08, train09,
                  train10, train11, test]

# Combine data
Y_train = np.vstack(fmri_arrays)
X_train = np.vstack(feature_arrays)

# Define cross-validation
run_onsets = []
current_index = 0
for arr in feature_arrays:
    next_index = current_index + arr.shape[0]
    run_onsets.append(current_index)
    current_index = next_index

n_samples_train = X_train.shape[0]
cv = generate_leave_one_run_out(n_samples_train, run_onsets)
cv = check_cv(cv)  # copy the cross-validation splitter into a reusable list

# Define the model
scaler = StandardScaler(with_mean=True, with_std=False)

delayer = Delayer(delays=[1, 2, 3, 4])

backend = set_backend("torch_cuda", on_error="warn")
print(backend)

X_train = X_train.astype("float32")

alphas = np.logspace(1, 20, 20)

ridge_cv = RidgeCV(
    alphas=alphas, cv=cv,
    solver_params=dict(n_targets_batch=500, n_alphas_batch=5,
                       n_targets_batch_refit=100))

pipeline = make_pipeline(
    scaler,
    delayer,
    ridge_cv,
)

set_config(display='diagram')  # requires scikit-learn 0.23
pipeline

_ = pipeline.fit(X_train, Y_train)

# # Calculate scores
# scores = pipeline.score(X_test, Y_test)
# print("(n_voxels,) =", scores.shape)
# scores = backend.to_numpy(scores)

best_alphas = backend.to_numpy(pipeline[-1].best_alphas_)
coef = pipeline[-1].coef_
coef = backend.to_numpy(coef)
print("(n_delays * n_features, n_voxels) =", coef.shape)

# Regularize coefficients
coef /= np.linalg.norm(coef, axis=0)[None]
# coef *= np.sqrt(np.maximum(0, scores))[None]

# split the ridge coefficients per delays
delayer = pipeline.named_steps['delayer']
coef_per_delay = delayer.reshape_by_delays(coef, axis=0)
print("(n_delays, n_features, n_voxels) =", coef_per_delay.shape)
del coef

# average over delays
average_coef = np.mean(coef_per_delay, axis=0)
print("(n_features, n_voxels) =", average_coef.shape)
del coef_per_delay

np.save('results/movie/' + subject + '_best_alphas.npy', best_alphas)
np.save('results/movie/' + subject + '_coefficients.npy', average_coef)
