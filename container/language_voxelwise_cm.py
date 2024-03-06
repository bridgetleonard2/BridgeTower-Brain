import numpy as np
from sklearn.model_selection import check_cv
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn import set_config
from functions import prep_data
from functions import generate_leave_one_run_out
from functions import Delayer
from himalaya.ridge import RidgeCV
from himalaya.backend import set_backend

# CHOOSE SUBJECT
subject = 'S1'

# Load data
print("Load story data")
# Load movie data
# We use all data since this is the cross modal model
fmri_alternateithicatom = np.load("data/storydata/" + subject + "/alternateithicatom.npy")
fmri_avatar = np.load("data/storydata/" + subject + "/avatar.npy")
fmri_howtodraw = np.load("data/storydata/" + subject + "/howtodraw.npy")
fmri_legacy = np.load("data/storydata/" + subject + "/legacy.npy")
fmri_life = np.load("data/storydata/" + subject + "/life.npy")
fmri_myfirstdaywiththeyankees = np.load("data/storydata/" + subject + "/myfirstdaywiththeyankees.npy")
fmri_naked = np.load("data/storydata/" + subject + "/naked.npy")
fmri_odetostepfather = np.load("data/storydata/" + subject + "/odetostepfather.npy")
fmri_souls = np.load("data/storydata/" + subject + "/souls.npy")
fmri_undertheinfluence = np.load("data/storydata/" + subject + "/undertheinfluence.npy")

print("Load story features")
alternateithicatom = np.load("data/feature_vectors/story/alternateithicatom_data.npy")
avatar = np.load("data/feature_vectors/story/avatar_data.npy")
howtodraw = np.load("data/feature_vectors/story/howtodraw_data.npy")
legacy = np.load("data/feature_vectors/story/legacy_data.npy")
life = np.load("data/feature_vectors/story/life_data.npy")
myfirstdaywiththeyankees = np.load("data/feature_vectors/story/myfirstdaywiththeyankees_data.npy")
naked = np.load("data/feature_vectors/story/naked_data.npy")
odetostepfather = np.load("data/feature_vectors/story/odetostepfather_data.npy")
souls = np.load("data/feature_vectors/story/souls_data.npy")
undertheinfluence = np.load("data/feature_vectors/story/undertheinfluence_data.npy")

# Prep data
ai_fmri, ai_features = prep_data(fmri_alternateithicatom, alternateithicatom)
avatar_fmri, avatar_features = prep_data(fmri_avatar, avatar)
howtodraw_fmri, howtodraw_features = prep_data(fmri_howtodraw, howtodraw)
legacy_fmri, legacy_features = prep_data(fmri_legacy, legacy)
life_fmri, life_features = prep_data(fmri_life, life)
yankees_fmri, yankees_features = prep_data(fmri_myfirstdaywiththeyankees, myfirstdaywiththeyankees)
naked_fmri, naked_features = prep_data(fmri_naked, naked)
odetostepfather_fmri, odetostepfather_features = prep_data(fmri_odetostepfather, odetostepfather)
souls_fmri, souls_features = prep_data(fmri_souls, souls)
undertheinfluence_fmri, undertheinfluence_features = prep_data(fmri_undertheinfluence, undertheinfluence)

fmri_arrays = [ai_fmri, avatar_fmri, howtodraw_fmri, legacy_fmri,
               life_fmri, yankees_fmri, naked_fmri,
               odetostepfather_fmri, souls_fmri, undertheinfluence_fmri]
feature_arrays = [ai_features, avatar_features, howtodraw_features,
                  legacy_features, life_features, yankees_features,
                  naked_features, odetostepfather_features,
                  souls_features, undertheinfluence_features]
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

# # Regularize coefficients
# coef /= np.linalg.norm(coef, axis=0)[None]
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

np.save('results/story/' + subject + '_best_alphas.npy', best_alphas)
np.save('results/story/' + subject + '_coefficients.npy', average_coef)
