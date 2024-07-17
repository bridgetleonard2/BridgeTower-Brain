# imports
import torch
import numpy as np
from torch.nn.functional import pad
from transformers import BridgeTowerModel, BridgeTowerProcessor
from tqdm import tqdm
import utils


# Define Model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = BridgeTowerModel.from_pretrained("BridgeTower/bridgetower-base")
model = model.to(device)

# placeholder for batch features
features = {}


def get_features(name):
    def hook(model, input, output):
        # detached_outputs = [tensor.detach() for tensor in output]
        last_output = output[-1].detach()
        features[name] = last_output  # detached_outputs
    return hook


# register forward hooks with layers of choice
layer_selected = model.cross_modal_image_pooler.register_forward_hook(
    get_features('layer'))

processor = BridgeTowerProcessor.from_pretrained(
    "BridgeTower/bridgetower-base")

data_path = 'data/raw_stimuli/shortclips/stimuli/'

print("loading HDF array")
movie_data = utils.load_hdf5_array(f"{data_path}train_00.hdf",
                                   key='stimuli')


# create overall data structure for average feature vectors
# a dictionary with layer names as keys and a
# list of vectors as it values
data = {}

# a dictionary to store vectors for n consecutive trials
avg_data = {}
n = 30
print("Running movie through model")
# loop through all inputs
for i, image in tqdm(enumerate(movie_data)):

    model_input = processor(image, "", return_tensors="pt")
    # Assuming model_input is a dictionary of tensors
    model_input = {key: value.to(device) for key,
                   value in model_input.items()}

    _ = model(**model_input)

    for name, tensor in features.items():
        if name not in avg_data:
            avg_data[name] = []
        avg_data[name].append(tensor)

    # check if average should be stored
    if (i + 1) % n == 0:
        for name, tensors in avg_data.items():
            first_size = tensors[0].size()
            print(first_size)

            if all(tensor.size() == first_size for tensor in tensors):
                avg_feature = torch.mean(torch.stack(tensors), dim=0)
                avg_feature_numpy = avg_feature.detach().cpu().numpy()
                # print(len(avg_feature_numpy))
            else:
                print("Shapes of tensors in avg_data are not equal")
                # find problem tensors
                for tensor in tensors:
                    if tensor.size() != first_size:
                        print(tensor.size())

            if name not in data:
                data[name] = []
            data[name].append(avg_feature_numpy)

        avg_data = {}

layer_selected.remove()

# Save data
data = np.array(data["layer"])
print("Got movie features")

# Data should be 2d of shape (n_images/n, num_features)
# if data is above 2d, flatten all but first dim
if data.ndim > 2:
    data = np.mean(data, axis=1)
