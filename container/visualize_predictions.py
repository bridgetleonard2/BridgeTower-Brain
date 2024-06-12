import numpy as np
from scipy.sparse import load_npz
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from nilearn.datasets import load_mni152_template

import sys


def create_flatmap(subject, layer, modality, prediction_path, top_pred=False):
    """Function to run the vision encoding model. Predicts brain activity
    to story listening and return predictions between predictions and real
    brain activity.

    Parameters
    ----------
    subject: string
        A reference to the subject for analysis. Used to load fmri data.
    layer: int
        A layer reference for the BridgeTower model. Set's the forward
        hook on the relevant layer.
    predictions: array
        Generated by story_prediction() or movie_prediction() function.
        Contains the prediction between predicted and real brain activity
        for each voxel.
    modality: string
        Which modality was used for the base encoding model: vision or
        language.

    Returns
    -------
    Flatmaps:
        Saves flatmap visualizations as pngs
    """
    # Load predictions
    predictions = np.load(prediction_path)

    # if top_pred is True, only plot the top 1% and bottom 1% of
    # predictions, rest become 0
    if top_pred is True:
        upper_pred = np.percentile(predictions, 97)
        bottom_pred = np.percentile(predictions, 3)
        predictions[(predictions > bottom_pred) &
                    (predictions < upper_pred)] = 0

    # Reverse flattening and masking
    fmri_alternateithicatom = np.load("data/fmri_data/storydata/" + subject +
                                      "/alternateithicatom.npy")

    mask = ~np.isnan(fmri_alternateithicatom[0])  # reference for the mask
    # Initialize an empty 3D array with NaNs for the prediction data
    reconstructed_predictions = np.full((31, 100, 100), np.nan)

    # Flatten the mask to get the indices of the non-NaN data points
    valid_indices = np.where(mask.flatten())[0]

    # Assign the prediction coefficients to their original spatial positions
    for index, pred_value in zip(valid_indices, predictions):
        # Convert the 1D index back to 3D index in the spatial dimensions
        z, x, y = np.unravel_index(index, (31, 100, 100))
        reconstructed_predictions[z, x, y] = pred_value

    flattened_predictions = reconstructed_predictions.flatten()

    # Load mappers
    lh_mapping_matrix = load_npz("data/fmri_data/mappers/" + subject +
                                 "_listening_forVL_lh.npz")
    lh_vertex_prediction_data = lh_mapping_matrix.dot(flattened_predictions)
    lh_vertex_coords = np.load("data/fmri_data/mappers/" + subject +
                               "_vertex_coords_lh.npy")

    rh_mapping_matrix = load_npz("data/fmri_data/mappers/" + subject +
                                 "_listening_forVL_rh.npz")
    rh_vertex_prediction_data = rh_mapping_matrix.dot(flattened_predictions)
    rh_vertex_coords = np.load("data/fmri_data/mappers/" + subject +
                               "_vertex_coords_rh.npy")

    vmin, vmax = -np.max(abs(predictions)), np.max(abs(predictions))

    fig, axs = plt.subplots(1, 2, figsize=(7, 4))

    # Plot the first flatmap
    sc1 = axs[0].scatter(lh_vertex_coords[:, 0], lh_vertex_coords[:, 1],
                         c=lh_vertex_prediction_data, cmap='RdBu_r',
                         vmin=vmin, vmax=vmax, s=.01)
    axs[0].set_aspect('equal', adjustable='box')  # Ensure equal scaling
    # axs[0].set_title('Left Hemisphere')
    axs[0].set_frame_on(False)
    axs[0].set_xticks([])  # Remove x-axis ticks
    axs[0].set_yticks([])  # Remove y-axis ticks

    # Plot the second flatmap
    _ = axs[1].scatter(rh_vertex_coords[:, 0], rh_vertex_coords[:, 1],
                       c=rh_vertex_prediction_data, cmap='RdBu_r',
                       vmin=vmin, vmax=vmax, s=.01)
    axs[1].set_aspect('equal', adjustable='box')  # Ensure equal scaling
    # axs[1].set_title('Right Hemisphere')
    axs[1].set_frame_on(False)
    axs[1].set_xticks([])  # Remove x-axis ticks
    axs[1].set_yticks([])  # Remove y-axis ticks

    # Adjust layout to make space for the top colorbar
    plt.subplots_adjust(top=0.85, wspace=0)

    # Add a single horizontal colorbar at the top
    cbar_ax = fig.add_axes([0.25, 0.9, 0.5, 0.03])
    cbar = fig.colorbar(sc1, cax=cbar_ax, orientation='horizontal')

    # Set the color bar to only display min and max values
    cbar.set_ticks([vmin, vmax])
    cbar.set_ticklabels([f'{vmin:.2f}', f'{vmax:.2f}'])

    # Remove the color bar box
    cbar.outline.set_visible(False)
    plt.title(f'{subject}\n{modality} predictions')

    if top_pred is True:
        plt.savefig(f'results/{modality}_check/{subject}/layer' +
                    f'{layer}_visual_top.png', format='png')
    else:
        plt.savefig(f'results/{modality}_check/{subject}/layer' +
                    f'{layer}_visual.png', format='png')
    plt.show()


def transform_to_mni(coords, affine):
    """Transform coordinates to MNI space using the affine matrix."""
    coords_homogeneous = np.c_[coords, np.ones((coords.shape[0], 1))]
    mni_coords = coords_homogeneous.dot(affine.T)[:, :3]
    return mni_coords


def create_3d_mni_plot(subject, layer, prediction_path, top_pred=False):
    """Function to create a 3D volume plot of reconstructed predictions
    in MNI space."""

    # Load predictions
    predictions = np.load(prediction_path)

    # if top_pred is True, only plot the top 1% and bottom 1% of
    # predictions, rest become 0
    if top_pred is True:
        upper_pred = np.percentile(predictions, 97)
        bottom_pred = np.percentile(predictions, 3)
        predictions[(predictions > bottom_pred) &
                    (predictions < upper_pred)] = 0

    # Reverse flattening and masking
    fmri_alternateithicatom = np.load("data/fmri_data/storydata/" + subject +
                                      "/alternateithicatom.npy")

    mask = ~np.isnan(fmri_alternateithicatom[0])  # reference for the mask
    # Initialize an empty 3D array with NaNs for the prediction data
    reconstructed_predictions = np.full((31, 100, 100), np.nan)

    # Flatten the mask to get the indices of the non-NaN data points
    valid_indices = np.where(mask.flatten())[0]

    # Assign the prediction coefficients to their original spatial positions
    for index, corr_value in zip(valid_indices, predictions):
        # Convert the 1D index back to 3D index in the spatial dimensions
        z, x, y = np.unravel_index(index, (31, 100, 100))
        reconstructed_predictions[z, x, y] = corr_value

    # Prepare the data for plotting
    x, y, z = np.indices(reconstructed_predictions.shape)
    x = x.flatten()
    y = y.flatten()
    z = z.flatten()
    values = reconstructed_predictions.flatten()

    # Create a mask for valid (non-NaN) values
    valid_mask = ~np.isnan(values)
    x = x[valid_mask]
    y = y[valid_mask]
    z = z[valid_mask]
    values = values[valid_mask]

    # Stack coordinates
    coords = np.vstack((x, y, z)).T

    # Load the MNI template
    mni_template = load_mni152_template()
    mni_affine = mni_template.affine

    # Transform coordinates to MNI space
    mni_coords = transform_to_mni(coords, mni_affine)

    # Create the 3D scatter plot
    fig = go.Figure(data=go.Scatter3d(
        x=mni_coords[:, 0],
        y=mni_coords[:, 1],
        z=mni_coords[:, 2],
        mode='markers',
        marker=dict(
            size=3,
            color=values,
            colorscale='RdBu_r',
            # cmin=-0.1,
            # cmax=0.1,
            opacity=0.8
        )
    ))

    fig.update_layout(
        title=f"3D MNI Volume Plot for {subject}",
        scene=dict(
            xaxis_title='X',
            yaxis_title='Y',
            zaxis_title='Z'
        ),
        coloraxis_colorbar=dict(
            title="prediction",
            ticks="outside",
            # tickvals=[-0.1, 0.1],
            # ticktext=[-0.1, 0.1]
        )
    )

    fig.show()


if __name__ == "__main__":
    if len(sys.argv) == 5:
        subject = sys.argv[1]
        layer = sys.argv[2]
        modality = sys.argv[3]
        prediction_path = sys.argv[4]
        create_flatmap(subject, layer, modality, prediction_path)
        create_3d_mni_plot(subject, layer, prediction_path)
    else:
        print("Please provide the subject, layer, modality, and prediction \
              path. Usage: python visualize_predictions.py <subject> \
              <layer> <modality> <prediction_path>")
        sys.exit(1)
