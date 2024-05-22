import numpy as np
from scipy.sparse import load_npz
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import sys


def create_flatmap(subject, layer, correlation_path, modality):
    """Function to run the vision encoding model. Predicts brain activity
    to story listening and return correlations between predictions and real
    brain activity.

    Parameters
    ----------
    subject: string
        A reference to the subject for analysis. Used to load fmri data.
    layer: int
        A layer reference for the BridgeTower model. Set's the forward
        hook on the relevant layer.
    correlations: array
        Generated by story_prediction() or movie_prediction() function.
        Contains the correlation between predicted and real brain activity
        for each voxel.
    modality: string
        Which modality was used for the base encoding model: vision or
        language.

    Returns
    -------
    Flatmaps:
        Saves flatmap visualizations as pngs
    """
    # Load correlations
    correlations = np.load(correlation_path)

    # Reverse flattening and masking
    fmri_alternateithicatom = np.load("data/fmri_data/storydata/" + subject +
                                      "/alternateithicatom.npy")

    mask = ~np.isnan(fmri_alternateithicatom[0])  # reference for the mask
    # Initialize an empty 3D array with NaNs for the correlation data
    reconstructed_correlations = np.full((31, 100, 100), np.nan)

    # Flatten the mask to get the indices of the non-NaN data points
    valid_indices = np.where(mask.flatten())[0]

    # Assign the correlation coefficients to their original spatial positions
    for index, corr_value in zip(valid_indices, correlations):
        # Convert the 1D index back to 3D index in the spatial dimensions
        z, x, y = np.unravel_index(index, (31, 100, 100))
        reconstructed_correlations[z, x, y] = corr_value

    flattened_correlations = reconstructed_correlations.flatten()

    # Load mappers
    lh_mapping_matrix = load_npz("data/fmri_data/mappers/" + subject +
                                 "_listening_forVL_lh.npz")
    lh_vertex_correlation_data = lh_mapping_matrix.dot(flattened_correlations)
    lh_vertex_coords = np.load("data/fmri_data/mappers/" + subject +
                               "_vertex_coords_lh.npy")

    rh_mapping_matrix = load_npz("data/fmri_data/mappers/" + subject +
                                 "_listening_forVL_rh.npz")
    rh_vertex_correlation_data = rh_mapping_matrix.dot(flattened_correlations)
    rh_vertex_coords = np.load("data/fmri_data/mappers/" + subject +
                               "_vertex_coords_rh.npy")

    vmin, vmax = -0.1, 0.1
    fig, axs = plt.subplots(1, 2, figsize=(7, 4))

    # Plot the first flatmap
    sc1 = axs[0].scatter(lh_vertex_coords[:, 0], lh_vertex_coords[:, 1],
                         c=lh_vertex_correlation_data, cmap='RdBu_r',
                         vmin=vmin, vmax=vmax, s=.005)
    axs[0].set_aspect('equal', adjustable='box')  # Ensure equal scaling
    # axs[0].set_title('Left Hemisphere')
    axs[0].set_frame_on(False)
    axs[0].set_xticks([])  # Remove x-axis ticks
    axs[0].set_yticks([])  # Remove y-axis ticks

    # Plot the second flatmap
    _ = axs[1].scatter(rh_vertex_coords[:, 0], rh_vertex_coords[:, 1],
                       c=rh_vertex_correlation_data, cmap='RdBu_r',
                       vmin=vmin, vmax=vmax, s=.005)
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
    cbar.set_ticklabels([f'{vmin}', f'{vmax}'])

    # Remove the color bar box
    cbar.outline.set_visible(False)
    if modality == 'vision':
        latex = r"$r_{\mathit{movie \rightarrow story}}"
        plt.title(f'{subject}\n{latex}$')

        plt.savefig('results/movie_to_story/' + subject + '/layer' + layer +
                    '_visual.png', format='png')
    elif modality == 'language':
        latex = r"$r_{\mathit{story \rightarrow movie}}"
        plt.title(f'{subject}\n{latex}$')
        plt.savefig('results/story_to_movie/' + subject + '/layer' + layer +
                    '_visual.png', format='png')
    plt.show()


def create_3d_volume_plot(subject, layer, correlation_path, modality):
    """Function to create a 3D volume plot of reconstructed correlations."""

    # Load correlations
    correlations = np.load(correlation_path)

    # Reverse flattening and masking
    fmri_alternateithicatom = np.load("data/fmri_data/storydata/" + subject +
                                      "/alternateithicatom.npy")

    mask = ~np.isnan(fmri_alternateithicatom[0])  # reference for the mask
    # Initialize an empty 3D array with NaNs for the correlation data
    reconstructed_correlations = np.full((31, 100, 100), np.nan)

    # Flatten the mask to get the indices of the non-NaN data points
    valid_indices = np.where(mask.flatten())[0]

    # Assign the correlation coefficients to their original spatial positions
    for index, corr_value in zip(valid_indices, correlations):
        # Convert the 1D index back to 3D index in the spatial dimensions
        z, x, y = np.unravel_index(index, (31, 100, 100))
        reconstructed_correlations[z, x, y] = corr_value

    # Prepare the data for plotting
    x, y, z = np.indices(reconstructed_correlations.shape)
    x = x.flatten()
    y = y.flatten()
    z = z.flatten()
    values = reconstructed_correlations.flatten()

    # Create a mask for valid (non-NaN) values
    valid_mask = ~np.isnan(values)
    x = x[valid_mask]
    y = y[valid_mask]
    z = z[valid_mask]
    values = values[valid_mask]

    # Create the 3D scatter plot
    fig = go.Figure(data=go.Scatter3d(
        x=x,
        y=y,
        z=z,
        mode='markers',
        marker=dict(
            size=3,
            color=values,
            colorscale='RdBu_r',
            cmin=-0.1,
            cmax=0.1,
            opacity=0.8
        )
    ))

    fig.update_layout(
        title=f"3D Volume Plot for {subject}",
        scene=dict(
            xaxis_title='X',
            yaxis_title='Y',
            zaxis_title='Z'
        ),
        coloraxis_colorbar=dict(
            title="Correlation",
            ticks="outside",
            tickvals=[-0.1, 0.1],
            ticktext=[-0.1, 0.1]
        )
    )

    fig.show()


if __name__ == "__main__":
    if len(sys.argv) == 5:
        subject = sys.argv[1]
        modality = sys.argv[2]
        layer = sys.argv[3]
        correlation_path = sys.argv[4]
        # create_flatmap(subject, layer, correlation_path, modality)
        create_3d_volume_plot(subject, layer, correlation_path, modality)
    else:
        print("Please provide the subject, modality, layer, and correlation \
              path. Usage: python visualize_correlations.py <subject> \
              <modality> <layer> <correlation_path>")
        sys.exit(1)
