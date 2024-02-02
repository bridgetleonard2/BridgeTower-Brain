# BridgeTower: Investigating Multi-Modal Transformers and Their Predictive Capabilities on Human Brain Activity

## Overview

Welcome to the BridgeTower project! This repository is dedicated to investigating the multi-modal transformer, [BridgeTower](https://github.com/microsoft/BridgeTower), and its predictive capabilities compared to the human brain. Leveraging the powerful [Hugging Face Transformers library](https://huggingface.co/transformers/), we explore the potential of neural networks in capturing and predicting mechanisms of the human brain. This collaborative effort between cognitive psychology and data science aims to deepen our understanding of neural networks and human cognition.

## Motivation

### Cognitive Psychology meets Data Science

As a cognitive psychologist and data scientist, I am deeply intrigued by the prospect of leveraging neural networks to comprehend and predict cognitive processes. The foundation for this exploration is inspired by the groundbreaking work conducted by the Huth Lab at UT Austin, as presented in their paper **(https://arxiv.org/abs/2305.12248)**. In their study, Huth and colleagues demonstrated the remarkable ability of visual and language encoding models to predict fMRI responses to diverse stimuli, highlighting the cross-modality transferability of neural representations within the human brain.

## Replication and Expansion

### Building on Huth Lab's Work

This project begins by replicating the Huth Lab's recent findings, serving as a solid foundation for further investigations. The successful cross-modality transfer showcased in their work sets the stage for an extensive exploration into the capabilities of neural networks in understanding human cognition.

## Key Objectives

- **Replication:** Reimplementing the experiments conducted by the Huth Lab to verify the robustness of their findings.
- **Extension:** Expanding upon the replicated work to delve deeper into the neural representations and their predictive power.
- **Innovation:** Introducing novel approaches and methodologies to push the boundaries of what is currently understood about the relationship between neural networks and human cognitive processes.

## Project Structure

- **data.zip:** Contains the datasets used for training and evaluation.
   - raw_stimuli: Contains repositories from past experiments including `shortclips`: [a dataset created from a movie watching experiment](https://doi.gin.g-node.org/10.12751/g-node.vy1zjd/), `textgrids`: [a dataset created from a story listening experiment](https://www.nature.com/articles/s41597-023-02437-z), and `test_data` for a BridgeTower introduction.
   - fmri_data: Contains the corresponding fMRI data for the shortclips and textgrids stimuli.
- **Models:** Houses the implementation of BridgeTower and other relevant models.
- **Notebooks:** Jupyter notebooks detailing the step-by-step process of replication and subsequent investigations.
- **Results:** Stores the results of experiments, visualizations, and analysis outputs.
- **Container:** Contains a script and an Apptainer definition file to run voxelwise encoding models on an HPC server.

## Getting Started

### Prerequisites

Before diving into the project, ensure you have the following prerequisites installed:

- Python (version 3.8 or later)
- pip (latest version)

This project utilizes the BridgeTower model, which is part of the [Hugging Face Transformers library](https://huggingface.co/transformers/). To install Hugging Face Transformers and other dependencies:

```bash
pip install transformers
pip install -r requirements.txt
```
### Data Sources

Data for model training and testing is available from open-source datasets. There are two repositories for the raw stimuli data and a single repository containing the fMRI data for these stimuli. Links to the datasets and descriptions are found below. Information on where to place them in this repository to make them accessible to the notebooks is found in `docs/data_sources.md`

- [Raw movie stimuli](https://gin.g-node.org/gallantlab/shortclips/src/master/stimuli): *Only the `stimuli` directory is needed for the notebooks.* This dataset contains BOLD fMRI responses in human subjects viewing a set of natural short movie clips. The functional data were collected in five subjects, in three sessions over three separate days for each subject. Details of the experiment are described in the original publication [1].

      [1] Huth, Alexander G., Nishimoto, S., Vu, A. T., & Gallant, J. L. (2012). A continuous semantic space describes the representation of thousands of object and action categories across the human brain. Neuron, 76(6), 1210-1224. https://dx.doi.org/10.1016/j.neuron.2012.10.014
      
      If you publish any work using the dataset, please cite the original publication [1], and cite the dataset [1b] in the following recommended format:
      
      [1b] Huth, A. G., Nishimoto, S., Vu, A. T., Dupre la Tour, T., & Gallant, J. L. (2022). Gallant Lab Natural Short Clips 3T fMRI Data. https://dx.doi.org/10.12751/g-node.vy1zjd

- [Raw story stimuli]((https://openneuro.org/datasets/ds003020/versions/2.2.0): *Only the `derivative/TextGrids` directory is needed for the notebooks.* This dataset now has a dataset descriptor currently available in Scientific Data, that describes all of the data and code available for working with this dataset. It can be found:

   LeBel, A., Wagner, L., Jain, S. et al. A natural language fMRI dataset for voxelwise encoding models. Sci Data 10, 555 (2023). https://doi.org/10.1038/s41597-023-02437-z

- [fMRI data](https://berkeley.app.box.com/s/l95gie5xtv56zocsgugmb7fs12nujpog/folder/142176724641): An fMRI dataset using story and movie stimuli.


1. **Clone the Repository:**
   ```bash
   git clone https://github.com/bridgetleonard2/BridgeTower-Brain.git
   cd BridgeTower

2. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   
3. **Explore Notebooks:**

    Delve into the Jupyter notebooks in the **`Notebooks`** directory to understand the workflow and analyses.
   
4. **Run Experiments:**
   
     Experiment with different parameters, models, and datasets to gain insights into the project's objectives.

## Contributions

Contributions are more than welcome! Whether it's bug fixes, feature enhancements, or new ideas, feel free to submit issues or pull requests.

## Acknowledgments
- Huth Lab at UT Austin for their pioneering work in this domain.
- Open source community for providing tools and frameworks that make projects like this possible.
  
Let the exploration begin! 🚀
