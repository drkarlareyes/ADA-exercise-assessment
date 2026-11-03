# Exercise assessment for squats and Romanian deadlifts (ADA)

Analysis code from the ADA study (Anthropometry and Diagnostic Aware deep learning for exercise assessment). ADA classifies squat and Romanian deadlift (RDL) repetitions as correct or incorrect from full-body Xsens IMU kinematics, and adds each person's anthropometry and diagnostic profile to the model.

## In this repository

| File | Content |
|---|---|
| `exploratory_analysis.ipynb` | Exploratory analyses on the labelled kinematics (see below). |
| `merged_data_rdl_id.csv` | RDL data, one row per frame: acceleration, orientation (quaternion) and angular velocity from six Xsens sensors, a label (0 pause, 1 correct, 2 incorrect) and a subject ID. |
| `confusion_matrix.pdf` | Confusion matrix of the CNN-LSTM window classifier. |

The notebook covers two questions:

1. **Do correct and incorrect executions separate without labels?** KMeans, DBSCAN and Isolation Forest on the standardised features, t-SNE, and an autoencoder latent space clustered with KMeans and HDBSCAN. Agreement with the true labels stayed low (adjusted Rand index 0.01 to 0.12), so movement quality is not recoverable from the raw feature structure alone. This motivated the supervised, subject-aware ADA model.
2. **How well does a kinematics-only CNN-LSTM classify windows?** 200-frame windows (pause, correct, incorrect), two 1D convolution layers and an LSTM, plus a comparison of single sensors. It reached 97% accuracy on a random 20% split of windows. Windows from the same recordings fall into both training and test sets, so this figure is optimistic; the paper uses subject-wise cross-validation.

The notebook also reads `merged_data_squat.csv` and `merged_data_rdl.csv`, which can be rebuilt from the raw recordings.

## The ADA model (paper)

- **Inputs.** Joint-angle sequences from 17 Xsens MVN sensors (60 Hz), plus static anthropometric measures (such as height and limb lengths) and diagnostic features (such as mobility tests and previous lower-limb surgery).
- **Architecture.** A CNN-LSTM branch for the kinematic sequences and a fully connected branch for the static features, fused with feature-level attention, with two outputs: movement quality and movement-related risk.
- **Training.** Pretraining on the whole group, then fine-tuning for each person.
- **Validation.** Subject-wise cross-validation, so no person appears in both training and test data.
- **Results.** Movement quality: 94.8% accuracy (F1 0.949, AUC 0.98) against 86.5% for kinematics alone. Personal fine-tuning added 3.3 to 4.5 points. Risk prediction: 97.8% accuracy (AUC 0.99).
- **Participants.** 15 healthy adults and about 1,500 labelled repetitions.

## Data

The raw Xsens recordings of all participants are in [Xsens-IMU-Raw-Data-Squat-RDL](https://github.com/drkarlareyes/Xsens-IMU-Raw-Data-Squat-RDL) (CC BY 4.0).

## Requirements

Python 3 with NumPy, pandas, scikit-learn, matplotlib, seaborn, PyTorch, TensorFlow/Keras and hdbscan.

## Citation

Reyes Leiva KM, Nikelova P, Cerny M. Anthropometry and diagnostic aware deep learning for exercise assessment. *Frontiers in Medical Technology* 7:1725661 (2026). https://doi.org/10.3389/fmedt.2025.1725661

## Author

Karla Miriam Reyes Leiva, PhD. [Website](https://drkarlareyes.github.io) · ORCID [0000-0002-3269-0830](https://orcid.org/0000-0002-3269-0830)

Code: MIT licence. Data: CC BY 4.0.
