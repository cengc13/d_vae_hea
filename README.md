## Data-efficient and interpretable inverse materials design using a disentangled variational autoencoder
Cheng Zeng, Zulqarnain Khan, Nathan Post. arXiv preprint, arXiv:2409.06740

To run the notebooks, some relevant Python packages should be installed, including `pyro`, `shap`, `pandas`, `seaborn`, `numpy`, `scipy` and `scikit-learn`.

## Folder structures

- Data
	- HEA_feature_engineered.csv: experimental high-entropy alloy dataset with engineered features.
	- HEA_top30_comps.csv: experimental high-entropy alloy dataset with 30-element composition features.
	- hyper-parameter-tuning.json: hyperparameter tuning results for `scikit-learn` neural network models. The key follows the format **[alpha]-[hls]-[lr]** where alpha, hls and lr refer to L2 regularization strength, hidden layer structure and learning rates, respectively.
	- test_data_reconstruction_analysis.csv: alloy reconstruction analysis results for all 138 test data points.
	- look_up_dict.pkl: look-up dictionary saved in `pickle` format to calculate engineered features for an arbitray alloy.
	- mixing_enthalpy_dict.pkl: mixing enthalpy look-up dictionary in `pickle` format
	- labelled_hea.pk, unlabelled_hea.pk, validation_hea.pk, test_hea.pk: pickle files for different types of data for training the model used for inference.

- models
	- ssvae.model: semi-supervised variational autoencoder model with the highest test accuracy 0.877. This model is used for all inference results shown in the manuscript.


- notebooks
	- 1_SSVAE_Model_training.ipynb: for training the ssvae model.
	- 2_SSVAE_model_inference.ipynb: for predictions and generating most figures present in the manuscript.
	- 3_SSVAE_SHAP_analysis.ipynb: shapley analysis using a kernel-based method. Including the overall and individual feature importance results.
	- 4_SSVAE_Alloy_reconstruction.ipynb: Alloy reconstruction study and summary for 138 test data points.
	- 5_SSVAE_Interpolation_study.ipynb: Interpolation study for the generative model focusing on a small latent region corresponding to refractory alloys.

- utils
	- custom_mlp.py: basic classes for building the SSVAE model.
	- featurization.py: some utility functions to calculate the composition feature vector and engineered feature vectors for a given chemical formula.
