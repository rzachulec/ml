import shap
import numpy as np
import os
import glob
import gc

from BCgraph import filter_and_predict

if __name__ == '__main__':
    data_path = "./data/training_data.csv"
    model_path = "BCmodel.h5"
    filters = None
    # filters = {"Date": "2024-07-19"}

    output_dir = "./shap_chunks"
    os.makedirs(output_dir, exist_ok=True)  # Create directory to store chunks

    # Load the model and dataset
    model, X_test_original, X_test, predicted_output_original, true_output_original = filter_and_predict(
        data_path, model_path, filters
    )

    # Subsample the dataset
    sample_size = 900
    X_test_sample = X_test.sample(n=sample_size, random_state=42)
    X_test_sample_np = X_test_sample.to_numpy()

    # Initialize SHAP explainer
    masker = shap.maskers.Independent(X_test_sample_np)
    explainer = shap.PermutationExplainer(model, masker=masker, feature_names=X_test.columns)

    # Process the subsampled dataset in chunks
    chunk_size = 300
    for i in range(0, len(X_test_sample), chunk_size):
        X_chunk = X_test_sample.iloc[i:i + chunk_size]
        X_chunk_np = X_chunk.to_numpy()

        # Calculate SHAP values for the current chunk
        shap_values_chunk = explainer(X_chunk_np).values
        np.save(os.path.join(output_dir, f"shap_chunk_{i // chunk_size}.npy"), shap_values_chunk)

        # Free memory
        del X_chunk, X_chunk_np, shap_values_chunk
        gc.collect()

    # Load and combine SHAP values from disk
    shap_values_list = [
        np.load(file_path) for file_path in sorted(glob.glob(os.path.join(output_dir, "shap_chunk_*.npy")))
    ]
    combined_shap_values = np.vstack(shap_values_list)

    # Generate the summary plot
    shap.summary_plot(combined_shap_values, X_test_sample, feature_names=X_test_sample.columns)
