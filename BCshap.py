import shap
import numpy as np

from BCgraph import filter_and_predict

if __name__ == '__main__':
    data_path = "./data/training_data.csv"
    model_path = "BCmodel.h5"
    filters = None

    # load the model and dataset
    model, X_test_original, X_test, predicted_output_original, true_output_original = filter_and_predict(
        data_path, model_path, filters
    )

    # subsample the dataset
    sample_size = 2900
    X_test_sample = X_test.sample(n=sample_size, random_state=42)
    X_test_sample_np = X_test_sample.to_numpy()
    X_test_np = X_test.to_numpy()
    
    print(X_test.columns)
    
    columns = ['Hour_Sin', 'Hour_Cos', 'Rain', 'Wind speed  [m/s]',
       'Air temperature [°C]', 'Relative humidity [%]',
       'Pressure at station level [hPa]', 'SunDirectionConv',
       'Facade_Orientation_N', 'Facade_Orientation_S', 'Facade_Orientation_E',
       'Facade_Orientation_W', 'Ground_floor', 'Upper_floor', 'Area_green',
       'Area_concrete', 'Insulation_insulated', 'Insulation_uninsulated']

    explainer = shap.GradientExplainer(model, X_test_sample_np)
    shap_values = explainer.shap_values(X_test_sample_np)
    print(np.array(shap_values).shape)
    shap_values = np.squeeze(shap_values)
    print(shap_values.shape)
    shap.summary_plot(shap_values, X_test_sample_np, feature_names=columns)