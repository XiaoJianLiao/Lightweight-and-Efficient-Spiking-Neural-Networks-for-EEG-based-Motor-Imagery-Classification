# @title Evaluate models and visualize results (Custom confusion matrix, green good, red bad)
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score
import matplotlib # For colormaps access

# Ensure all necessary variables from previous cells are available:
# cnn_model_lenet, snn_model_lenet, cnn_model_lenet_fcl, snn_model_lenet_fcl
# test_data, test_label (from the last train_test_split)
# BATCH_SIZE, TIME_STEPS, device
# data_loader function

# 0. Prepare data loader for evaluation (using test_data from the last split)
eval_test_loader = data_loader(test_data, test_label, batch=BATCH_SIZE, shuffle=False, drop=False)

true_labels_list = []
for _, targets_batch in eval_test_loader:
    true_labels_list.extend(targets_batch.cpu().numpy())
true_labels_np = np.array(true_labels_list)

# --- 1. Get predictions for all models ---
print("Generating predictions for all models...")
all_model_predictions = {}
model_objects = {
    "LENet CNN": cnn_model_lenet,
    "LENet SNN": snn_model_lenet,
    "LENet_FCL CNN": cnn_model_lenet_fcl,
    "LENet_FCL SNN": snn_model_lenet_fcl
}

for model_name, model_obj in model_objects.items():
    print(f"Evaluating {model_name}...")
    model_obj.eval().to(device)
    current_preds = []
    is_snn = "SNN" in model_name

    with torch.no_grad():
        for inputs, _ in eval_test_loader:
            inputs = inputs.to(device)
            if is_snn:
                for m_module in model_obj.modules():
                    if hasattr(m_module, 'reset'):
                        m_module.reset()
                accumulated_outputs = None
                for t in range(TIME_STEPS):
                    outputs_t = model_obj(inputs)
                    if accumulated_outputs is None:
                        accumulated_outputs = outputs_t.clone()
                    else:
                        accumulated_outputs += outputs_t
                _, predicted = accumulated_outputs.max(1)
            else: # ANN
                outputs = model_obj(inputs)
                _, predicted = outputs.max(1)
            current_preds.extend(predicted.cpu().numpy())
    all_model_predictions[model_name] = current_preds
print("All predictions generated.")

# --- 2. Calculate Accuracies for all models ---
accuracy_results = {}
class_names = ['Rest', 'Elbow', 'Hand'] # Corresponds to labels 0, 1, 2

for model_name, predictions in all_model_predictions.items():
    overall_acc = accuracy_score(true_labels_np, predictions)
    accuracy_results[model_name] = {"Overall": overall_acc}
    for class_idx, class_name_key in enumerate(class_names):
        class_indices = np.where(true_labels_np == class_idx)[0]
        if len(class_indices) > 0:
            class_true = true_labels_np[class_indices]
            class_pred = np.array(predictions)[class_indices]
            class_acc = accuracy_score(class_true, class_pred)
            accuracy_results[model_name][class_name_key] = class_acc
        else:
            accuracy_results[model_name][class_name_key] = np.nan

# --- 3. Plot Confusion Matrices and Print Descriptions ---
print("\n--- Confusion Matrices and Descriptions ---")
class_names_display = ['Rest', 'Elbow', 'Hand'] # For display purposes in CM
figure_counter = 1

try:
    cmap_greens = matplotlib.colormaps['Greens']
    cmap_reds = matplotlib.colormaps['Reds']
except AttributeError: # Older matplotlib
    cmap_greens = plt.cm.get_cmap('Greens')
    cmap_reds = plt.cm.get_cmap('Reds')

# Thresholds for color logic
threshold_diagonal_good = 0.5  # Above this is green on diagonal
threshold_off_diagonal_bad = 0.2 # Above this is red on off-diagonal (significant misclassification)

for model_name, predictions in all_model_predictions.items():
    cm = confusion_matrix(true_labels_np, predictions, labels=range(len(class_names_display)))

    # --- Custom Color Logic V2 ---
    row_sums = cm.sum(axis=1, keepdims=True)
    cm_normalized_row = np.zeros_like(cm, dtype=float)
    for r_idx in range(cm.shape[0]):
        if row_sums[r_idx, 0] > 0:
            cm_normalized_row[r_idx, :] = cm[r_idx, :] / row_sums[r_idx, 0]

    num_classes = cm.shape[0]
    color_matrix_rgb = np.zeros((num_classes, num_classes, 3))
    # Store the value used for colormap intensity for text contrast decision
    colormap_input_values = np.zeros((num_classes, num_classes))


    for i in range(num_classes):
        for j in range(num_classes):
            norm_value = cm_normalized_row[i, j]
            color_val_for_cmap = 0.0 # Default for gray or zero row_sum

            if row_sums[i, 0] == 0:
                color_matrix_rgb[i, j, :] = [0.95, 0.95, 0.95] # Light gray
            elif i == j: # Diagonal
                if norm_value > threshold_diagonal_good:
                    color_val_for_cmap = norm_value # Intensity based on how good it is
                    color_matrix_rgb[i, j, :] = cmap_greens(color_val_for_cmap)[:3]
                else: # Bad for diagonal (low correct classification)
                    # Intensity based on how bad it is (1-norm_value)
                    color_val_for_cmap = 1.0 - norm_value
                    color_matrix_rgb[i, j, :] = cmap_reds(color_val_for_cmap)[:3]
            else: # Off-diagonal
                if norm_value > threshold_off_diagonal_bad: # Significant misclassification
                    color_val_for_cmap = norm_value # Intensity based on how bad it is
                    color_matrix_rgb[i, j, :] = cmap_reds(color_val_for_cmap)[:3]
                else: # Low/acceptable misclassification
                    # Intensity based on how good it is (1-norm_value, closer to 1 is better)
                    color_val_for_cmap = 1.0 - norm_value
                    color_matrix_rgb[i, j, :] = cmap_greens(color_val_for_cmap)[:3]
            colormap_input_values[i,j] = color_val_for_cmap
    # --- End Custom Color Logic V2 ---

    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names_display)
    fig, ax = plt.subplots(figsize=(8, 7))
    disp.plot(ax=ax, include_values=True, cmap='Greys', colorbar=False, values_format='d')

    if ax.images:
        ax.images[0].remove()
    ax.imshow(color_matrix_rgb)

    if disp.text_ is not None:
        for i in range(num_classes):
            for j in range(num_classes):
                if disp.text_[i, j] is not None:
                    # Use the stored colormap_input_values for text contrast
                    text_color = "white" if colormap_input_values[i,j] > 0.5 else "black"
                    if row_sums[i,0] == 0: # Ensure black text on gray
                        text_color = "black"
                    disp.text_[i, j].set_color(text_color)
    
    ax.set_title(f'Confusion Matrix - {model_name}\n(Custom Good/Bad Row-Normalized Colors)')
    plt.tight_layout()
    plt.show()

    print(f"\nFigure {figure_counter}: Confusion Matrix for {model_name}.")
    print(f"This matrix visualizes classification performance with custom colors based on row-normalized values:")
    print(f"Diagonal cells (Correct Classifications for true class in row i):")
    print(f"  - Green: >{threshold_diagonal_good*100:.0f}% of true class samples correctly classified. Darker green = higher %.")
    print(f"  - Red:   <={threshold_diagonal_good*100:.0f}% of true class samples correctly classified. Darker red = lower % (i.e., more misclassified).")
    print(f"Off-diagonal cells (Misclassifications for true class in row i into predicted class j):")
    print(f"  - Green: <={threshold_off_diagonal_bad*100:.0f}% of true class samples misclassified here. Darker green = lower % (better).")
    print(f"  - Red:   >{threshold_off_diagonal_bad*100:.0f}% of true class samples misclassified here. Darker red = higher % (worse).")

    overall_acc_val = accuracy_results[model_name]['Overall'] * 100
    print(f"Overall Accuracy: {overall_acc_val:.2f}%.")
    for class_idx, class_name_label in enumerate(class_names_display):
        class_acc_val = accuracy_results[model_name][class_names[class_idx]]
        if not np.isnan(class_acc_val):
            print(f"Accuracy for {class_name_label}: {class_acc_val*100:.2f}%.")
        else:
            print(f"Accuracy for {class_name_label}: N/A.")
    print("-" * 70) # Increased separator width
    figure_counter += 1

# --- 4. Accuracy Table ---
print("\n\n--- Model Performance Summary Table ---")
header = f"| {'Model':<17} | {'Overall Acc.':<15} | {class_names[0]+' Acc.':<12} | {class_names[1]+' Acc.':<12} | {class_names[2]+' Acc.':<12} |"
separator = "|-------------------|-----------------|--------------|--------------|--------------|"
print(header)
print(separator)
for model_name_key in accuracy_results:
    overall_str = f"{accuracy_results[model_name_key]['Overall']*100:.2f}%"
    rest_acc_val = accuracy_results[model_name_key][class_names[0]]
    rest_str = f"{rest_acc_val*100:.2f}%" if not np.isnan(rest_acc_val) else "N/A"
    elbow_acc_val = accuracy_results[model_name_key][class_names[1]]
    elbow_str = f"{elbow_acc_val*100:.2f}%" if not np.isnan(elbow_acc_val) else "N/A"
    hand_acc_val = accuracy_results[model_name_key][class_names[2]]
    hand_str = f"{hand_acc_val*100:.2f}%" if not np.isnan(hand_acc_val) else "N/A"
    row = f"| {model_name_key:<17} | {overall_str:<15} | {rest_str:<12} | {elbow_str:<12} | {hand_str:<12} |"
    print(row)

