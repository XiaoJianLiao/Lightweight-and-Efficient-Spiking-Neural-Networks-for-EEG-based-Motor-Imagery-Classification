# @title Evaluate models and visualize results
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score

# Ensure all necessary variables from previous cells are available:
# cnn_model_lenet, snn_model_lenet, cnn_model_lenet_fcl, snn_model_lenet_fcl
# test_data, test_label (from the last train_test_split)
# BATCH_SIZE, TIME_STEPS, device
# data_loader function

# 0. Prepare data loader for evaluation (using test_data from the last split)
# This ensures all models are evaluated on the same test set.
eval_test_loader = data_loader(test_data, test_label, batch=BATCH_SIZE, shuffle=False, drop=False)

true_labels_list = []
# Ensure targets are collected correctly from the loader
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

for model_name, predictions in all_model_predictions.items():
    # Create and plot confusion matrix
    cm = confusion_matrix(true_labels_np, predictions, labels=range(len(class_names_display)))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names_display)
    disp.plot()
    plt.title(f'Confusion Matrix - {model_name}')
    plt.show()

    # Print figure description
    print(f"\nFigure {figure_counter}: Confusion Matrix for {model_name}.")
    print(f"This matrix visualizes the classification performance of the {model_name} on the test set.")

    overall_acc_val = accuracy_results[model_name]['Overall'] * 100
    print(f"Overall Accuracy: {overall_acc_val:.2f}%.")

    for class_idx, class_name_label in enumerate(class_names_display):
        # Use class_names for dictionary key, class_names_display for printing label
        class_acc_val = accuracy_results[model_name][class_names[class_idx]]
        if not np.isnan(class_acc_val):
            print(f"Accuracy for {class_name_label}: {class_acc_val*100:.2f}%.")
        else:
            print(f"Accuracy for {class_name_label}: N/A (no samples in test set or error).")
    print("-" * 50)
    figure_counter += 1


# --- 4. Accuracy Table ---
print("\n\n--- Model Performance Summary Table ---")

header = f"| {'Model':<17} | {'Overall Acc.':<15} | {class_names[0]+' Acc.':<12} | {class_names[1]+' Acc.':<12} | {class_names[2]+' Acc.':<12} |"
separator = "|-------------------|-----------------|--------------|--------------|--------------|"
print(header)
print(separator)

for model_name_key in accuracy_results: # Iterate in the order models were processed
    overall_str = f"{accuracy_results[model_name_key]['Overall']*100:.2f}%"
    
    rest_acc_val = accuracy_results[model_name_key][class_names[0]]
    rest_str = f"{rest_acc_val*100:.2f}%" if not np.isnan(rest_acc_val) else "N/A"
    
    elbow_acc_val = accuracy_results[model_name_key][class_names[1]]
    elbow_str = f"{elbow_acc_val*100:.2f}%" if not np.isnan(elbow_acc_val) else "N/A"
    
    hand_acc_val = accuracy_results[model_name_key][class_names[2]]
    hand_str = f"{hand_acc_val*100:.2f}%" if not np.isnan(hand_acc_val) else "N/A"
    
    row = f"| {model_name_key:<17} | {overall_str:<15} | {rest_str:<12} | {elbow_str:<12} | {hand_str:<12} |"
    print(row)

