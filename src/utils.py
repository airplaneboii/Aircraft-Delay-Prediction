import os
import torch
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, accuracy_score, f1_score

def ensure_dir(
        directory: str
        ) -> None:
    
    if not os.path.exists(directory):
        os.makedirs(directory)


######################
# Evaluation metrics #
######################
def regression_metrics(
        y_true: torch.Tensor,
        y_pred: torch.Tensor
        ) -> dict:

    y_pred_np = y_pred.detach().cpu().numpy()
    y_true_np = y_true.detach().cpu().numpy()

    mse = mean_squared_error(y_true_np, y_pred_np)
    rmse = mse ** 0.5
    mae = mean_absolute_error(y_true_np, y_pred_np)
    r2 = r2_score(y_true_np, y_pred_np)

    return {
        "MSE": mse,
        "RMSE": rmse,
        "MAE": mae,
        "R2": r2
    }

def classification_metrics(
        y_true: torch.Tensor,
        y_pred: torch.Tensor
        ) -> dict:

    y_true_np = y_true.cpu().numpy()
    y_pred_np = y_pred.cpu().numpy()

    accuracy = accuracy_score(y_true_np, y_pred_np)
    f1 = f1_score(y_true_np, y_pred_np, average='weighted')

    return {
        "Accuracy": accuracy,
        "F1_Score": f1
    }

###############################
# To work on GPU if available #
###############################
def move_graph_to_device(graph, device):
    for node_type in graph.node_types:
        graph[node_type].x = graph[node_type].x.to(device)
        if "y" in graph[node_type]:
            graph[node_type].y = graph[node_type].y.to(device)
    for edge_type in graph.edge_types:
        graph[edge_type].edge_index = graph[edge_type].edge_index.to(device)
    return graph