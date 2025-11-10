import torch
from torch import nn, optim
from src.utils import regression_metrics, classification_metrics

def train(
        model: nn.Module,
        graph,
        args
    ) -> None:

    # Set model to training mode
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # Define loss function
    if args.prediction_type == "regression":
        criterion = nn.MSELoss()
    else:
        criterion = nn.CrossEntropyLoss()

    # Training loop
    for epoch in range(args.epochs):
        # Always remember to zero the gradients
        optimizer.zero_grad()

        # Forward pass
        out = model(graph.x_dict, graph.edge_index_dict)
        if args.prediction_type == "regression":
            labels = graph["flight"].y.squeeze(-1).to(out.device)
            preds = out.squeeze(-1) 
        else:
            labels = graph["flight"].y.long().to(out.device)
            preds = out 

        # Compute loss
        loss = criterion(preds, labels)
        loss.backward()
        optimizer.step()

        # Compute metrics for monitoring
        if args.prediction_type == "regression":
            metrics_results = regression_metrics(labels, preds)
            metrics_str = f"MSE: {metrics_results['MSE']:.4f}, MAE: {metrics_results['MAE']:.4f}, RMSE: {metrics_results['RMSE']:.4f}"
        else:
            metrics_results = classification_metrics(labels, torch.argmax(out, dim=1))
            metrics_str = f"Accuracy: {metrics_results['Accuracy']:.4f}, F1_Score: {metrics_results['F1_Score']:.4f}"

        print(f"Epoch {epoch+1}/{args.epochs} - {metrics_str}")