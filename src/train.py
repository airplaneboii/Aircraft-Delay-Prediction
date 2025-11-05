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
        out = model(graph["airport"].x, graph["airport", "flies_to", "airport"].edge_index)
        if args.prediction_type == "regression":
            labels = graph["airport"].y.float().squeeze(-1)
        else:
            labels = graph["airport"].y.long()

        # Compute loss
        loss = criterion(out, labels)
        loss.backward()
        optimizer.step()

        # Compute metrics for monitoring
        if args.prediction_type == "regression":
            metrics_results = regression_metrics(labels, out.squeeze())
            metrics_str = f"MSE: {metrics_results['MSE']:.4f}, MAE: {metrics_results['MAE']:.4f}, RMSE: {metrics_results['RMSE']:.4f}"
        else:
            metrics_results = classification_metrics(labels, torch.argmax(out, dim=1))
            metrics_str = f"Accuracy: {metrics_results['Accuracy']:.4f}, F1_Score: {metrics_results['F1_Score']:.4f}"

        print(f"Epoch {epoch+1}/{args.epochs} - {metrics_str}")