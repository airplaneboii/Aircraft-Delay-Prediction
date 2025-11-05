import torch
from src.utils import regression_metrics, classification_metrics

def test(
        model: torch.nn.Module,
        graph,
        args
        ) -> None:
    model.eval()
    with torch.no_grad():
        out = model(graph["airport"].x, graph["airport", "flies_to", "airport"].edge_index)

        if args.prediction_type == "regression":
            labels = graph["airport"].y.float().squeeze(-1)
            print(regression_metrics(labels, out.squeeze()))
        else:
            labels = graph["airport"].y.long()
            preds = torch.argmax(out, dim=1)
            print(classification_metrics(labels, preds))
