import time
from datetime import datetime
import torch
from src.utils import regression_metrics, classification_metrics

try:
    import psutil
except Exception:
    psutil = None


def test(
        model: torch.nn.Module,
        graph,
        args
        ) -> None:
    start_dt = datetime.now()
    start_ts = time.time()
    print(f"Test start: {start_dt.isoformat()}")

    model.eval()
    with torch.no_grad():
        out = model(graph["airport"].x, graph["airport", "flies_to", "airport"].edge_index)

        if args.prediction_type == "regression":
            labels = graph["airport"].y.float().squeeze(-1)
            results = regression_metrics(labels, out.squeeze())
            print(results)
        else:
            labels = graph["airport"].y.long()
            preds = torch.argmax(out, dim=1)
            results = classification_metrics(labels, preds)
            print(results)

    end_ts = time.time()
    end_dt = datetime.now()
    print(f"Test end: {end_dt.isoformat()}")
    print(f"Elapsed: {end_ts - start_ts:.2f}s")

    if torch.cuda.is_available():
        try:
            print(f"GPU memory allocated: {torch.cuda.max_memory_allocated()/1024**2:.1f} MB")
        except Exception:
            pass

    if psutil:
        p = psutil.Process()
        print(f"Process mem: {p.memory_info().rss/1024**2:.1f} MB | CPU%: {psutil.cpu_percent(interval=None):.1f}")
