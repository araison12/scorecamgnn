Here is the an example code for using ScoreCAM GNN from the [ScoreCAM GNN : a generalization of an optimal local post-hoc explaining method to any geometric deep learning models](https://arxiv.org/abs/2207.12748) paper

```python
from torch_geometric.datasets import TUDataset

    dataset = TUDataset(root="/tmp/ENZYMES", name="ENZYMES")
    data = dataset[0]
    from scgnn.scgnn import SCGNN

    import torch.nn.functional as F
    from torch_geometric.nn import GCNConv, global_mean_pool


    model = Sequential(
        "data",
        [
            (
                lambda data: (data.x, data.edge_index, data.batch),
                "data -> x, edge_index, batch",
            ),
            (GCNConv(dataset.num_node_features, 64), "x, edge_index -> x"),
            (GCNConv(64, dataset.num_classes), "x, edge_index -> x"),
            (global_mean_pool, "x, batch -> x"),
        ],
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    data = dataset[0].to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    model.eval()
    out = model(data)
    explainer = SCGNN()
    explained = explainer.forward(
        model,
        data.x,
        data.edge_index,
        target=2,
        interest_map_norm=True,
        score_map_norm=True,
    )
```
