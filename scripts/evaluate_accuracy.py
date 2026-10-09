import os
import torch
import numpy as np
from sklearn.metrics import accuracy_score

from explainability_utils.data_loader import (
    load_preprocessed_data,
    get_train_val_test_split,
    create_data_loaders,
)


def evaluate_model_accuracy(model, data_loader, device: str = 'cpu') -> float:
    model.to(device)
    model.eval()
    all_preds = []
    all_targets = []
    with torch.no_grad():
        for features, masks, targets in data_loader:
            features = features.to(device)
            masks = masks.to(device)
            targets = targets.to(device)

            logits = model(features, masks)
            probs = torch.sigmoid(logits).squeeze(1)
            preds = (probs >= 0.5).long().cpu().numpy()
            all_preds.append(preds)
            all_targets.append(targets.squeeze(1).cpu().numpy())
    y_pred = np.concatenate(all_preds)
    y_true = np.concatenate(all_targets)
    return float(accuracy_score(y_true, y_pred))


def main():
    # Try multiple possible paths for the dataset
    dataset_paths = ['Dataset.csv', 'Capstone/Dataset.csv']
    data_path = None
    for path in dataset_paths:
        if os.path.exists(path):
            data_path = path
            break
    
    if data_path is None:
        raise FileNotFoundError(f"Dataset.csv not found. Tried paths: {', '.join(dataset_paths)}")
    
    data, feature_cols = load_preprocessed_data(data_path)
    train_data, val_data, test_data = get_train_val_test_split(data, train_ratio=0.7, val_ratio=0.15)
    _, _, test_loader = create_data_loaders(
        train_data=train_data,
        val_data=val_data,
        test_data=test_data,
        batch_size=128,
        sequence_length=24,
        prediction_horizon=4,
        features=feature_cols,
    )

    model_paths = {
        'grud': 'outputs/models/grud_demo_model.pt',
        'lstm': 'outputs/models/lstm_demo_model.pt',
        'cnn_lstm': 'outputs/models/cnn_lstm_demo_model.pt',
        'transformer': 'outputs/models/transformer_demo_model.pt',
    }

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    results = {}
    for name, path in model_paths.items():
        if not os.path.exists(path):
            results[name] = None
            continue
        model = torch.load(path, map_location=device, weights_only=False)
        acc = evaluate_model_accuracy(model, test_loader, device=device)
        results[name] = acc
        print(f"{name.upper()} accuracy: {acc:.4f}")

    # Final summary line for easy grepping
    print("ACCURACY_SUMMARY:", {k: (None if v is None else round(v, 4)) for k, v in results.items()})


if __name__ == '__main__':
    main()


