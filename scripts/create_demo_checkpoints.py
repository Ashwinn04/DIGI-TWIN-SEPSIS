import os
import pickle

from models.demo_models import DummyModel


def main():
    output_dir = os.path.join('outputs', 'models')
    os.makedirs(output_dir, exist_ok=True)

    checkpoints = {
        'grud_demo_model.pt': DummyModel('GRU-D'),
        'lstm_demo_model.pt': DummyModel('LSTM'),
        'cnn_lstm_demo_model.pt': DummyModel('CNN-LSTM'),
        'transformer_demo_model.pt': DummyModel('Transformer'),
    }

    for filename, model in checkpoints.items():
        path = os.path.join(output_dir, filename)
        with open(path, 'wb') as f:
            pickle.dump(model, f, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"Wrote demo checkpoint: {path}")


if __name__ == '__main__':
    main()


