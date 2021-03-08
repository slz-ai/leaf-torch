MODEL_PARAMS = {
    'sent140.bag_dnn': (0.0003, 2),  # lr, num_classes
    'sent140.stacked_lstm': (0.0003, 25, 2, 100),  # lr, seq_len, num_classes, num_hidden
    'sent140.bag_log_reg': (0.0003, 2),  # lr, num_classes
    'femnist.cnn': (0.0003, 62),  # lr, num_classes
    'shakespeare.stacked_lstm': (0.0003, 80, 80, 256),  # lr, seq_len, num_classes, num_hidden
    'celeba.cnn': (0.1, 2),  # lr, num_classes
    'synthetic.log_reg': (0.0003, 5, 60),  # lr, num_classes, input_dim
    'reddit.stacked_lstm': (0.0003, 10, 256, 2),  # lr, seq_len, num_hidden, num_layers
}
"""dict: Model specific parameter specification"""

ACCURACY_KEY = 'accuracy'
BYTES_WRITTEN_KEY = 'bytes_written'
BYTES_READ_KEY = 'bytes_read'
LOCAL_COMPUTATIONS_KEY = 'local_computations'
NUM_ROUND_KEY = 'round_number'
NUM_SAMPLES_KEY = 'num_samples'
CLIENT_ID_KEY = 'client_id'

INPUT_SIZE = (1, 28, 28)