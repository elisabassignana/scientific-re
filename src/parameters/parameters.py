from dataclasses import dataclass
import torch

@dataclass
class Parameters:

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    seeds = [3828, 3152, 2396]

    # Files
    train = 'sample-data/sample-train.txt'
    train_relations = 'sample-data/sample-train-rel.txt'
    dev = 'sample-data/sample-dev.txt'
    dev_relations = 'sample-data/sample-dev-rel.txt'
    test = 'sample-data/ai-ml.txt'
    test_relations = 'sample-data/ai-ml-rel.txt'

    # Relations
    relations = ['COMPARE','USAGE', 'MODEL-FEATURE', 'PART_WHOLE', 'RESULT']

    # Data
    min_position: int = -10
    max_position: int = 10

    # Preprocessing parameters
    len_sentences: int = 105
    n_pos: int = 18

    # Model parameters
    word_emb_fasttext_size: int = 300
    position_emb_size: int = 50
    bert_emb_size: int = 768
    num_positions = max_position - min_position + 3 # add positions for 0, for <PAD> and for LONG DISTANCE
    kernels = [2, 3, 4]
    out_size: int = 15
    stride: int = 1
    dropout: int = 0.5

    # Training parameters
    epochs: int = 50
    batch_size: int = 12
    learning_rate: float = 0.001