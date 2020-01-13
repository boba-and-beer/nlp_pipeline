"""
Run after preprocessing to create datasets in `datasets` folder.
Training for model
"""
from ..utils import *
import wandb
import logging

name = "".join(random.choices(string.ascii_uppercase + string.digits, k=10))
name = "ans_" + name
notes = "Add more features and start using 0.9 of the data instead of just 0.5"
# MODEL_NAME = 'Adding Features'
# change the name of the run to follow the features
logger = logging.getLogger(__name__)
console_handler = logging.StreamHandler()
model_logger.setLevel(logging.INFO)

wandb.init(name=name, notes=notes, project="ans_googlequestchallenge")
wandb.config.head = 1
wandb.config.start_pct = 0.55
wandb.config.wd = 0.05
wandb.config.opt_func = "ranger"
wandb.config.dropout = 0.05

wandb.config.betas_0 = 0.9
wandb.config.betas_1 = 0.99

wandb.config.seed = random.randint(0, 999)
seed_all(wandb.config.seed)
logger.info(name)

#####################################
### Model Things
#####################################

PRETRAINED_MODEL_NAME = "../notebooks/pretrained_models"
# PRETRAINED_MODEL_NAME = 'bert-base-uncased'
DATASET_DIR = "dataloaders/"

MODEL_NAME = name
MODEL_VAR = Path("..", "models", MODEL_NAME) / "ans"

input_dir = Path("data/raw")
sample_submissions = pd.read_csv(input_dir / "sample_submission.csv")

input_dir = Path("data/processed")
train = pd.read_csv(input_dir / "train_engineered.csv")
test = pd.read_csv(input_dir / "test_engineered.csv")

ALL_LABELS = list(sample_submissions.columns)[1:]
ALL_COLS = ["question_title", "question_body", "answer"]

QUESTION_LABELS = [x for x in ALL_LABELS if x.startswith("question")]
QUESTION_COLS = ["question_title", "question_body"]

ANSWER_COLS = ["answer"]
ANSWER_LABELS = [x for x in ALL_LABELS if x.startswith("answer")]

# ANSWER_FEATURES = [col for col in train.columns if col.startswith('a_')]
# QUESTION_FEATURES = [col for col in train.columns if col.startswith('q_')]

# MODEL_COLS = ALL_COLS + ANSWER_FEATURES + QUESTION_FEATURES

MODEL_LABELS = ALL_LABELS


class QuestDataset(Dataset):
    def __init__(
        self,
        embed_tensors: List[torch.Tensor],
        attention_tensors: List[torch.Tensor],
        label_tensors: torch.FloatTensor = None,
        engineered_features=None,
    ):

        if engineered_features is not None:
            self.x = list(zip(embed_tensors, attention_tensors, engineered_features))
        else:
            self.x = list(zip(embed_tensors, attention_tensors))
        self.y = label_tensors

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]
