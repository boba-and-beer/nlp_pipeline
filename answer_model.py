"""
Here we load in the models
"""
from logger import *
from utils import *
from nlp_pipeline.models.pooler import *
from nlp_pipeline.models.bert import *
from nlp_pipeline.models.transformers import *
from nlp_pipeline.metrics.ranks import *
from generate import *

model_logger = setup_logger()

from abc import ABC, abstractmethod
from transformers import BertPreTrainedModel, BertModel
from fastai.text import *
from nlp_pipeline.models.pooler import *

# Setting up notes for wandb
notes = "Add Layernorm after dropout layer"
# change the name of the run to follow the features
name = "".join(random.choices(string.ascii_uppercase + string.digits, k=10))
name = "ans_not_pretrained_" + name
wandb.init(name=name, notes=notes, project="ans_googlequestchallenge")

## Features i do not want to change
wandb.config.token_type = "Head Only"
wandb.config.start_pct = 0.55
wandb.config.wd = 0.05
# opt func is either ranger of flatten anneal
wandb.config.opt_func = "adam"
wandb.config.dropout = 0.05
wandb.config.betas_0 = 0.9
wandb.config.betas_1 = 0.99
wandb.config.hidden_layer_pool = -1
wandb.config.seed = random.randint(0, 999)

bs = 8

seed_all(wandb.config.seed)
model_logger.info(" The name of the model is %s", name)
model_logger.info("The random seed is %d", wandb.config.seed)

#####################################
### Loading Models
#####################################

PRETRAINED_MODEL_NAME = 'bert-base-uncased'

DATASET_DIR = "googlequestchallenge/torch_datasets/"

MODEL_NAME = name
OUTPUT_MODEL_DIR = Path("..", "models", "ans_models", MODEL_NAME)

create_dir(OUTPUT_MODEL_DIR)

input_dir = Path("googlequestchallenge/data/raw")
sample_submissions = pd.read_csv(input_dir / "sample_submission.csv")

engineered_input_dir = Path("googlequestchallenge/data")

train = pd.read_csv(engineered_input_dir / "train_engineered.csv")
test = pd.read_csv(engineered_input_dir / "test_engineered.csv")


#####################################
### Loading the datasets
#####################################

# Saving the datasets
train_ans_dataset = torch.load(DATASET_DIR + "train_ans_ds_" + str(wandb.config.fold) + "_" + feature_set)
valid_ans_dataset = torch.load(DATASET_DIR + "valid_ans_ds_" + str(wandb.config.fold) + "_" + feature_set)
test_ans_dataset = torch.load(DATASET_DIR + "test_ans_ds_" + str(wandb.config.fold) + "_" + feature_set)

# Load in the custom tokenizer still
pretrain_model.config.num_labels = len(ANSWER_LABELS)
pretrain_model.config.output_hidden_states = True

# Defining the right sampler
trn_ans_samp = SortishSampler(
    train_ans_dataset, key=lambda x: len(train_ans_dataset.x), bs=bs
)
valid_ans_samp = SortishSampler(
    valid_ans_dataset, key=lambda x: len(valid_ans_dataset.x), bs=bs
)
test_ans_samp = SortishSampler(
    valid_ans_dataset, key=lambda x: len(test_ans_dataset.x), bs=bs
)

dl_kwargs = {
    "batch_size": bs,
    "shuffle": False,
    "batch_sampler": None,
    "num_workers": 0,
    "pin_memory": True,
}

train_ans_dl = DataLoader(train_ans_dataset, sampler=trn_ans_samp, **dl_kwargs)
valid_ans_dl = DataLoader(valid_ans_dataset, sampler=valid_ans_samp, **dl_kwargs)
test_ans_dl = DataLoader(test_ans_dataset, sampler=test_ans_samp, **dl_kwargs)

#####################################
### Model
#####################################

model_class = BertSequenceClassification

transformer_model = model_class.from_pretrained(
    PRETRAINED_MODEL_NAME, config=pretrain_model.config, dropout_rate=wandb.config.dropout,
    hidden_layer_output=wandb.config.hidden_layer_output
)

custom_transformer_model = CustomTransformerModel(transformer_model)

# Save all models here - to upload to python 
ans_databunch = TextDataBunch(
    train_dl=train_ans_dl, valid_dl=valid_ans_dl, test_dl=test_ans_dl, device="cuda:0"
)

if wandb.config.opt_func == "ranger":
    opt_func = partial(
        Ranger, betas=(wandb.config.betas_0, wandb.config.betas_1), eps=wandb.config.wd
    )

if wandb.config.opt_func == "adam":
    from transformers import AdamW
    opt_func = partial(
        AdamW,
        correct_bias=False,
        betas=(wandb.config.betas_0, wandb.config.betas_1),
        weight_decay=wandb.config.wd,
    )


learner = BertLearner(
    ans_databunch,
    custom_transformer_model,
    opt_func=opt_func,
    bn_wd=False,
    true_wd=True,
    metrics=[SpearmanRho()],
    callback_fns=partial(
        WandbCallback,
        save_model=True,
        mode="max",
        monitor="spearman_rho",
        seed=wandb.config.seed,
    ),
).to_fp16()

learner.freeze()
learner.lr_find()
lr = learner.recorder.lrs[np.array(learner.recorder.losses).argmin()] / 10

model_logger.info("We chose to use lR: %d", lr)
lr_array = np.array(
    [
        lr / (2.6 ** 8),
        lr / (2.6 ** 7),
        lr / (2.6 ** 6),
        lr / (2.6 ** 5),
        lr / (2.6 ** 4),
        lr / (2.6 ** 3),
        lr / (2.6 ** 2),
        lr / (2.6 ** 1),
        lr,
    ]
)



def flattenAnneal(learn: Learner, lr: float, n_epochs: int, start_pct: float):
    n = len(learn.data.train_dl)
    anneal_start = int(n * n_epochs * start_pct)
    anneal_end = int(n * n_epochs) - anneal_start
    lr_array = np.array(
        [
            lr / (2.6 ** 8),
            lr / (2.6 ** 7),
            lr / (2.6 ** 6),
            lr / (2.6 ** 5),
            lr / (2.6 ** 4),
            lr / (2.6 ** 3),
            lr / (2.6 ** 2),
            lr / (2.6 ** 1),
            lr,
        ]
    )
    phases = [
        TrainingPhase(anneal_start).schedule_hp("lr", lr_array),
        TrainingPhase(anneal_end).schedule_hp("lr", lr_array, anneal=annealing_cos),
    ]
    sched = GeneralScheduler(learn, phases)
    learn.callbacks.append(sched)
    learn.fit(n_epochs)


freeze_to_counter = 1
while freeze_to_counter < 5:
    print("Freezing up to " + str(freeze_to_counter))
    freeze_to = freeze_to_counter
    learner.freeze_to(-freeze_to)
    if wandb.config.opt_func == "adam":
        learner.fit_one_cycle(2, max_lr=lr_array)
    if wandb.config.opt_func == "ranger":
        flattenAnneal(learner, lr, 4, wandb.config.start_pct)
        learner.save("ans_tmp_" + str(freeze_to), with_opt=False)
        del learner
        gc.collect()
        torch.cuda.empty_cache()

        ans_databunch = TextDataBunch(
            train_dl=train_ans_dl,
            valid_dl=valid_ans_dl,
            test_dl=test_ans_dl,
            device="cuda:0",
        )

        learner = BertLearner(
            ans_databunch,
            custom_transformer_model,
            opt_func=opt_func,
            bn_wd=False,
            true_wd=True,
            metrics=[SpearmanRho()],
            callback_fns=partial(
                WandbCallback,
                save_model=True,
                mode="max",
                monitor="spearman_rho",
                seed=wandb.config.seed,
            ),
        ).to_fp16()
        learner.freeze()
        learner.load("ans_tmp_" + str(freeze_to))

    freeze_to_counter += 1
