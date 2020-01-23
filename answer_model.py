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
notes = "Testing Universal Sentence Encoder"

# change the name of the run to follow the features
name = "".join(random.choices(string.ascii_uppercase + string.digits, k=3))
name = "ans_use_" + name

wandb.init(name=name, notes=notes, project="ans_googlequestchallenge")

## Features i do not want to change
wandb.config.token_type = "Head Only"
wandb.config.start_pct = 0.55
wandb.config.wd = 0.05
# opt func is either ranger of flatten anneal
wandb.config.dropout = 0.05
wandb.config.betas_0 = 0.9
wandb.config.betas_1 = 0.99
wandb.config.seed = random.randint(0, 999)
bs = 8
seed_all(wandb.config.seed)
wandb.config.hidden_layer_output = 2

wandb.config.with_features = True

# The 3 was 
# if 'fold' not in wandb.config.keys(): wandb.config.fold = 3

model_logger.info(" The name of the model is %s", name)
model_logger.info("The random seed is %d", wandb.config.seed)

#####################################
### Loading Models
#####################################

PRETRAINED_MODEL_NAME = "bert-base-uncased"

DATASET_DIR = "googlequestchallenge/torch_datasets/"

# Saving the model here
OUTPUT_MODEL_DIR = Path("models", "ans_models", name)

input_dir = Path("googlequestchallenge/data/raw")
sample_submissions = pd.read_csv(input_dir / "sample_submission.csv")
engineered_input_dir = Path("googlequestchallenge/data")
train = pd.read_csv(engineered_input_dir / "train_engineered.csv")
test = pd.read_csv(engineered_input_dir / "test_engineered.csv")

#####################################
### Loading the datasets
#####################################

# Saving the datasets
train_ans_dataset = torch.load(
    DATASET_DIR + "train_ans_ds_" + str(wandb.config.fold) + "_" + feature_set
)
valid_ans_dataset = torch.load(
    DATASET_DIR + "valid_ans_ds_" + str(wandb.config.fold) + "_" + feature_set
)
test_ans_dataset = torch.load(DATASET_DIR + "test_ans_ds_" + feature_set)

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
    test_ans_dataset, key=lambda x: len(test_ans_dataset.x), bs=bs
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

if wandb.config.with_features:
    model_class = BertForFeatures
else:
    model_class = BertSequenceClassification

transformer_model = model_class.from_pretrained(
    PRETRAINED_MODEL_NAME,
    config=pretrain_model.config,
    dropout_rate=wandb.config.dropout,
    hidden_layer_output=wandb.config.hidden_layer_output,
)

if wandb.config.with_features:
    # Add the number of custom features as well.
    custom_transformer_model = CTMWithFeatures(
        transformer_model, len(ANSWER_FEATURES + CAT_FEATURES)
    )
else:
    custom_transformer_model = CustomTransformerModel(transformer_model)

# Save all models here - to upload to python
ans_databunch = TextDataBunch(
    train_dl=train_ans_dl, valid_dl=valid_ans_dl, test_dl=test_ans_dl, device="cuda:0"
)

opt_func = partial(
    Ranger, betas=(wandb.config.betas_0, wandb.config.betas_1), eps=wandb.config.wd
)

if wandb.config.with_features:
    learn_class = BertFeatLearner
else:
    learn_class = BertLearner

learner = learn_class(
    ans_databunch,
    custom_transformer_model,
    opt_func=opt_func,
    bn_wd=False,
    true_wd=True,
    metrics=[SpearmanRho()],
    callback_fns=[
        partial(
            WandbCallback,
            save_model=True,
            mode="max",
            monitor="spearman_rho",
            seed=wandb.config.seed,
        ),
        partial(
            ReduceLROnPlateauCallback,
            monitor="spearman_rho",
            mode="max",
            patience=3,
            min_delta=0.004,
            min_lr=1e-6,
        ),
    ],
).to_fp16()

#####################################
### Saving the models
#####################################
# Save the model configs here.

if __name__ == "__main__":
    # If this script is called, THEN add it to the directory and save the model
    model_logger.info("Writing the file...")
    add_to_list(filehandler="ans_list.pkl", name=name)
    create_dir(OUTPUT_MODEL_DIR)

    transformer_model.save_pretrained(OUTPUT_MODEL_DIR)
    pretrain_model.config.save_pretrained(OUTPUT_MODEL_DIR)
    pretrain_model.tokeniser.save_pretrained(OUTPUT_MODEL_DIR)

    learner.freeze()
    learner.lr_find()
    lr = learner.recorder.lrs[np.array(learner.recorder.losses).argmin()] / 10

    model_logger.info("We chose to use lR: %d", lr)

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

    learner.model_dir = OUTPUT_MODEL_DIR
    freeze_to_counter = 1
    while freeze_to_counter < 6:
        freeze_to = freeze_to_counter
        model_logger.info("Freezing up to "+str(freeze_to))
        # Saving the model name
        model_save_name = name + "_" + str(wandb.config.fold) + str(freeze_to)
        learner.freeze_to(-freeze_to)
        flattenAnneal(learner, lr, 5, wandb.config.start_pct)
        # Save the models 
        learner.save(model_save_name, with_opt=False)
        del learner
        gc.collect()
        torch.cuda.empty_cache()
        ans_databunch = TextDataBunch(
            train_dl=train_ans_dl,
            valid_dl=valid_ans_dl,
            test_dl=test_ans_dl,
            device="cuda:0",
        )
        
        learner = learn_class(
            ans_databunch,
            custom_transformer_model,
            opt_func=opt_func,
            metrics=[SpearmanRho()],
            callback_fns=[
                partial(
                    WandbCallback,
                    save_model=True,
                    mode="max",
                    monitor="spearman_rho",
                    seed=wandb.config.seed,
                ),
                    partial(
                    ReduceLROnPlateauCallback,
                    monitor="spearman_rho",
                    mode="max",
                    patience=3,
                    min_delta=0.004,
                    min_lr=1e-6,
                )
            ],
        ).to_fp16()
        # Kaggle is better than a legal drug
        learner.model_dir = OUTPUT_MODEL_DIR
        learner.load(model_save_name);
        freeze_to_counter += 1
