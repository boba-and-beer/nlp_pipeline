"""
Model for questions
"""
from logger import *
from utils import *
from nlp_pipeline.models.pooler import *
from nlp_pipeline.models.bert import *
from nlp_pipeline.models.transformers import *
from nlp_pipeline.metrics.ranks import *
from googlequestchallenge.utils import *
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
name = "q_use_" + name

wandb.init(name=name, notes=notes, project="q_googlequestchallenge")

## Features i do not want to change
wandb.config.token_type = "Head Only"
wandb.config.start_pct = 0.55
wandb.config.wd = 0.05
wandb.config.with_features = True
# opt func is either ranger of flatten anneal
wandb.config.opt_func = "adam"
wandb.config.dropout = 0.05
wandb.config.betas_0 = 0.9
wandb.config.betas_1 = 0.99
wandb.config.seed = random.randint(0, 999)
bs = 8
seed_all(wandb.config.seed)
wandb.config.hidden_layer_output = 2
wandb.config.fold = 0

model_logger.info(" The name of the model is %s", name)
model_logger.info("The random seed is %d", wandb.config.seed)

#####################################
### Loading Models
#####################################

PRETRAINED_MODEL_NAME = "bert-base-uncased"

DATASET_DIR = "googlequestchallenge/torch_datasets/"

MODEL_NAME = name
OUTPUT_MODEL_DIR = Path("..", "models", "q_models", MODEL_NAME)

input_dir = Path("googlequestchallenge/data/raw")
sample_submissions = pd.read_csv(input_dir / "sample_submission.csv")
engineered_input_dir = Path("googlequestchallenge/data")
train = pd.read_csv(engineered_input_dir / "train_engineered.csv")
test = pd.read_csv(engineered_input_dir / "test_engineered.csv")


#####################################
### Loading the datasets
#####################################

dataset_name = 'use_'

# Iterating over datasets
train_q_dataset = torch.load(
    DATASET_DIR + "train_q_" + dataset_name + str(wandb.config.fold)
)
valid_q_dataset = torch.load(
    DATASET_DIR + "valid_q_" + dataset_name + str(wandb.config.fold)
)
test_q_dataset = torch.load(DATASET_DIR + "test_q_" + dataset_name)

# Load in the custom tokenizer still
pretrain_model.config.num_labels = len(QUESTION_LABELS)
pretrain_model.config.output_hidden_states = True

trn_q_samp = SortishSampler(
    train_q_dataset, key=lambda x: len(train_q_dataset.y), bs=bs
)

valid_q_samp = SortishSampler(
    valid_q_dataset, key=lambda x: len(valid_q_dataset.y), bs=bs
)

test_q_samp = SortishSampler(
    test_q_dataset, key=lambda x: len(test_q_dataset.y), bs=bs
)

dl_kwargs = {
    "batch_size": bs,
    "shuffle": False,
    "batch_sampler": None,
    "num_workers": 0,
    # "pin_memory": True, - this has been defined in the collate wrapper implementation
}

train_q_dl = DataLoader(train_q_dataset, sampler=trn_q_samp, 
    # collate_fn=collate_wrapper, 
    **dl_kwargs)
valid_q_dl = DataLoader(valid_q_dataset, sampler=valid_q_samp, 
    # collate_fn=collate_wrapper, 
    **dl_kwargs)
test_q_dl = DataLoader(test_q_dataset, sampler=test_q_samp, 
    # collate_fn=collate_wrapper, 
    **dl_kwargs)

#####################################
### Model
#####################################

# Model for universal sentence encoder
wandb.config.universal_sent_encoder = True
if wandb.config.universal_sent_encoder:
    model_class = BertWithEmbeds
elif wandb.config.with_features:
    model_class = BertForFeatures
else:
    model_class = BertSequenceClassification

q_transformer_model = model_class.from_pretrained(
    PRETRAINED_MODEL_NAME,
    config=pretrain_model.config,
    dropout_rate=wandb.config.dropout,
    hidden_layer_output=wandb.config.hidden_layer_output,
)

if wandb.config.universal_sent_encoder:
    q_custom_transformer_model = CTMEncoded(
        q_transformer_model, len(ANSWER_FEATURES + CAT_FEATURES)
    )
elif wandb.config.with_features:
    # Add the number of custom features as well.
    q_custom_transformer_model = CTMWithFeatures(
        q_transformer_model, len(ANSWER_FEATURES + CAT_FEATURES)
    )
else:
    q_custom_transformer_model = CustomTransformerModel(q_transformer_model)

# Save all models here - to upload to python
q_databunch = TextDataBunch(
    train_dl=train_q_dl, valid_dl=valid_q_dl, test_dl=test_q_dl, device="cuda:0",
)

opt_func = partial(
    Ranger, betas=(wandb.config.betas_0, wandb.config.betas_1), eps=wandb.config.wd
)

# Adds a multi-layer perceptron
if wandb.config.with_features:
    q_learn_class = BertFeatLearner
else:
    q_learn_class = BertLearner

q_learner = q_learn_class(
    q_databunch,
    q_custom_transformer_model,
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
        # partial(
        #     ReduceLROnPlateauCallback,
        #     monitor="spearman_rho",
        #     mode="max",
        #     patience=3,
        #     min_delta=0.4,
        #     min_lr=1e-6,
        # )
    ],
).to_fp16()


if __name__ == "__main__":
    # add_to_list("q_list.pkl", name)
    # create_dir(OUTPUT_MODEL_DIR)

    # q_transformer_model.save_pretrained(OUTPUT_MODEL_DIR)
    # pretrain_model.config.save_pretrained(OUTPUT_MODEL_DIR)
    # pretrain_model.tokeniser.save_pretrained(OUTPUT_MODEL_DIR)
    
    q_learner.freeze()

    q_learner.lr_find()
    lr = q_learner.recorder.lrs[np.array(q_learner.recorder.losses).argmin()] / 10

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
    
    q_learner.model_dir = OUTPUT_MODEL_DIR
    freeze_to_counter = 1
    while freeze_to_counter < 6:
        freeze_to = freeze_to_counter
        model_logger.info("Freezing up to "+str(freeze_to))
        # Saving the model name
        model_save_name = name + "_" + str(wandb.config.fold) + str(freeze_to)
        q_learner.freeze_to(-freeze_to)
        flattenAnneal(q_learner, lr, 5, wandb.config.start_pct)
        # Save the models 
        q_learner.save(model_save_name, with_opt=False)
        del q_learner
        gc.collect()
        torch.cuda.empty_cache()
        q_databunch = TextDataBunch(
            train_dl=train_q_dl,
            valid_dl=valid_q_dl,
            test_dl=test_q_dl,
            device="cuda:0",
        )

        q_learner = q_learn_class(
            q_databunch,
            q_custom_transformer_model,
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
                #     partial(
                #     ReduceLROnPlateauCallback,
                #     monitor="spearman_rho",
                #     mode="max",
                #     patience=3,
                #     min_delta=0.004,
                #     min_lr=1e-6,
                # )
            ],
        ).to_fp16()
        # Kaggle is better than a legal drug
        q_learner.model_dir = OUTPUT_MODEL_DIR
        q_learner.load(model_save_name);
        freeze_to_counter += 1
