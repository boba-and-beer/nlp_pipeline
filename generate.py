"""
Initial pre-processing of datasets
"""
from nlp_pipeline.textpreprocessing.text_to_tensor import *
from nlp_pipeline.splits.split import *
from nlp_pipeline.textpreprocessing.summary import *
from utils import *

# Import the utils in the current module
from googlequestchallenge.utils import *

# basic
import logging
import random
import string
import argparse

console_handler = logging.StreamHandler()

dl_logger = logging.getLogger(__name__)
dl_logger.addHandler(console_handler)
dl_logger.setLevel(logging.INFO)

#####################################
### Global Parameters That Need To Be Set
#####################################

SEED = random.randint(9, 999)
dl_logger.info("The seed chosen is " + str(SEED))

seed_all(SEED)

dl_logger.info("Creating pretrained model")

# PRETRAINED_MODEL_NAME = '../notebooks/pretrained_models/'
PRETRAINED_MODEL_NAME = "bert"

DATASET_DIR = "googlequestchallenge/torch_datasets"
create_dir(Path(DATASET_DIR))

input_dir = Path("googlequestchallenge/data/raw")
train = pd.read_csv(input_dir / "train.csv")
test = pd.read_csv(input_dir / "test.csv")
sample_submissions = pd.read_csv(input_dir / "sample_submission.csv")

ALL_LABELS = list(sample_submissions.columns)[1:]
ALL_COLS = ["question_title", "question_body", "answer"]

QUESTION_LABELS = [x for x in ALL_LABELS if x.startswith("question")]
QUESTION_COLS = ["question_title", "question_body"]

ANSWER_COLS = ["answer"]
ANSWER_LABELS = [x for x in ALL_LABELS if x.startswith("answer")]

#####################################
### Defining the tokenizer class and config class
#####################################
pretrain_model = Text2Tensor()
pretrain_model.choose_model("bert")

#####################################
### Feature Engineering the words
#####################################
dl_logger.info("Adding features that need to be be added to the pooler.")

train, test, ss, ANSWER_FEATURES, QUESTION_FEATURES = get_features(
    train, test, pretrain_model
)

QUESTION_ALL = QUESTION_COLS + QUESTION_FEATURES
ANSWER_ALL = ANSWER_COLS + ANSWER_FEATURES


# outputting engineered data
train.to_csv("googlequestchallenge/data/train_engineered.csv")
test.to_csv("googlequestchallenge/data/test_engineered.csv")

CAT_FEATURES = [
    col
    for col in train.columns
    if col.startswith("subcat") or col.startswith("category")
]

def main():
    # Include a flag that lets you summarise text
    # Need to also include flag to determine how best to generate the text
    text_parser = argparse.ArgumentParser(description='Process some integers.')
    
    text_parser.add_argument('--nltk', dest='nltk', action='store_true',
        help='Apply NTLK summarisation if required')
    
    text_parser.add_argument('--bert', dest='bert', action='store_true',
        help='Apply BERT Model summarisation if required')
    
    text_parser.add_argument('--USE', dest='USE', action='store_true',
        help='Apply Universal Sentence Encoder')

    text_parser.add_argument('--folds', dest='num_of_folds', type=int,
    default=5, help='Number of folds')

    text_args = text_parser.parse_args()

    for i in range(text_args.num_of_folds):

        #####################################
        ### Splitting the words
        #####################################
        dl_logger.info("Currently Using multilabel stratified split with 80 percent of data.")

        # Splitting Questions
        train_q_x, train_q_y, valid_q_x, valid_q_y = train_test_split(
            train,
            split_method=multilabelstratsplit,
            model_cols=QUESTION_ALL + CAT_FEATURES,
            model_labels=QUESTION_LABELS,
            group_col=train["question_title"],
            index=i,
            num_of_splits=text_args.num_of_folds,
            random_state=SEED,
        )

        # Splitting answers
        train_ans_x, train_ans_y, valid_ans_x, valid_ans_y = train_test_split(
            train,
            split_method=multilabelstratsplit,
            model_cols=ANSWER_ALL + CAT_FEATURES,
            model_labels=ANSWER_LABELS,
            group_col=train["question_title"],
            index=i,
            num_of_splits=text_args.num_of_folds,
            random_state=SEED,
        )

        test_q_x = test[QUESTION_ALL + CAT_FEATURES]
        test_ans_x = test[ANSWER_ALL + CAT_FEATURES]

        #####################################
        ### Change text data into iterable 
        ### In this case, we change this into pandas series
        #####################################
        train_questions = train_q_x["question_title"] + train_q_x["question_body"]
        train_answers = train_ans_x["answer"]

        valid_questions = valid_q_x["question_title"] + valid_q_x["question_body"]
        valid_answers = valid_ans_x["answer"]

        test_questions = test_q_x["question_title"] + test_q_x["question_body"]
        test_answers = test_ans_x["answer"]

        #####################################
        ### Applying Summarisation
        #####################################
        MIN_SUM_LIMIT = 180
        all_text_data = [
            train_questions, train_answers, 
            valid_questions, valid_answers, 
            test_questions, test_answers
        ]

        if text_args.nltk:
            dl_logger.info("Summarising text using nltk...")
            for text_data in tqdm(all_text_data):
                for i, text in tqdm(enumerate(text_data)):
                    if len(text.split(' ')) > MIN_SUM_LIMIT:
                        text_data.iloc[i] = nltk_summarise(text, max_sent_len=15,
                        n_largest=4)

        if text_args.bert:
            # Bert summariser
            from summarizer import Summarizer
            bert_model = Summarizer()
            dl_logger.info("SUmmarising text using Bert...")
            for i, text in tqdm(enumerate(train_answers)):
                if len(text.split(' ')) > MIN_SUM_LIMIT:
                    train_answers.iloc[i] = bert_model(text)
            
            for i, text in tqdm(enumerate(valid_answers)):
                if len(text.split(' ')) > MIN_SUM_LIMIT:
                    valid_answers.iloc[i] = bert_model(text)
            
            for i, text in tqdm(enumerate(test_answers)):
                if len(text.split(' ')) > MIN_SUM_LIMIT:
                    test_answers.iloc[i] = bert_model(text)

        #####################################
        ### Converting Text to tensor
        #####################################

        dl_logger.info("Tokenising data using only head")

        # Input dataset
        if text_args.USE:
            encode_method = "USE"
        else:
            encode_method = None

        dl_logger.info("Tokenising Question data...")
        train_q_input_list, train_q_att_list = pretrain_model.convert_text_to_tensor(
            train_questions, encode_method=encode_method
        )

        dl_logger.info("Tokenising answer data...")
        (
            train_ans_input_list,
            train_ans_att_list,
        ) = pretrain_model.convert_text_to_tensor(train_answers, 
            encode_method=encode_method
            )

        # Validation dataset
        valid_q_input_list, valid_q_att_list = pretrain_model.convert_text_to_tensor(
            valid_questions, encode_method=encode_method
        )
        
        (
            valid_ans_input_list,
            valid_ans_att_list,
        ) = pretrain_model.convert_text_to_tensor(valid_answers, 
            encode_method=encode_method)

        # Test dataset
        test_q_input_list, test_q_att_list = pretrain_model.convert_text_to_tensor(
            test_questions, encode_method=encode_method
        )

        test_ans_input_list, test_ans_att_list = pretrain_model.convert_text_to_tensor(
            test_answers, encode_method=encode_method
        )

        # Calculate the train tens+ str(wandb.config.fold) + "_ors
        train_q_label_tensors = torch.FloatTensor(train_q_y[QUESTION_LABELS].values)
        valid_q_label_tensors = torch.FloatTensor(valid_q_y[QUESTION_LABELS].values)
        test_q_label_tensors = torch.zeros(test_q_x.size, 30)

        train_ans_label_tensors = torch.FloatTensor(train_ans_y[ANSWER_LABELS].values)
        valid_ans_label_tensors = torch.FloatTensor(valid_ans_y[ANSWER_LABELS].values)
        test_ans_label_tensors = torch.zeros(test_ans_x.size, 30)

        # Question features
        train_q_features = torch.FloatTensor(
            train_q_x[QUESTION_FEATURES + CAT_FEATURES].values
        )
        valid_q_features = torch.FloatTensor(
            valid_q_x[QUESTION_FEATURES + CAT_FEATURES].values
        )
        test_q_features = torch.FloatTensor(
            test_q_x[QUESTION_FEATURES + CAT_FEATURES].values
        )

        # # Answer features
        train_ans_features = torch.FloatTensor(
            train_ans_x[ANSWER_FEATURES + CAT_FEATURES].values
        )
        valid_ans_features = torch.FloatTensor(
            valid_ans_x[ANSWER_FEATURES + CAT_FEATURES].values
        )
        test_ans_features = torch.FloatTensor(
            test_ans_x[ANSWER_FEATURES + CAT_FEATURES].values
        )
        
        dl_logger.info("Creating datasets")
        train_q_dataset = QuestDataset(
            input_ids=train_q_input_list,
            attention_mask=train_q_att_list,
            label_tensors=train_q_label_tensors,
            engineered_features=train_q_features,
        )

        valid_q_dataset = QuestDataset(
            input_ids=valid_q_input_list,
            attention_mask=valid_q_att_list,
            label_tensors=valid_q_label_tensors,
            engineered_features=valid_q_features,
        )

        test_q_dataset = QuestDataset(
            input_ids=test_q_input_list,
            attention_mask=test_q_att_list,
            label_tensors=valid_q_label_tensors,
            engineered_features=test_q_features,
        )
        
        train_ans_dataset = QuestDataset(
            input_ids=train_ans_input_list,
            attention_mask=train_ans_att_list,
            label_tensors=train_ans_label_tensors,
            engineered_features=train_ans_features,
        )

        valid_ans_dataset = QuestDataset(
            input_ids=valid_ans_input_list,
            attention_mask=valid_ans_att_list,
            label_tensors=valid_ans_label_tensors,
            engineered_features=valid_ans_features,
        )

        test_ans_dataset = QuestDataset(
            input_ids=test_q_input_list,
            attention_mask=test_q_att_list,
            label_tensors=valid_ans_label_tensors,
            engineered_features=test_ans_features,
        )
        
        dl_logger.info("Saving dataloaders in " + DATASET_DIR)
        create_dir(Path("googlequestchallenge/" + DATASET_DIR))

        # Saving the datasets depending on flags 
        dataset_name = ''

        for name in text_args.__dict__:
            # If flag is true
            if text_args.__dict__[name]:
                # We want features, nut the number of folds
                if name == "num_of_folds": continue
                dataset_name += name + '_'
        
        torch_save_dict = {
            train_q_dataset: "train_q_",
            valid_q_dataset: "valid_q_",
            test_q_dataset: "test_q_",
            train_ans_dataset: "train_ans_",
            valid_ans_dataset: "valid_ans_",
            test_ans_dataset: "test_ans_"
        }

        for dataset, value in torch_save_dict.items():
            # dataset dir, description with underscores and fold number
            if "test" not in value:
                output_path = Path(DATASET_DIR, value + str(dataset_name) + str(i))
                torch.save(dataset, output_path)
                dl_logger.info("Saving to: "+ str(output_path))
            else:
                output_path = Path(DATASET_DIR, value + str(dataset_name))
                # We do not save the test fold because there is no fold
                torch.save(dataset, output_path)
                dl_logger.info("Saving to: "+str(output_path))

        # torch.save(
        #     train_q_dataset,
        #     Path(DATASET_DIR, "train_q_" + str(dataset_name) + str(i)),
        # )
        # torch.save(
        #     valid_q_dataset,
        #     Path(DATASET_DIR, "valid_q_" + str(dataset_name) + str(i)),
        # )
        # torch.save(test_q_dataset, Path(DATASET_DIR, "test_q_use_"))

        # torch.save(
        #     train_ans_dataset,
        #     Path(DATASET_DIR, "train_ans_" + str(dataset_name) + str(i))
        # )
        # torch.save(
        #     valid_ans_dataset,
        #     Path(DATASET_DIR, "valid_ans_" + str(dataset_name) + str(i))
        # )
        # torch.save(
        #     test_ans_dataset, Path(DATASET_DIR, "test_ans_"),
        # )

if __name__ == "__main__":
    main()