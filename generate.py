"""
Generate the required datasets in python.
"""
from nlp_pipeline.textpreprocessing.text_to_tensor import *
from nlp_pipeline.splits.split import *
from nlp_pipeline.textpreprocessing.summary import *
from utils import *

# Important thing to look at
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

# Used For Naming The Models
feature_set = "head_features_9"


def get_features(train, test, pretrain_model):
    """Generate a few features here"""
    # Mix question title and body together
    df = pd.concat([train, test], sort=True)
    df["question"] = df["question_title"] + df["question_body"]

    # Generate the features
    # 0
    df["q_word_count"] = df["question"].apply(lambda x: len(x.split()))
    df["a_word_count"] = df["answer"].apply(lambda x: len(x.split()))

    # 1
    df["q_char_count"] = df["question"].apply(lambda x: len(x.replace(" ", "")))
    df["a_char_count"] = df["answer"].apply(lambda x: len(x.replace(" ", "")))

    # 2
    df["a_word_density"] = df["a_word_count"] / (df["a_char_count"] + 1)

    # 3
    df["q_count_numbers"] = df["question"].apply(count_numbers)
    df["ans_count_numbers"] = df["answer"].apply(count_numbers)

    # 4
    df["q_brackets"] = df["question"].apply(lambda x: x.count("()"))
    df["a_brackets"] = df["answer"].apply(lambda x: x.count("()"))

    df["q_or"] = df["question"].apply(lambda x: x.count("or"))
    df["a_or"] = df["answer"].apply(lambda x: x.count("or"))

    # 5
    df["q_total_length"] = df["question"].apply(len)
    df["a_total_length"] = df["answer"].apply(len)

    # 6
    df["q_capitals"] = df["question"].apply(
        lambda comment: sum(1 for c in comment if c.isupper())
    )
    df["a_capitals"] = df["answer"].apply(
        lambda comment: sum(1 for c in comment if c.isupper())
    )

    # 7
    df["q_caps_vs_length"] = df.apply(
        lambda row: float(row["q_capitals"]) / float(row["q_total_length"]), axis=1
    )
    df["a_caps_vs_length"] = df.apply(
        lambda row: float(row["a_capitals"]) / float(row["q_total_length"]), axis=1
    )

    # 9
    df["q_num_question_marks"] = df["question"].apply(lambda x: x.count("?"))
    df["a_num_question_marks"] = df["answer"].apply(lambda x: x.count("?"))

    df["q_eq"] = df["question"].apply(lambda x: x.count("="))
    df["a_eq"] = df["answer"].apply(lambda x: x.count("="))

    # 10
    df["q_num_punctuation"] = df["question"].apply(
        lambda x: sum(x.count(w) for w in ".,;:")
    )
    df["a_num_punctuation"] = df["answer"].apply(
        lambda x: sum(x.count(w) for w in ".,;:")
    )

    # 12
    df["q_num_unique_words"] = df["question"].apply(
        lambda x: len(set(w for w in x.split()))
    )
    df["a_num_unique_words"] = df["answer"].apply(
        lambda x: len(set(w for w in x.split()))
    )

    # 13
    df["q_words_vs_unique"] = df["q_num_unique_words"] / df["q_word_count"]
    df["a_words_vs_unique"] = df["a_num_unique_words"] / df["a_word_count"]

    # 14 - num of lines
    df["q_num_of_lines"] = df["question"].apply(lambda x: x.count("\n"))
    df["a_num_of_lines"] = df["answer"].apply(lambda x: x.count("\n"))

    # ask yourself why you know
    df["q_why"] = df["question"].apply(lambda x: x.lower().count("why"))
    df["a_why"] = df["answer"].apply(lambda x: x.lower().count("why"))

    df["q_how"] = df["question"].apply(lambda x: x.lower().count("how "))
    df["a_how"] = df["answer"].apply(lambda x: x.lower().count("how "))

    # Adding categorical column data
    dummy_cols = pd.get_dummies(df["category"], drop_first=False, prefix="category")
    df = pd.concat([df, dummy_cols], axis=1)

    # Adding subcategories - change the 4
    df = df.reset_index()
    df["subcategory"] = (
        df["host"]
        .str.replace(".stackexchange", "")
        .str.replace(".com", "")
        .str.replace(".net", "")
        .str.replace("meta.", "")
    )

    subcategory_text = df["subcategory"]
    df.drop(["category", "subcategory"], axis=1, inplace=True)

    df_subcat, _ = pretrain_model.convert_text_to_tensor(
        text=subcategory_text,
        max_length=4,
        add_special_tokens=False,
        return_tensors=None,
    )

    subcat_df = pd.DataFrame(
        df_subcat, columns=["subcat_0", "subcat_1", "subcat_2", "subcat_3"]
    )

    # Normalize
    from sklearn.preprocessing import StandardScaler

    ss = StandardScaler(with_mean=True, with_std=True)

    ANSWER_FEATURES = [col for col in df.columns if col.startswith("a_")]
    QUESTION_FEATURES = [col for col in df.columns if col.startswith("q_")]
    subcat_features = [col for col in subcat_df.columns if col.startswith("subcat_")]

    # For each column we have to perform standard scaling
    train_features = df.head(len(train))
    train_subcat = subcat_df.head(len(train))
    test_features = df.tail(len(test))
    test_subcat = subcat_df.tail(len(test))

    # Combining the trian features
    train_features = pd.concat([train_features, train_subcat], axis=1)

    test_subcat.index = test_features.index
    test_features = pd.concat([test_features, test_subcat], axis=1)

    # Normalize the training data and then apply to test data
    train_features.loc[:, ANSWER_FEATURES + QUESTION_FEATURES] = ss.fit_transform(
        train_features.loc[:, ANSWER_FEATURES + QUESTION_FEATURES]
    )

    test_features[ANSWER_FEATURES + QUESTION_FEATURES] = ss.transform(
        test_features[ANSWER_FEATURES + QUESTION_FEATURES]
    )

    # Converting to test features cos it's a bit weird (removing bottom NA values)
    test_features = test_features.iloc[: test_features.shape[0]]

    return train_features, test_features, ss, ANSWER_FEATURES, QUESTION_FEATURES


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


#####################################
### Torch Datasets
#####################################
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

def main():
    # Include a flag that lets you summarise text
    text_parser = argparse.ArgumentParser(description='Process some integers.')
    text_parser.add_argument('--nltk', dest='nltk', action='store_true',
        help='Apply NTLK summarisation if required')
    text_parser.add_argument('--bert', dest='bert', action='store_true',
        help='Apply BERT Model summarisation if required')

    text_args = text_parser.parse_args()

    for i in range(5):

        #####################################
        ### Splitting the words
        #####################################
        dl_logger.info("Using multilabel stratified split with 90 percent of data.")

        # Splitting Questions
        train_q_x, train_q_y, valid_q_x, valid_q_y = train_test_split(
            train,
            split_method=multilabelstratsplit,
            model_cols=QUESTION_ALL + CAT_FEATURES,
            model_labels=QUESTION_LABELS,
            group_col=train["question_title"],
            index=i,
            num_of_splits=5,
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
            num_of_splits=5,
            random_state=SEED,
        )

        test_q_x = test[QUESTION_ALL + CAT_FEATURES]
        test_ans_x = test[ANSWER_ALL + CAT_FEATURES]

        #####################################
        ### Create Text Data Into Pandas Series
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
        MIN_SUM_LIMIT = 150
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
                        n_largest=3)

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
        dl_logger.info("Tokenising Question data...")
        train_q_input_list, train_q_att_list = pretrain_model.convert_text_to_tensor(
            train_questions, head_prop=1
        )
        dl_logger.info("Tokenising answer data...")
        (
            train_ans_input_list,
            train_ans_att_list,
        ) = pretrain_model.convert_text_to_tensor(train_answers, head_prop=1)

        # Validation dataset
        valid_q_input_list, valid_q_att_list = pretrain_model.convert_text_to_tensor(
            valid_questions, head_prop=1
        )
        (
            valid_ans_input_list,
            valid_ans_att_list,
        ) = pretrain_model.convert_text_to_tensor(valid_answers, head_prop=1)

        # Test dataset
        test_q_input_list, test_q_att_list = pretrain_model.convert_text_to_tensor(
            test_questions, head_prop=1
        )
        test_ans_input_list, test_ans_att_list = pretrain_model.convert_text_to_tensor(
            test_answers, head_prop=1
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
            embed_tensors=train_q_input_list,
            attention_tensors=train_q_att_list,
            label_tensors=train_q_label_tensors,
            engineered_features=train_q_features,
        )

        valid_q_dataset = QuestDataset(
            embed_tensors=valid_q_input_list,
            attention_tensors=valid_q_att_list,
            label_tensors=valid_q_label_tensors,
            engineered_features=valid_q_features,
        )

        test_q_dataset = QuestDataset(
            embed_tensors=test_q_input_list,
            attention_tensors=test_q_att_list,
            label_tensors=valid_q_label_tensors,
            engineered_features=test_q_features,
        )

        train_ans_dataset = QuestDataset(
            embed_tensors=train_ans_input_list,
            attention_tensors=train_ans_att_list,
            label_tensors=train_ans_label_tensors,
            engineered_features=train_ans_features,
        )

        valid_ans_dataset = QuestDataset(
            embed_tensors=valid_ans_input_list,
            attention_tensors=valid_ans_att_list,
            label_tensors=valid_ans_label_tensors,
            engineered_features=valid_ans_features,
        )

        test_ans_dataset = QuestDataset(
            embed_tensors=test_q_input_list,
            attention_tensors=test_q_att_list,
            label_tensors=valid_ans_label_tensors,
            engineered_features=test_ans_features,
        )

        dl_logger.info("Saving dataloaders in " + DATASET_DIR)
        create_dir(Path("googlequestchallenge/" + DATASET_DIR))

        # Saving the datasets
        torch.save(
            train_q_dataset,
            Path(DATASET_DIR, "train_q_ds_" + str(i) + "_" + feature_set),
        )
        torch.save(
            valid_q_dataset,
            Path(DATASET_DIR, "valid_q_ds_" + str(i) + "_" + feature_set),
        )
        torch.save(test_q_dataset, Path(DATASET_DIR, "test_q_ds_" + feature_set))

        torch.save(
            train_ans_dataset,
            Path(DATASET_DIR, "train_ans_ds_" + str(i) + "_" + feature_set),
        )
        torch.save(
            valid_ans_dataset,
            Path(DATASET_DIR, "valid_ans_ds_" + str(i) + "_" + feature_set),
        )
        torch.save(
            test_ans_dataset, Path(DATASET_DIR, "test_ans_ds_" + feature_set),
        )

if __name__ == "__main__":
    main()