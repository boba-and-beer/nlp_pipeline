"""
This script aims to ensemble the best models and is run after 
answer_model.py and question_model.py.
"""
from collections import defaultdict
from generate import *
from answer_model import *
from nlp_pipeline.inference.predict import *
import pickle

os.environ["WANDB_MODE"] = "dryrun"

# Idenfiy the best models for us here.
model_names = pickle.load(open("ans_list.pkl", "rb"))

# Select which models to get
ans_learner = Learner(
    ans_databunch,
    custom_transformer_model,
    opt_func=opt_func,
    bn_wd=False,
    true_wd=True,
)

ans_learner.model_dir = "models"

# Load the dictionary
predictions_path = Path("predictions.pkl")
if predictions_path.exists():
    predictions = pickle.load(open(predictions_path, "rb"))
else:
    predictions = defaultdict(dict)

# Loading the model using wandb restore?
all_names = []
for path in model_names:
    # if the first fold exists, we load the other models
    if Path(ans_learner.model_dir, path + "_4.pth").exists():
        # Found the fold - now we save the model
        for i in range(1, 5):
            full_model_name = path + "_" + str(i)
            # The prediction has already been made
            if full_model_name in predictions.keys():
                continue
            ans_learner.load(full_model_name)
            ans_predictions, ans_truth = get_ordered_preds(
                ans_learner, DatasetType.Test
            )
            predictions[full_model_name] = ans_predictions
            all_names.append(full_model_name)

# Save the predictions
pickle.dump(predictions, open(predictions_path, "wb"))

# for each pair of models in the list of models:
for path_1 in all_names:
    for path_2 in all_names:
        if path_1 == path_2:
            continue
        if (
            Path(ans_learner.model_dir, path_1).exists()
            and Path(ans_learner.model_dir, path_2).exists()
        ):
            scores = []
            for i, col in enumerate(ANSWER_LABELS):
                preds_1 = predictions[path_1][:, i]
                preds_2 = predictions[path_2][:, i]
                sr_score, p_value = spearmanr(preds_1, preds_2)
                print(sr_score)
                scores.append(sr_score)
            avg_spearman_score = np.mean(scores)
            print("The spearman spearman score between")
            print("Path 1 is" + path_1)
            print("Path 2 is " + path_2)
            print(avg_spearman_score)
