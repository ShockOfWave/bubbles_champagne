import torch 
import pickle 
import numpy as np 
from tqdm import tqdm
from glob import glob 
from pytorch_tabnet.pretraining import TabNetPretrainer
from pytorch_tabnet.tab_model import TabNetClassifier
from embeds_generator import extract_embeddings

converter = {
            "pink": 0,
            "white": 1,
            "plastic": 0,
            "glass": 1,
            "0": 0,
            "10": 1,
            "15": 2,
            "20": 3,
            "min": 0
            }


train_paths = glob.glob("data/train/*/*.jpg")
val_paths = glob.glob("data/val/*/*.jpg")

X_train = []
for path in tqdm(train_paths, desc="Generating train embeddings"):
    X_train += [extract_embeddings(path)]

X_val = []
for path in tqdm(val_paths, desc="Generating val embeddings"):
    X_train += [extract_embeddings(path)]

X_train = np.array(X_train)
X_val = np.array(X_val)

pickle.dump(X_train, open("train.pkl", "wb"))
pickle.dump(X_val, open("val.pkl", "wb"))
print("Embeddings saved!")

print("Train task 1!")
y_train = []
for path in tqdm(train_paths):
    path_parts = path.split('/')
    label1, label2, label3, _ = [converter[x] for x in path_parts[-2].split('_')]
    y_train.append(label1)
y_train = np.array(y_train)

y_val = []
for path in tqdm(val_paths):
    path_parts = path.split('/')
    label1, label2, label3, _ = [converter[x] for x in path_parts[-2].split('_')]
    y_val.append(label1)
y_val = np.array(y_val)


unsupervised_model = TabNetPretrainer(
    optimizer_fn=torch.optim.AdamW,
    optimizer_params=dict(lr=2e-2),
    mask_type='entmax', # "sparsemax",
    verbose=10
)

unsupervised_model.fit(
    X_train=X_train,
    eval_set=[X_val],
    pretraining_ratio=0.8,
)

clf = TabNetClassifier(
    n_d=64,
    optimizer_fn=torch.optim.Adam,
    optimizer_params=dict(lr=2e-2),
    scheduler_params={"step_size":10, # how to use learning rate scheduler
                        "gamma":0.9},
    scheduler_fn=torch.optim.lr_scheduler.StepLR,
    mask_type='entmax', # This will be overwritten if using pretrain model
    verbose=10
    )

clf.fit(
  X_train, y_train,
  eval_set=[(X_val, y_val)],
  eval_metric=["accuracy", "balanced_accuracy"],
  patience=30,
  from_unsupervised=unsupervised_model,
)

saving_path_name = "chamange_model_1"
saved_filepath = clf.save_model(saving_path_name)
print("Model for task 1 saved!")

print("Training task 2!")
y_train = []
for path in tqdm(train_paths):
    path_parts = path.split('/')
    label1, label2, label3, _ = [converter[x] for x in path_parts[-2].split('_')]
    y_train.append(label2)
y_train = np.array(y_train)

y_val = []
for path in tqdm(val_paths):
    path_parts = path.split('/')
    label1, label2, label3, _ = [converter[x] for x in path_parts[-2].split('_')]
    y_val.append(label2)
y_val = np.array(y_val)

clf = TabNetClassifier(
    n_d=64,
    optimizer_fn=torch.optim.Adam,
    optimizer_params=dict(lr=2e-2),
    scheduler_params={"step_size":10, # how to use learning rate scheduler
                        "gamma":0.9},
    scheduler_fn=torch.optim.lr_scheduler.StepLR,
    mask_type='entmax', # This will be overwritten if using pretrain model
    verbose=10
    )

clf.fit(
  X_train, y_train,
  eval_set=[(X_val, y_val)],
  eval_metric=["auc", "accuracy", "balanced_accuracy"],
  patience=30,
  from_unsupervised=unsupervised_model,
    # batch_size=128, virtual_batch_size=256,
)

saving_path_name = "chamange_model_2"
saved_filepath = clf.save_model(saving_path_name)
print("Model for task 2 saved!")

print("Training task 2!")
y_train = []
for path in tqdm(train_paths):
    path_parts = path.split('/')
    label1, label2, label3, _ = [converter[x] for x in path_parts[-2].split('_')]
    y_train.append(label3)
y_train = np.array(y_train)

y_val = []
for path in tqdm(val_paths):
    path_parts = path.split('/')
    label1, label2, label3, _ = [converter[x] for x in path_parts[-2].split('_')]
    y_val.append(label3)
y_val = np.array(y_val)

clf = TabNetClassifier(
    n_d=64, n_a=64, n_steps=5,
    gamma=1.5, n_independent=2, n_shared=2,
    cat_emb_dim=1,
    lambda_sparse=1e-4, momentum=0.3, clip_value=2.,
    optimizer_fn=torch.optim.Adam,
    optimizer_params=dict(lr=2e-2),
    scheduler_params = {"gamma": 0.95,
                     "step_size": 20},
    scheduler_fn=torch.optim.lr_scheduler.StepLR, epsilon=1e-15,
    verbose=25
)

clf.fit(
  X_train, y_train,
  eval_set=[(X_val, y_val)],
  eval_metric=["accuracy", "balanced_accuracy"],
  patience=100,
  from_unsupervised=unsupervised_model,
  batch_size=128, virtual_batch_size=256,
)


saving_path_name = "chamange_model_3"
saved_filepath = clf.save_model(saving_path_name)
print("Model for task 3 saved!")