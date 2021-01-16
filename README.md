# Riiid! Answer Correctness Prediction solution

This is the 3rd place solution source code to [Kaggle's Riiid! Answer Correctness Prediction competition](https://www.kaggle.com/c/riiid-test-answer-prediction/overview). For a brief write-up and comments please check the [discussion topic on Kaggle](https://www.kaggle.com/c/riiid-test-answer-prediction/discussion/209585).

The solution will be presented at the [35th AAAI Conference on Artificial Intelligence (2021)](https://sites.google.com/view/tipce-2021/home).

# Steps to reproduce

## Env setup

Clone this repo and create `input` directory:

```
git clone https://github.com/jamarju/riiid-acp-pub
mkdir input
```

Unzip the dataset into `input` or just copy over the required files:

* `train.csv`
* `lectures.csv`
* `questions.csv`

Install conda env and run jupyter:

```
conda env create -f env/env.yaml
conda activate riiid-acp
jupyter notebook --ip 0.0.0.0 --no-browser --NotebookApp.iopub_msg_rate_limit=10000000000
```

## Run notebooks

Run `01_pre.ipynb` to preprocess data. A minimum 128 GiB RAM is required. This will generate the following pkl files in `input/`

* `input/data_v210101b.pkl`
* `input/meta_v210101b.pkl`

Run `02_train.ipynb` to train the model. The default parameters will produce an AUROC score of 0.812 using 2.5% holdout validation users.

The script supports distributed training on multi-GPU setups. See the instructions at the beginning of the notebook for the exact steps.

Additionally more models can be trained and later ensembled changing the number of encoder/decoder layers, heads, transformer activation, dropout, T-Fixup initialization and optimizer without further changes to the code by simply changing `main`'s default parameters. Output:

* `models/best210105.pth`

Run `03_pre_sub.ipynb` to prepare data for submission. This will cut down user's historic data to the last 500 interactions. Outputs:

* `input/data_500_last_interactions_v210101b.pkl`
* `input/data_attempt_num_v210101b.npy`
* `input/data_attempts_correct_v210101b.npy`

Run `04_pre_validation_set.ipynb` to generate a validation split off of `train.csv` in a format suitable for the inference script (similar to `example_test.csv`). Outputs:

* `input/validation_x_0.025.csv`
* `input/validation_y_0.025.csv`
* `input/validation_submission_0.025.csv`

Copy or hard-link the trained models and the following files into `kaggle_dataset/root/resources`:

```
ln input/data_500_last_interactions_v210101b.pkl kaggle_dataset/root/resources
ln input/data_attempt_num_v210101b.npy kaggle_dataset/root/resources
ln input/data_attempts_correct_v210101b.npy kaggle_dataset/root/resources
ln input/meta_v210101b.pkl kaggle_dataset/root/resources
```

For convenience the following two pre-trained models are provided in `kaggle/root/resources`:

* `210105_0.812154_gelu_e4d4_ep30.pth`
* `210105_0.812534_relu_e3e3.pth`

At this point, `ls -l kaggle_dataset/root/resources` should look like this:

```
-rw-rw-r-- 1 javi javi   79921941 Jan 15 23:06 210105_0.812154_gelu_e4d4_ep30.pth
-rw-rw-r-- 1 javi javi   65210849 Jan 15 23:06 210105_0.812534_relu_e3e3.pth
-rw-rw-r-- 2 javi javi 6811424004 Jan 15 22:54 data_500_last_interactions_v210101b.pkl
-rw-rw-r-- 2 javi javi 6085350128 Jan 15 22:54 data_attempt_num_v210101b.npy
-rw-rw-r-- 2 javi javi 6085350128 Jan 15 22:54 data_attempts_correct_v210101b.npy
-rw-rw-r-- 2 javi javi    1953960 Jan 11 11:12 meta_v210101b.pkl
```

Run `05_inference`.

The default inference script will attempt to ensemble up to two models with dynamic fallback to single model inference in order to fulfill the allocated time budget (8.75h by default).

It should produce an AUROC=0.816 on the default 0.025 user holdout validation set. Sample output:

```
10000it [6:32:02,  2.35s/it, model 1=2482136, model 1+2=2482075, eta=6.581/8.726, auroc (pub)=0.816131, auroc (pvt)=0.816148]
```

The script will also produce a `submission.csv` file with predictions in the same format as `example_sample_submission.csv`.

