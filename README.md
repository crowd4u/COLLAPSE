# On Aggregating Labels from Humans and AIs with Asymmetric Performance

This repository provides supplementary material for the paper "On Aggregating Labels from Humans and AIs with Asymmetric Performance," currently under peer review.

## Appendix of the paper
Please see `appendix.pdf`

## Methods
The `methods` folder contains code for BDS, HS-DS, and CBCC in Crowd-Kit format.

Some of the code uses Crowd-Kit code under license. We would like to express our gratitude to the Crowd-Kit team.
Additionally, we have made minimal modifications to the original CBCC code by the authors and included it in this repository. We would also like to express our gratitude to the authors of the CBCC code.

## Main Experiment
We provide a Docker container for easy reproduction.

FYI: You can use jupyter lab on `http://localhost:8008/` when running this container (Please check the token following command `docker exec -it collapse jupyter server list`).

### Re-Run this experiment with the same data of the paper
```sh
$ docker compose up -d
$ docker exec -it collapse bash
$ python main_experiment/exp.py
```
Note: CBCC cannot be run in a non-Windows environment, so please run `exp_cbcc.py` on a Windows PC. We used `Python 3.11.3` with the libraries listed in `requirements_python3_win.txt`.

However, data containing only human worker results (with `num_ai=0`) cannot be generated using this method. Please run `notebooks/human_only_results.ipynb` and `notebooks/human_only_results_cbcc.ipynb`.

### Data Preprocessing
For reproducibility, we provide the code used to process and generate human and AI responses in the `preprocessing` folder.

If you want to reproduce the experiment from the data generation process, you can regenerate the data using the following command.
```sh
$ docker compose up -d
$ docker exec -it collapse bash
$ cd main_experiment/preprocessing
$ python generate_human_responses.py
$ python generate_ai_responses.py
```

The `preprocessing/raw_datasets` directory contains the raw datasets before redundancy adjustment.
These data were copied from the following publicly available data sources (excluding `Tiny`).

 - `Dog` : https://github.com/zhydhkcws/crowd_truth_infer/tree/master/datasets/s4_Dog%20data
 - `Face` : https://github.com/zhydhkcws/crowd_truth_infer/tree/master/datasets/s4_Face%20Sentiment%20Identification
  - `Tiny` : We firstly published online.
  - `Adult` : It was available at https://toloka.ai/datasets/ as `Toloka Aggregation Features`, but is no longer distributed.

### Visualize Results
We obtained a total of 38,125 lines of experimental results and provide a visualization tool to analyze them.

```sh
$ docker compose up -d
$ docker exec -it collapse bash
$ cd main_experiment/streamlit
$ streamlit run app.py --server.port 9999
```

Please visit http://localhost:9009/ to use this app.

### Case Studies
We provide a notebook that allows you to re-run the case studies performed in our paper.

 - Confusion Matrices (Figure 6) : `notebooks\cm_analysis.ipynb`
 - Communities of CBCC (Figure 7): `notebooks\CBCC_analysis.ipynb`

## Additinal Experiment (Evaluation with Empirical Asymmetric AI Performance)
Our experimental results can be found in `results`.

### Methods
The `additional_methods` folder contains implementations of various aggregation methods, copied from the following repositories with minimal modifications.

| Method                                       | Link                                           |
|----------------------------------------------|------------------------------------------------|
| CATD, LFC, Minmax, PM-CRH, ZC                | https://github.com/zhydhkcws/crowd_truth_infer |
| LA                                           | https://github.com/yyang318/LA_onepass         |

### Human and AI Data
The human data is `human_responses_with_gt.csv`

The AI's response data is stored in `ai_responses`.

If you need to regenerate the AI's response data, please run the following command.
```sh
$ docker compose up -d
$ docker exec -it collapse bash
$ cd additinal_experiment
$ python generate_ai_responses.py
```

### Re-Run the additinal Experiment
Each method uses a different environment, notebook, and script. You will need to properly configure the file paths to match your execution environment.

#### EMDS, OneCoin, GLAD, MACE, MMSR (, MV)
Run `notebooks\evaluate_crowdkit.ipynb` in the container.

#### CBCC
Run `notebooks\evaluate_CBCC.ipynb` on Windows computer.

#### CATD, LFC, PM-CRH, ZC, LA, Minmax

1. Run `notebooks\transform_to_truth_infer_format.ipynb` in the container.
2. Set up a Python 2.7.13 execution environment on Windows PC and activate the venv.
3. Install the libraries listed in `requirements_python27_win.txt` (in the project root).
4. Run `scripts/***.bat` in the `scripts` folder except for `LA.bat`.
5. Deactivate the python2 venv.
5. Run `scripts/LA.bat` in the `scripts` folder in the python3 windows venv for CBCC.
6. For running Minmax, you have to use MATLAB (paid) or MATLAB online (free).
7. Run `additinal_methods/l_minimax-s/prepare.m` using MATLAB with `truth_infer_0_.csv`, and `truth_infer_5_.csv` and `truth_infer_10_.csv`.
8. Using `notebooks/evaluate_truth_infer.ipynb`, calculate the scores in the container.

#### BDS, HS-DS
Run `notebooks\evaluate_bds_hsds.ipynb` in the container.








