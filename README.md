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

### Re-Run this experiment with the same data of the paper
```sh
$ docker compose up -d
$ docker exec -it collapse bash
$ python main_experiment/exp.py
```
Note: CBCC cannot be run in a non-Windows environment, so please run `exp_cbcc.py` on a Windows PC.

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
We provide a notebook that allows you to reproduce the case studies performed in our paper.
 - 

## Additinal Experiment (Evaluation with Empirical Asymmetric AI Performance)
We provide human and AI response data, the code that generates the AI ​​responses, and a CSV of the raw experiment results.

Implementations of each aggregation method used in the experiments are available below.

| Method                                       | Link                                           |
|----------------------------------------------|------------------------------------------------|
| BDS, HS-DS, CBCC                             | This repository                                |
| DS, GLAD, MACE, MMSR, OneCoin (, MV)         | https://github.com/Toloka/crowd-kit            |
| CATD, LFC, Minmax, PM-CRH, ZenCrowd (, CBCC) | https://github.com/zhydhkcws/crowd_truth_infer |
| LA                                           | https://github.com/yyang318/LA_onepass         |


