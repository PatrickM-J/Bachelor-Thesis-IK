# Bachelor-Thesis-IK
Code and analysis for my bachelor thesis (Information Science, RUG 2026), investigating whether the psychological distance of an argument as described by Construal Level Theory (CLT) is associated with persuasion on the Reddit community ChanegMyView (CMV)

This repository contains the notebooks for the models I used for the classifier comparison, the statistical analysis and data-preparation for the analysis. Not all files regarding the data preparation for the analysis can be found here, but rather on this Github release by one of my peers: https://github.com/KrisHoffmann/Bachelor-Thesis-IK/releases/tag/v1.0-peer-handoff

## What this project does
1. Samples and prepares CMv threads from the Webis-CMV-20 corpus
2. Annotates sentences for salience and psychological distance across all four CLT dimensions
3. Trains/prompts classifiers. 
4. Applies the best selected classifier (from among 9 classifiers, 2 peers also trained/prompted 3 each), then fits a mixed-effects logistic regression comparing far-sentence proportions of delta and non-delta comments, with a classifier-error analysis

## Repository contents
Data preparation scripts:
- segment_fast.py: sentence segmentation of comments.
- select_matched.py:matched case-control selection (1 delta : 3 non-delta per set, within thread).
- flatten_to_sentences.py: flattens comments into the classifier's per-sentence input format.
- aggregate_and_match.py: aggregates sentence predictions to the comment level and builds the final matched analysis dataset.
- make_allowlist.py: thread/comment allowlist generation.
- run_pipeline_pre_classifier.sh: pipeline driver up to the classification step.

SLURM job files (Habrok HPC cluster):
- run_prep.slurm, run_stage3.slurm: corpus-prep files
- run_cascade.slurm: runs the shared BERT cascade classifier
- verify_cascade.slurm: smoke test that the classifier weights load and run

Model Notebooks:
- psych_distance_pipeline.ipynb: the two encoder-based models. Set MODEL_KEY = 'deberta' for microsoft/deberta-v3-base or MODEL_KEY = 'sbert' for sentence-transformers/all-mpnet-base-v2, then run the notebook once per model. Covers both stages (salience + dimension classification).
- gemma_pipeline.ipynb: the generative LLM (google/gemma-3-1b-it), prompted few-shot rather than fine-tuned. Requires a Hugging Face token (Gemma is a gated model) — see Setup below.

Statistical analysis:
- clt_regression.Rmd: the mixed-effects logistic regression, descriptive comparison, odds ratios, collinearity check, and the classifier-error sensitivity analysis. This is the source of truth for the analysis.
- clt_regression.pdf: knitted output of the above file, as a convenience record. Generated from the .Rmd file.

Provenance:
- consort/: JSON counts for documenting the sample funnel (selection counts, aggregate counts, stage-3 counts) for the CONSORT-style flow

Not included:
- Model weights for the shared BERT cascade. These can be found in the peer_handoff found on the Github release from Kris: https://github.com/KrisHoffmann/Bachelor-Thesis-IK/releases/tag/v1.0-peer-handoff
- The raw Webis-CMV-20 corpus: available here: https://zenodo.org/records/3778298
- The segmented comment file and the per-sentence prediction shards: regenerate by running the prep scripts and then the cascade.
