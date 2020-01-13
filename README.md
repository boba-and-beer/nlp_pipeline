# nlp_pipeline
This repo is my nlp pipeline that I am using for my NLP competitions on Kaggle. 

Ideally, this will run similar to sklearn's fit_predict with a few edits on 
features that are generated.

It uses the following: 
- wandb to record experimentation procedures 

Currently supports the following competitions formats:
- Supports text multi-labelling 

Built on:
- FastAI
- Pytorch 
- transformers library for pre-built models

Assumes there is CUDA installed with Pytorch on computer.