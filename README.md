# DNA LLM

A proposal by Hassan Ahmed

## Abstract

DNA-only LLMs that can be used as: encoders for comparisons/classifications, base models for finetuned sequence generators (an antiphage generator) or even quality checks of the sequencing process

## Introduction and Prior Work

Currently most dna based models are species specific, trained on small high quality datasets or large low quality ones. This project hopes to create high quality datasets of variying sequence lengths. 

Examples: 
- https://github.com/AIRI-Institute/GENA_LM
- [Nucleotide transformers](https://github.com/instadeepai/nucleotide-transformer)

This means the models trained will be of different scale and scope. Scope - The models will be trained on a large variety of species unlike most current models which are either trained on humans or a limited range of species. Scale- There are currently no models trained on context windows above 10k+. This project will mean the first foundational model capable of generating full genome. 


## Deliverables

### Datasets
We plan to create datasets of varying sequences lengths which are deduplicated to varying levels of similarity e.g. sequence length 10k, max similarity 50%, kmer 7
The largest dataset will be 990GB for the largest dataset i.e. all the sequences on Genbank. This dataset has already been scraped from NCBI and uploaded to Huggingface, all thats left to deduplicate/process it for the other smaller datasets. To do that, we'll need a machine with roughly 1TB of RAM.

### Models

If applicable, does the project aim to release more than one model? 
Yes, we hope to release models of varying sizes trained on sequences of varying simliarities/lengths
What would be the input modality?
Nucleotides encoded as text
What about the output modality? 
Depends on the model type. The decoder models will generate nucleotides, the encoder models will generate a vector/embedding. 
How large are the models that the project aims to release?
The sizes of the models will depend on our datasets.

## Resources

### Requirements

What kinds of resources (e.g. GPU hours, RAM, storage) are needed to complete the project?
- Roughly 100-1000GPU hours(dependant on the dataset sizes)
- 1TB of RAM for the processing stage
- 5 TB of storage on S3

### Timeline

What is a (rough) timeline for this project?
- 1 week to deduplicate/prepare the dataset
- 1-2 weeks to train the model
- 1 week to analyse the results

## Reproducibility

What steps are going to be taken to ensure the project's reproducibility?
We'll be releasing the code written & used in the project + the training logs

## Failure Case

If our findings are unsatisfactory, do we have an exit plan? Do we have deliverables along the way that we can still provide the community with?
Yes, the datasets on their own would be useful for the wider community

## Next Steps

If the project is successfully completed, are there any obvious next steps?
Finetuned generative models for various viruses/virus-like particles e.g. bacteriophages etc.

