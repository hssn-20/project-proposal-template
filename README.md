
# DNA LLM

A proposal by Hassan Ahmed (@utterly_butterly), Max Sprang(@de_muedi). [Project slides](https://docs.google.com/presentation/d/1VxHHlj-oJJP8QqPrabcQv0-YYwXhQwiZx7HRmBJ3lb4/edit?usp=sharing)

## Abstract

This project aims to: 
- Create a DNA sequence - natural language description dataset of diverse species combining publicly available sequences & research papers.
- Build optimised models for DNA that outperform the current state of the art
- Build DNA models able to generate whole genomes

Possible downstream tasks for Nucleotide-only LLMs could include:
- Encoders for comparisons/classifications
- Base models for finetuned sequence generators & predictors (for example a page generator for a given bacterial genome, whole genomes for de novo organisms or antibiotic classifiers that take into account bacterial plasmid dna) 
- Predicting sequence quality from small sequences (as found in functional genomics) and large sequences as found in sequencing projects for whole genomes. 


## Introduction and Prior Work

Currently most nucleotide based models are species specific, trained on small high quality datasets or large low quality ones. This project hopes to create a high quality dataset of various species of varying sequence lengths.

| Model                         	| Parameters | Dataset                                                                                           	| Dataset size | Context Window | Our improvements                                  	|
| --------------------------------- | ---------- | ----------------------------------------------------------------------------------------------------- | ------------ | -------------- | ----------------------------------------------------- |
| HyenaDNA                      	| 0.44M-6.6M | [Human Reference Genome](https://www.ncbi.nlm.nih.gov/datasets/genome/GCF_000001405.26/)           	| 3.1B     	| 1k-1M      	| Larger dataset + Longer context windows + Multispecie |
| DNABert2                      	| 117M   	| [Multi-species genome](https://arxiv.org/pdf/2306.15006.pdf#table.7)                              	| 32.49B   	| 512        	| Larger dataset + Longer context windows           	|
| Nucleotide Transformer        	| 50M-2.5B   | [Multi-species genome](https://www.biorxiv.org/content/10.1101/2023.01.11.523679v2.full.pdf#A.2)  	| 1T       	| 2024       	| Larger dataset + Longer context windows           	|
| Genomic Pre-trained Network (GPN) | 30M ~  	| [Single Specie(Brassicales)](https://huggingface.co/datasets/songlab/genomes-brassicales-balanced-v1) | 1.23GB   	| 512        	| Larger dataset + Longer context windows + Multispecie           	|
| Gena-LM                       	| 110M   	| Human + Multispecie genome                                                                        	| ~10B     	| 512-4096   	| Larger dataset + Longer context windows           	|
| Grover                        	| 350M   	| [HG19](https://zenodo.org/records/8373053)                                                        	| 2.3GiB   	| 512        	| Larger dataset + Longer context windows + Multispecie           	|

 

The project is different because the models trained will be both of a different scale and scope.
Scope - The models will be trained on a large variety of species unlike most current models which are either trained on humans or a limited range of species.
Scale - There are currently no models trained on context windows above 1M+. This project will hopefully be the first foundational model capable of generating full genomes.

To validate the thesis that scaling of context size improves model results, we [tested](https://huggingface.co/spaces/Hack90/context_and_viral_dna_models) a range of context windows and their effect on results. The results indicate potential for gains from increased context windows. 

To gain intuition with regard to the importance of increasing organism variety, we created a [dashboard](https://huggingface.co/spaces/Hack90/virus_explorer) to display the underlying complexity of genomes. 

## Deliverables

### Datasets

For our initial nucleotide models we have decided to use the RefSeq dataset: 

| Type         	| Tokens | Size	| Huggingface                                                                                                              	|
| ---------------- | ------ | ------- | ---------------------------------------------------------------------------------------------------------------------------- |
| Fungi        	| 18B	| 5.4GB   | [https://huggingface.co/datasets/Hack90/ref_seq_fungi](https://huggingface.co/datasets/Hack90/ref_seq_fungi)             	|
| Bacteria     	| 1368B  | 402GB   |                                                                                                                          	|
| Invertebrate 	| 369B   | 108GB   | [https://huggingface.co/datasets/Hack90/ref_seq_invertebrate](https://huggingface.co/datasets/Hack90/ref_seq_invertebrate)   |
| Mammals      	| 859B   | 252GB   |                                                                                                                          	|
| Vertebrate Other | 867B   | 255GB   |                                                                                                                          	|
| Protozoa     	| 3.7B   | 1GB 	| [https://huggingface.co/datasets/Hack90/ref_seq_protozoa](https://huggingface.co/datasets/Hack90/ref_seq_protozoa)       	|
| Plasmids     	| 6.4B   | 1.89GB  | [https://huggingface.co/datasets/Hack90/ref_seq_plasmid](https://huggingface.co/datasets/Hack90/ref_seq_plasmid)         	|
| Plastids     	| 2.1B   | 0.63GB  | [https://huggingface.co/datasets/Hack90/ref_seq_plastid](https://huggingface.co/datasets/Hack90/ref_seq_plastid)         	|
| Archea       	| 5.4B   | 1.588GB | [https://huggingface.co/datasets/Hack90/ref_seq_archaea](https://huggingface.co/datasets/Hack90/ref_seq_archaea)         	|
| Viruses      	| 0.54B  | 0.161GB | [https://huggingface.co/datasets/Hack90/ref_seq_viral](https://huggingface.co/datasets/Hack90/ref_seq_viral)             	|
| Plants       	| 299B   | 88.2GB  |                                                                                                                          	|
| Mitochondrion	| 0.537B | 0.158GB | [https://huggingface.co/datasets/Hack90/ref_seq_mitochondrion](https://huggingface.co/datasets/Hack90/ref_seq_mitochondrion) |
| Total        	| 3.8T   | 1.12TB  |                                                                                                                          	|
 
On top of the RefSeq dataset, we hope to build a DNA - natural language description dataset.  The main reason is that [in-context learning](https://arxiv.org/abs/2402.12530) is the direct result of parallel structure, hence for us to be able to generate sequences based on natural language input, it's not enough to just fine tune on a question-answer dataset but we must also encode the structure we want our outputs to be in our pre-training step. 

### Models


| Model   	| Parameters | Dataset | Context Window | Input Modality | Output Modality | Experiment Purpose                                   	| GPU hours   |
| ----------- | ---------- | ------- | -------------- | -------------- | --------------- | -------------------------------------------------------- | ----------- |
| Transformer | 14M-310M   | RefSeq  | 64-4096    	| Text - DNA 	| Text - DNA  	| Test scaling laws with regards to DNA sequences      	| 150-1000	|
| DiT     	| 14M-1B 	| RefSeq  | 64-262,144 	| Text - DNA 	| Images      	| Testing whether DNA models can work in longer contexts   | 200 - 4000+ |
| RMKW    	| 14M-310M   | RefSeq  | 64-64,000,000  | Text - DNA 	| Text - DNA  	| Testing architecture usefulness vs standard transformers | 150     	|
| Wavelet 	| 14M-310M   | RefSeq  | 64-64,000,000  | Text - DNA 	| Text - DNA  	| Testing architecture usefulness vs standard transformers | 150     	|

#### Ethical concerns and potential remedies
The model Github wont’ openly available and will be only made available through the discord during development. When open sourcing the project in the end, we’ll have to consider soft walling the weights. Generally, when considering downstream tasks, concerning potential pathogens, we’ll need to employ some biosafety tests.


### Timeline

What is a (rough) timeline for this project?
3-6 months

## Reproducibility

What steps are going to be taken to ensure the project's reproducibility?
We'll be releasing the code written & used in the project, validation/benchmarks as well as the training logs

## Validation

We're building a [validation library](https://github.com/hssn-20/dvq) to compare generated sequences with source sequences. The library will include a wide range of methods to allow us to inspect sequences both visually as well as statistically.

## Failure Case

If our findings are unsatisfactory, do we have an exit plan? Do we have deliverables along the way that we can still provide the community with?
Yes, the datasets on their own would be useful for the wider community. As would the validation library and the model comparisons.
 
## Next Steps

Finetuned generative models for various viruses/virus-like particles e.g. bacteriophages. We can also combine the models created with other projects (eg. Protein Scaling Project, ChemLLM etc.)
