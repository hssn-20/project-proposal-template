
# DNA LLM

A proposal by Hassan Ahmed (@utterly_butterly), Max Sprang(@de_muedi). [Project slides](https://docs.google.com/presentation/d/1VxHHlj-oJJP8QqPrabcQv0-YYwXhQwiZx7HRmBJ3lb4/edit?usp=sharing)


## Abstract
Life is abundant, its everywhere. Yet the chance of an echoli arising spontaneously is:
```math
10^{{-10}^{10}} 
```
A tiny number. Somehow DNA enables that to happen day in, day out. DNA stands for deoxyribonucleic acid, it is the instruction set which creates life as well as the script it’s written in. At its simplest, it can be thought of as a sequence of letters drawn from a four letter alphabet, [A,T,C,G], structured as a base-paired helix. These simple chemicals, Adenine, Thymine, Guanine and Cytosine encode all the complexity that we see.

However, we mustn’t think of genomes as a simple sequence of base pairs, our concept of the genome must also include the physical properties of each bond as well as their first/second/third/n-order effects of base pairs on each other.

Be that as it may, all we have for most genomes are just sequences. Some of which are not even aligned. So how can we build model that is able to generate out-of-distribution (i.e. not in the training data) sequences when our training data is so poor? How do we trust and verify the models that we create? Or at an even more basic level, where does a DNA foundation model sit in the bioinformatics toolkit?

With this project, we aim to answer those questions and more. 

## Project Objectives

This project aims to: 
- Create a DNA-sequence:natural-language-description dataset of diverse species combining publicly available sequences with their associated texts/research papers. 
- Build homologicaly/topologically optimised DNA models that outperform the current state of the art
- Build DNA models capable of generating biologically viable whole genomes

Potential downstream applications for Nucleotide-only Language Models (LLMs) include:
- Encoders for sequence comparisons and classifications
- Base models for fine-tuned sequence generators and predictors, such as:
  - Bacteria specific phage generators
  - Whole genome generators for de novo organisms
  - Antibiotic resistance classifiers based on bacterial plasmid DNA
- Sequence quality predictors for both small sequences (as found in functional genomics) and large sequences (as found in whole-genome sequencing projects)


## Introduction and Prior Work
Currently, most nucleotide-based models are either: phylum-specific, trained on either small high-quality datasets or large low-quality ones. Even [Evo](https://github.com/evo-design/evo?tab=readme-ov-file) does not take into account the topological diversity of the training data. This project aims to create a high-quality dataset comprising various species with varying sequence lengths

| Model                             | Parameters | Dataset                                                                                               | Dataset size | Context Window | Our improvements                                      | Code                                                                                                                                                 | Weights                                                                                                                                                                    | Huggingface                                                                                                                                                                | Model Type                                                                                                                                                                 |
| --------------------------------- | ---------- | ----------------------------------------------------------------------------------------------------- | ------------ | -------------- | ----------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| HyenaDNA                          | 0.44M-6.6M | [Human Refrence Genome](https://www.ncbi.nlm.nih.gov/datasets/genome/GCF_000001405.26/)               | 3.1B         | 1k-1M          | Larger dataset + Longer context windows + Multispecie | [HazyResearch/hyena-dna](https://github.com/HazyResearch/hyena-dna)                                                                                  | [LongSafari](https://huggingface.co/LongSafari)                                                                                                                            | [LongSafari](https://huggingface.co/LongSafari)                                                                                                                            | SSM                                                                                                                                                                        |
| DNABert2                          | 117M       | [Multi-species genome](https://arxiv.org/pdf/2306.15006.pdf#table.7)                                  | 32.49B       | 512            | Larger dataset + Longer context windows               | [Zhihan1996/DNABERT_2](https://github.com/Zhihan1996/DNABERT_2)                                                                                      | [DNABERT-2-117M](https://huggingface.co/zhihan1996/DNABERT-2-117M)                                                                                                         | [DNABERT-2-117M](https://huggingface.co/zhihan1996/DNABERT-2-117M)                                                                                                         | Encoder - Transformer                                                                                                                                                      |
| Nucleotide Transformer            | 50M-2.5B   | [Multi-species genome](https://www.biorxiv.org/content/10.1101/2023.01.11.523679v2.full.pdf#A.2)      | 1T           | 2024           | Larger dataset + Longer context windows               | [Nucleotide Transformer: Building and Evaluating Robust Foundation Models for Human Genomics](https://github.com/instadeepai/nucleotide-transformer) | [InstaDeepAI (InstaDeep Ltd)](https://huggingface.co/InstaDeepAI)                                                                                                          | [https://huggingface.co/InstaDeepAI](https://huggingface.co/InstaDeepAI)                                                                                                   | Decoder - Transformer                                                                                                                                                      |
| Genomic Pre-trained Network (GPN) | 30M ~      | [Single Specie(Brassicales)](https://huggingface.co/datasets/songlab/genomes-brassicales-balanced-v1) | 1.23GB       | 512            | Larger dataset + Longer context windows               | [songlab-cal/gpn](https://github.com/songlab-cal/gpn)                                                                                                | No weights released                                                                                                                                                        |                                                                                                                                                                            | Encoder - Transformer                                                                                                                                                      |
| Gena-LM                           | 110M       | Human + Multispecis genome                                                                            | ~10B         | 512-4096       | Larger dataset + Longer context windows               | [GitHub - AIRI-Institute/GENA_LM](https://github.com/AIRI-Institute/GENA_LM)                                                                         | [AIRI - Artificial Intelligence Research Institute](https://huggingface.co/AIRI-Institute)                                                                                 | [AIRI - Artificial Intelligence Research Institute](https://huggingface.co/AIRI-Institute)                                                                                 | Encoder - Transformer                                                                                                                                                      |
| Grover                            | 350M       | [HG19](https://zenodo.org/records/8373053)                                                            | 2.3GiB       | 512            | Larger dataset + Longer context windows               | [GROVER pretrained DNA language model of the human genome.](https://zenodo.org/records/8373117)                                                      | [https://zenodo.org/records/8373117](https://zenodo.org/records/8373117)                                                                                                   | N/A                                                                                                                                                                        | Encoder - Transformer                                                                                                                                                      |
| EVO                               | 7B         | [OpenGenome](https://www.biorxiv.org/content/10.1101/2024.02.27.582234v1.full.pdf#appendix.B)         | 250B         | 8k-131k        | Larger dataset + Longer context windows + Multispecie | [EVO](https://github.com/evo-design/evo?tab=readme-ov-file)                                                                                          | [togethercomputer/evo-1-131k-base](https://huggingface.co/togethercomputer/evo-1-131k-base)                                                                                | [togethercomputer/evo-1-131k-base](https://huggingface.co/togethercomputer/evo-1-131k-base)                                                                                | SSM - Transformer Hybrid                                                                                                                                                   |
| Caduceus                          | 30M        | [Human Refrence Genome](https://www.ncbi.nlm.nih.gov/datasets/genome/GCF_000001405.26/)               | 3.1B         | 131k           | Larger dataset + Longer context windows + Multispecie | [kuleshov-group/caduceus](https://github.com/kuleshov-group/caduceus?tab=readme-ov-file)                       | [https://huggingface.co/collections/kuleshov-group/caduceus-65dcb89b4f54e416ef61c350](https://huggingface.co/collections/kuleshov-group/caduceus-65dcb89b4f54e416ef61c350) | [kuleshov-group/caduceus](https://huggingface.co/collections/kuleshov-group/caduceus-65dcb89b4f54e416ef61c350) | [kuleshov-group/caduceus](https://huggingface.co/collections/kuleshov-group/caduceus-65dcb89b4f54e416ef61c350) |

The project is different because the models trained will be both of a different scale and scope.
- Scope - The models will be: trained on a large variety of species unlike most current models which are either trained on humans or a limited range of species, test a variety of loss functions and compare different model architectures. 
- Scale - There are currently no models trained on context windows above 1M+. This project will create the first foundational model capable of generating full genomes.

To validate the thesis that scaling of context size improves model results, we [tested](https://huggingface.co/spaces/Hack90/context_and_viral_dna_models) a range of context windows and their effect on results. The results indicate potential for gains from increased context windows. 

To gain intuition with regard to the importance of increasing organism variety, we created a [dashboard](https://huggingface.co/spaces/Hack90/virus_explorer) to display the underlying complexity of genomes. 

## Deliverables

### Datasets

For our initial nucleotide models, we will use the RefSeq dataset: 

| Type         	| Tokens | Size	| Huggingface                                                                                                              	|
| ---------------- | ------ | ------- | ---------------------------------------------------------------------------------------------------------------------------- |
| Fungi        	| 18B	| 5.4GB   | [Fungi Genomes](https://huggingface.co/datasets/Hack90/ref_seq_fungi)             	|
| Bacteria     	| 1368B  | 402GB   |    [Bacteria Genomes Part 1](https://huggingface.co/datasets/Hack90/ref_seq_bacteria_part_1)  [Bacteria Genomes Part 2](https://huggingface.co/datasets/Hack90/ref_seq_bacteria_part_2) [Bacteria Genomes Part 3](https://huggingface.co/datasets/Hack90/ref_seq_bacteria_part_3)  [Bacteria Genomes Part 4](https://huggingface.co/datasets/Hack90/ref_seq_bacteria_part_4)                                                                                                                        	|
| Invertebrate 	| 369B   | 108GB   | [Invertebrate Genomes](https://huggingface.co/datasets/Hack90/ref_seq_invertebrate)   |
| Mammals      	| 859B   | 252GB   |    [Mammal Genomes Part 1](https://huggingface.co/datasets/Hack90/ref_seq_mammals_part_1) [Mammal Genomes Part 2](https://huggingface.co/datasets/Hack90/ref_seq_mammals_part_2)                                                                                                                       	|
| Vertebrate Other | 867B   | 255GB   |  [Non-mammal Vertebrate Genomes Part 1](https://huggingface.co/datasets/Hack90/ref_seq_vertebrate_non_mammal_part_1) [Non-mammal Vertebrate Genomes Part 2](https://huggingface.co/datasets/Hack90/ref_seq_vertebrate_non_mammal_part_2)                                                                                                                        	|
| Protozoa     	| 3.7B   | 1GB 	| [Protozoa Genomes](https://huggingface.co/datasets/Hack90/ref_seq_protozoa)       	|
| Plasmids     	| 6.4B   | 1.89GB  | [Plasmid Genomes](https://huggingface.co/datasets/Hack90/ref_seq_plasmid)         	|
| Plastids     	| 2.1B   | 0.63GB  | [Plastid Genomes](https://huggingface.co/datasets/Hack90/ref_seq_plastid)         	|
| Archea       	| 5.4B   | 1.588GB | [Archea Genomes](https://huggingface.co/datasets/Hack90/ref_seq_archaea)         	|
| Viruses      	| 0.54B  | 0.161GB | [Viral Genomes](https://huggingface.co/datasets/Hack90/ref_seq_viral)             	|
| Plants       	| 299B   | 88.2GB  | [Plant Genomes](https://huggingface.co/datasets/Hack90/ref_seq_plants)                                                                                                                          	|
| Mitochondrion	| 0.537B | 0.158GB | [Mitochondrion Genomes](https://huggingface.co/datasets/Hack90/ref_seq_mitochondrion) |
| Total        	| 3.8T   | 1.12TB  |                                                                                                                          	|
 
In addition to the RefSeq dataset, we will create a DNA-natural language description dataset. The main reason for this is that [in-context learning](https://arxiv.org/abs/2402.12530) is a direct result of parallel structure. Therefore, to generate sequences based on natural language input, it is not sufficient to fine-tune the model on a question-answer dataset alone. Instead, we must also encode the desired output structure during the pre-training step.

### Models


| Model   	| Parameters | Dataset | Context Window | Input Modality | Output Modality | Experiment Purpose                                   	| GPU hours   |
| ----------- | ---------- | ------- | -------------- | -------------- | --------------- | -------------------------------------------------------- | ----------- |
| Transformer | 14M-310M   | RefSeq  | 64-4096    	| Text - DNA 	| Text - DNA  	| Test scaling laws with regards to DNA sequences      	| 150-1000	|
| DiT     	| 14M-1B 	| RefSeq  | 64-262,144 	| Text - DNA 	| Images      	| Testing whether DNA models can work in longer contexts   | 200 - 4000+ |
| Based(Hybrid SSM/Transformer model)    	| 14M-310M   | RefSeq  | 64-64,000,000  | Text - DNA 	| Text - DNA  	| Testing architecture usefulness vs standard transformers | 150     	|
| RL Model 	| 14M-310M   | RefSeq  | 64-64,000,000  | Text - DNA 	| Text - DNA  	| Testing architecture usefulness vs standard transformers | 150     	|

#### Ethical concerns and potential remedies
The model and its code will not be openly available on GitHub during development and will only be accessible through the project's Discord server. When open-sourcing the project upon completion, we will need to consider implementing access restrictions to the model's weights. Furthermore, when exploring downstream tasks involving potential pathogens, it will be crucial to employ appropriate biosafety tests and adhere to relevant regulations and guidelines.


### Timeline

We anticipate that the initial experiments will take approximately 3-6 months to complete. The following is an overview of our planned initial experiments; however, these tasks are subject to change based on the results we obtain.

| Task                         | Questions being answered                                                                                                                                                                                                                         | Dev Time | Training Time | GPU required | Parallelisable | Colab compatible | Testing For                                                       | Controlling For | Metric   | Blog/Paper |
| ---------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | -------- | ------------- | ------------ | -------------- | ---------------- | ----------------------------------------------------------------- | --------------- | -------- | ---------- |
| Optimal Tokenizer + Loss Function            | Whats the best tokenizer method in this context? Whats our optimal kmer length? What are the trade offs with regards to tokenization in this context? What do we lose, what do we gain, information wise? Similar questions with regards to loss functions. Eg.g does using persistant homology help or hinder a model?                                                         | 6 hrs    | 15 hrs        | Yes          | Yes            | Yes              | k-mer, vocab size, tokenizer method, loss function                                                 | ~ model size    | Loss     | Blog 1     |
| Inter-Phylum/Class Structure | Whats the underlying structure of each phylum? Can we measure it? What are the weaknesses/strengths of each measure? What are the distances within phylums? Most common kmer at each length?                                                     | 10 hrs   | 50 hrs        | No           | Yes            | Yes              | persistant homology, kl-divergence, colorsquare, wens method, CGR | phylum/class    | Distance | Paper 1    |
| Intra-Phylum/Class Structure | What are distances between phylums,both local and global? Whats the effect of genome length on those distances? Looking at the full dataset can we measure biological viability, if so, how?                                                     | 0.5 hrs  | 60 hrs        | No           | Kinda          | Yes              | persistant homology, kl-divergence, colorsquare, wens method, CGR | Sample Size     | Distance | Paper 1    |
| Transformer Scaling          | As we increase context windows and model params, does loss go down in commiserate manner?                                                                                                                                                        | 0.5 hrs  | 50 hrs        | Yes          | Yes            | Yes              | Context Size, Model Params                                        | Training Data   | Loss     | Paper 2    |
| State Space Model Scaling    | As we increase context windows and model params, does loss go down in commiserate manner?                                                                                                                                                        | 3 hrs    | 50 hrs        | Yes          | Yes            | Yes              | Context Size, Model Params                                        | Training Data   | Loss     | Paper 2    |
| RL-based model scaling  | As we increase context windows and model params, does loss go down in commiserate manner?                                                                                                                                                        | 4 hrs    | 50 hrs        | Yes          | Yes            | Yes              | Context Size, Model Params                                        | Training Data   | Loss     | Paper 2    |
| DiT Scaling                  | As we increase context windows and model params, does loss go down in commiserate manner?                                                                                                                                                        | 2 hrs    | 50 hrs        | Yes          | Yes            | Yes              | Context Size, Model Params                                        | Training Data   | Loss     | Paper 2    |
| Model Selection              | Which of the different models trained is best suited to our domain? How do we define best? Do the models capture the complexity of DNA sequences? Are the models capable of in-context learning? Are the generated sequences biological viabile? | 15 hrs   | 0 hrs         | No           | No             | No               | persistant homology, kl-divergence, colorsquare, wens method, CGR | Test Data       | Distance | Paper 2    |
| Model Scaling                | Can scaled models generate OOD dna sequences or are they just memorising the training data? Can we interpolate between sequences? How much more biologically viable are generated sequences as we scale?                                         | 8 hrs    | ~1000 hrs      | Yes          | No             | No               | persistant homology, kl-divergence, colorsquare, wens method, CGR | OOD Eval Set    | Distance | Paper 3    |

## Reproducibility

What steps are going to be taken to ensure the project's reproducibility?
We'll be releasing the code written & used in the project, validation tests, benchmarks as well as our training logs. 

## Validation

We're building a [validation library](https://github.com/hssn-20/dvq) to compare generated sequences with source sequences. The library will include a wide range of methods to allow us to inspect sequences both visually as well as statistically.

## Failure Case

If our findings are unsatisfactory, do we have an exit plan? Do we have deliverables along the way that we can still provide the community with?
Yes, the datasets on their own would be useful for the wider community. As would the validation library and the model comparisons.
 
## Next Steps

Finetuned generative models for various viruses/virus-like particles e.g. bacteriophages. We can also combine the models created with other projects (eg. Protein Scaling Project, ChemLLM etc.)
