# Awesome-Multilingual-LLMs-Papers

This repository contains list of papers according to [our survey](https://arxiv.org/pdf/2310.19736.pdf):

<p align="center"><strong>Multilingual Large Language Models: A Systematic Survey</strong></p>

<p align="center">Shaolin Zhu<sup>1</sup>,   Supryadi<sup>1</sup>,   Shaoyang Xu<sup>1</sup>,   Haoran Sun<sup>1</sup>,   Leiyu Pan<sup>1</sup>,   Menglong Cui<sup>1</sup>, </p>

<p align="center">Jiangcun Du<sup>1</sup>,   Renren Jin<sup>1</sup>,   António Branco<sup>2</sup>†,   Deyi Xiong<sup>1</sup>†*</p>

<p align="center"><sup>1</sup>TJUNLP Lab, College of Intelligence and Computing, Tianjin University</p>

<p align="center"><sup>2</sup>NLX, Department of Informatics, University of Lisbon</p>

<p align="center">(*: Corresponding author, †: Advisory role)</p>

<div align=center>
    <img src="./assets/fig.png" style="zoom:30%"/>
</div>

## Papers

### Multilingual Evaluation

#### Tokenizer Evaluation

1. **"How Good is Your Tokenizer? On the Monolingual Performance of Multilingual Language Models"**. 
   
    *Phillip Rust and Jonas Pfeiffer et al.* ACL-IJCNLP 2021. [[Paper](https://aclanthology.org/2021.acl-long.243.pdf)] [[GitHub](https://github.com/Adapter-Hub/hgiyt)] 
   
2. **"ByT5: Towards a Token-Free Future with Pre-trained Byte-to-Byte Models"**. 

    *Linting Xue, Aditya Barua, Noah Constant, and Rami Al-Rfou et al.* TACL 2023. [[Paper](https://aclanthology.org/2022.tacl-1.17.pdf)] [[GitHub](https://github.com/google-research/byt5)]

3. **"Language Model Tokenizers Introduce Unfairness Between Languages"**. 

    *Aleksandar Petrov et al.* NeurIPS 2023. [[Paper](https://arxiv.org/pdf/2305.15425)] [[GitHub](https://github.com/AleksandarPetrov/tokenization-fairness)]

4. **"Tokenizer Choice For LLM Training: Negligible or Crucial?"**. 

    *Mehdi Ali, Michael Fromm, and Klaudia Thellmann et al.* NAACL (Findings) 2024. [[Paper](https://aclanthology.org/2024.findings-naacl.247.pdf)]

#### Multilingual Evaluation Benchmarks and Datasets

##### Multilingual Holistic Evaluation

1. **"MEGA: Multilingual Evaluation of Generative AI"**. 
   
    *Kabir Ahuja et al.* EMNLP 2023. [[Paper](https://aclanthology.org/2023.emnlp-main.258.pdf)] [[GitHub](https://github.com/microsoft/Multilingual-Evaluation-of-Generative-AI-MEGA)] 
   
2. **"MEGAVERSE: Benchmarking Large Language Models Across Languages, Modalities, Models and Tasks"**. 

   *Sanchit Ahuja et al.* arXiv 2024. [[Paper](https://arxiv.org/pdf/2311.07463)]

3. **"ByT5: Towards a Token-Free Future with Pre-trained Byte-to-Byte Models"**. 

    *Viet Dac Lai, Nghia Trung Ngo, and Amir Pouran Ben Veyseh et al.* EMNLP (Findings) 2023. [[Paper](https://aclanthology.org/2023.findings-emnlp.878.pdf)]

##### Multilingual Task-Specific Evaluation

###### Translation Evaluation

1. **"Investigating the Translation Performance of a Large Multilingual Language Model: the Case of BLOOM"**. 
   
    *Rachel Bawden et al.* EAMT 2023. [[Paper](https://aclanthology.org/2023.eamt-1.16.pdf)] [[GitHub](https://github.com/rbawden/mt-bigscience)] 
   
2. **"Multilingual Machine Translation with Large Language Models: Empirical Results and Analysis"**. 

    *Wenhao Zhu et al.* NAACL (Findings) 2024. [[Paper](https://aclanthology.org/2024.findings-naacl.176.pdf)] [[GitHub](https://github.com/NJUNLP/MMT-LLM)]

###### Question Answering Evaluation

1. **"M3Exam: A Multilingual, Multimodal, Multilevel Benchmark for Examining Large Language Models"**. 
   
    *Wenxuan Zhang et al.* NeurIPS 2023. [[Paper](https://arxiv.org/pdf/2306.05179)] [[GitHub](https://github.com/DAMO-NLP-SG/M3Exam)] 
   
2. **"Evaluating the Elementary Multilingual Capabilities of Large Language Models with MULTIQ"**. 

    *Carolin Holtermann and Paul Röttger et al.* ACL (Findings) 2024. [[Paper](https://aclanthology.org/2024.findings-acl.265.pdf)] [[GitHub](https://github.com/paul-rottger/multiq)]

###### Summarization Evaluation

1. **"SEAHORSE: A Multilingual, Multifaceted Dataset for Summarization Evaluation"**. 
   
    *Elizabeth Clark et al.* EMNLP 2023. [[Paper](https://aclanthology.org/2023.emnlp-main.584.pdf)] [[GitHub](https://github.com/google-research-datasets/seahorse)] 

###### Dialogue Evaluation

1. **"xDial-Eval: A Multilingual Open-Domain Dialogue Evaluation Benchmark"**. 
   
    *Chen Zhang et al.* EMNLP (Findings) 2023. [[Paper](https://aclanthology.org/2023.findings-emnlp.371.pdf)] [[GitHub](https://github.com/e0397123/xDial-Eval)] 
   
2. **"MEEP: Is this Engaging? Prompting Large Language Models for Dialogue Evaluation in Multilingual Settings"**. 

    *Amila Ferron et al.* EMNLP (Findings) 2023. [[Paper](https://aclanthology.org/2023.findings-emnlp.137.pdf)] [[GitHub](https://github.com/PortNLP/MEEP)]

##### Multilingual Alignment Evaluation

###### Multilingual Ethics Evaluation

1. **"Ethical Reasoning and Moral Value Alignment of LLMs Depend on the Language we Prompt them in"**. 
   
    *Utkarsh Agarwal, Kumar Tanmay, and Aditi Khandelwal et al.* LREC-COLING 2024. [[Paper](https://aclanthology.org/2024.lrec-main.560.pdf)]

###### Multilingual Toxicity Evaluation

1. **"RTP-LX: Can LLMs Evaluate Toxicity in Multilingual Scenarios?"**. 
   
    *Adrian de Wynter et al.* arXiv 2024. [[Paper](https://arxiv.org/pdf/2404.14397)] [[GitHub](https://github.com/microsoft/RTP-LX)] 

2. **"PolygloToxicityPrompts: Multilingual Evaluation of Neural Toxic Degeneration in Large Language Models"**. 
   
    *Devansh Jain and Priyanshu Kumar et al.* COLM 2024. [[Paper](https://arxiv.org/pdf/2405.09373)] [[GitHub](https://github.com/kpriyanshu256/polyglo-toxicity-prompts)] 

###### Multilingual Bias Evaluation

1. **"On Evaluating and Mitigating Gender Biases in Multilingual Settings"**. 
   
    *Aniket Vashishtha and Kabir Ahuja et al.* ACL (Findings) 2021. [[Paper](https://aclanthology.org/2023.findings-acl.21.pdf)] [[GitHub](https://github.com/microsoft/MultilingualBiasEvaluation)] 

##### Multilingual Safety Evaluation

###### Multilingual Safety Benchmarks

1. **"All Languages Matter: On the Multilingual Safety of LLMs"**. 
   
    *Wenxuan Wang et al.* ACL (Findings) 2024. [[Paper](https://aclanthology.org/2024.findings-acl.349.pdf)] [[GitHub](https://github.com/Jarviswang94/Multilingual_safety_benchmark)] 

###### Multilingual Jailbreaking/Red-Teaming

1. **"Low-Resource Languages Jailbreak GPT-4"**. 
   
    *Zheng-Xin Yong et al.* NeurIPS (Workshop) 2023. [[Paper](https://arxiv.org/pdf/2310.02446)]

2. **"Multilingual Jailbreak Challenges in Large Language Models"**. 
   
    *Yue Deng et al.* ICLR 2024. [[Paper](https://arxiv.org/pdf/2310.06474)] [[GitHub](https://github.com/DAMO-NLP-SG/multilingual-safety-for-LLMs)] 

3. **"A Cross-Language Investigation into Jailbreak Attacks in Large Language Models"**. 
   
    *Jie Li et al.* arXiv 2024. [[Paper](https://arxiv.org/pdf/2401.16765)]

#### Multilingualism Evaluation

1. **"How Vocabulary Sharing Facilitates Multilingualism in LLaMA?"**. 
   
    *Fei Yuan et al.* ACL (Findings) 2024. [[Paper](https://aclanthology.org/2024.findings-acl.721.pdf)] [[GitHub](https://github.com/CONE-MT/Vocabulary-Sharing-Facilitates-Multilingualism)] 

#### MLLMs as Multilingual Evaluator

1. **"Are Large Language Model-based Evaluators the Solution to Scaling Up Multilingual Evaluation?"**. 
   
    *Rishav Hada et al.* EACL (Findings) 2024. [[Paper](https://aclanthology.org/2024.findings-eacl.71.pdf)] [[GitHub](https://github.com/microsoft/METAL-Towards-Multilingual-Meta-Evaluation)] 

2. **"METAL: Towards Multilingual Meta-Evaluation"**. 
   
    *Rishav Hada and Varun Gumma et al.* NAACL (Findings) 2024. [[Paper](https://aclanthology.org/2024.findings-naacl.148.pdf)] [[GitHub](https://github.com/microsoft/METAL-Towards-Multilingual-Meta-Evaluation)] 

