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

## Interpretability

### Interpretability of Multilingual Capabilities

#### Model-Wide Interpretation

1. "How do Large Language Models Handle Multilingualism?".
   
   Zhao Y, Zhang W, Chen G, et al. arXiv 2024. [[Paper](https://arxiv.org/abs/2402.18815)]

2. "Do Llamas Work in English? On the Latent Language of Multilingual Transformers".
   
   Wendler C, Veselovsky V, Monea G, et al. ACL 2024. [[Paper](https://arxiv.org/abs/2402.10588)]

3. "Analyzing the Mono- and Cross-Lingual Pretraining Dynamics of Multilingual Language Models".
   
   Blevins T, Gonen H, Zettlemoyer L. EMNLP 2022. [[Paper](https://aclanthology.org/2022.emnlp-main.234/)]

#### Component-Based Interpretation

1. "Unveiling Multilinguality in Transformer Models: Exploring Language Specificity in Feed-Forward Networks".
   
   Bhattacharya S, Bojar O. Proceedings of the 6th BlackboxNLP Workshop: Analyzing and Interpreting Neural Networks for NLP 2023. [[Paper](https://aclanthology.org/2023.blackboxnlp-1.9/)]

#### Neuron-Level Interpretation

1. "Unveiling Linguistic Regions in Large Language Models".
   
   Zhang Z, Zhao J, Zhang Q, et al. ACL 2024. [[Paper](https://arxiv.org/abs/2402.14700)]

2. "Language-Specific Neurons: The Key to Multilingual Capabilities in Large Language Models".
   
   Tang T, Luo W, Huang H, et al. ACL 2024. [[Paper](https://arxiv.org/abs/2402.16438)]

3. "Unraveling Babel: Exploring Multilingual Activation Patterns of LLMs and Their Applications".
   
   Liu W, Xu Y, Xu H, et al. EMNLP 2024. [[Paper](https://arxiv.org/abs/2402.16367)]

4. "On the Multilingual Ability of Decoder-based Pre-trained Language Models: Finding and Controlling Language-Specific Neurons".
   
   Kojima T, Okimura I, Iwasawa Y, et al. NAACL 2024. [[Paper](https://arxiv.org/abs/2404.02431)]

#### Representation-Driven Interpretation

1. "The Geometry of Multilingual Language Model Representations".
   
   Chang T, Tu Z, Bergen B. EMNLP 2022. [[Paper](https://aclanthology.org/2022.emnlp-main.9/)]

2. "Language-agnostic Representation from Multilingual Sentence Encoders for Cross-lingual Similarity Estimation".
   
   Tiyajamorn N, Kajiwara T, Arase Y, et al. EMNLP 2021. [[Paper](https://aclanthology.org/2021.emnlp-main.612/)]

3. "An Isotropy Analysis in the Multilingual BERT Embedding Space".
   
   Rajaee S, Pilehvar M T. ACL 2022. [[Paper](https://aclanthology.org/2022.findings-acl.103/)]

4. "Discovering Low-rank Subspaces for Language-agnostic Multilingual Representations".
   
   Xie Z, Zhao H, Yu T, et al. EMNLP 2022. [[Paper](https://aclanthology.org/2022.emnlp-main.379/)]

5. "Emerging Cross-lingual Structure in Pretrained Language Models".
   
   Conneau A, Wu S, Li H, et al. ACL 2020. [[Paper](https://aclanthology.org/2020.acl-main.536/)]

6. "Probing LLMs for Joint Encoding of Linguistic Categories".
   
   Starace G, Papakostas K, Choenni R, et al. EMNLP 2023. [[Paper](https://aclanthology.org/2023.findings-emnlp.476/)]

7. "Morph Call: Probing Morphosyntactic Content of Multilingual Transformers".
   
   Mikhailov V, Serikov O, Artemova E. Proceedings of the Third Workshop on Computational Typology and Multilingual NLP 2021. [[Paper](https://arxiv.org/abs/2104.12847)]

8. "Same Neurons, Different Languages: Probing Morphosyntax in Multilingual Pre-trained Models".
   
   Stanczak K, Ponti E, Hennigen L T, et al. NAACL 2022. [[Paper](https://aclanthology.org/2022.naacl-main.114/)]

9. "Probing Cross-Lingual Lexical Knowledge from Multilingual Sentence Encoders".
   
   Vulić I, Glavaš G, Liu F, et al. EACL 2023. [[Paper](https://aclanthology.org/2023.eacl-main.153/)]

10. "The Emergence of Semantic Units in Massively Multilingual Models".
    
    de Varda A G, Marelli M. LREC-COLING 2024. [[Paper](https://aclanthology.org/2024.lrec-main.1382/)]

11. "X-FACTR: Multilingual Factual Knowledge Retrieval from Pretrained Language Models".
    
    Jiang Z, Anastasopoulos A, Araki J, et al. EMNLP 2020. [[Paper](https://aclanthology.org/2020.emnlp-main.479/)]

12. "Multilingual LAMA: Investigating Knowledge in Multilingual Pretrained Language Models".
    
    Kassner N, Dufter P, Schütze H. EACL 2021. [[Paper](https://aclanthology.org/2021.eacl-main.284/)]

13. "Cross-Lingual Consistency of Factual Knowledge in Multilingual Language Models".
    
    Qi J, Fernández R, Bisazza A. EMNLP 2023. [[Paper](https://aclanthology.org/2023.emnlp-main.658/)]

14. "Language Representation Projection: Can We Transfer Factual Knowledge across Languages in Multilingual Language Models?".
    
    Xu S, Li J, Xiong D. EMNLP 2023. [[Paper](https://aclanthology.org/2023.emnlp-main.226/)]

### Interpretability of Cross-lingual Transfer

1. "Are Structural Concepts Universal in Transformer Language Models? Towards Interpretable Cross-Lingual Generalization"
   
   Xu N, Zhang Q, Ye J, et al. EMNLP 2023. [[Paper](https://aclanthology.org/2023.findings-emnlp.931/)]

2. "When is BERT Multilingual? Isolating Crucial Ingredients for Cross-lingual Transfer".
   
   Deshpande A, Talukdar P, Narasimhan K. NACCL 2022. [[Paper](https://aclanthology.org/2022.naacl-main.264/)]

3. "Emerging Cross-lingual Structure in Pretrained Language Models".
   
   Conneau A, Wu S, Li H, et al. ACL 2020. [[Paper](https://aclanthology.org/2020.acl-main.536/)]

4. "Cross-Lingual Ability of Multilingual BERT: An Empirical Study".
   
   Karthikeyan K, Wang Z, Mayhew S, et al. ICLR 2020. [[Paper](https://arxiv.org/abs/1912.07840)]

5. "Unveiling Linguistic Regions in Large Language Models".
   
   Zhang Z, Zhao J, Zhang Q, et al. ACL 2024. [[Paper](https://arxiv.org/abs/2402.14700)]

6. "Unraveling Babel: Exploring Multilingual Activation Patterns of LLMs and Their Applications".
   
   Liu W, Xu Y, Xu H, et al. EMNLP 2024. [[Paper](https://arxiv.org/abs/2402.16367)]

7. "The Geometry of Multilingual Language Model Representations".
   
   Chang T, Tu Z, Bergen B. EMNLP 2022. [[Paper](https://aclanthology.org/2022.emnlp-main.9/)]

### Interpretability of Linguistic Bias

1. "How do Large Language Models Handle Multilingualism?".
   
   Zhao Y, Zhang W, Chen G, et al. arXiv 2024. [[Paper](https://arxiv.org/abs/2402.18815)]

2. "Do Llamas Work in English? On the Latent Language of Multilingual Transformers".
   
   Wendler C, Veselovsky V, Monea G, et al. ACL 2024. [[Paper](https://arxiv.org/abs/2402.10588)]
