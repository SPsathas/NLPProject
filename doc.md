## RQ 1
**MAKING RETRIEVAL-AUGMENTED LANGUAGE MODELS ROBUST TO IRRELEVANT CONTEXT**

_Yoran 2024_

Investiagted how negative context influences performance in single- and multi-hop QA tasks and developed a strategy to select relevant context. They found that negative context degrades answer performance while training the model to ignore negative context achieves at least baseline or better results compared to a parametrical model.

> To test our methods, we evaluate retrieval robustness on five ODQA benchmarks, four of which contain multi-hop questions, where the retriever is called multiple times (Jiang et al., 2023). Fig. 2 shows that even with a strong retriever (top-1 Google search) incorporating the retrieved context actually hurts model performance on two of the benchmarks (STRATEGYQA and FERMI).**Moreover, adding randomly-retrieved contexts dramatically decreases accuracy on all five datasets.** Our analysis (§5) shows that irrelevant context causes a wide range of errors, which include copying irrelevant answers from the retrieved sentences and hallucinating incorrect answers and decompositions.

> This suggests that irrelevant context can cause errors even when the generated entities are not retrieved


**Large Language Models with Controllable Working Memory**

_https://aclanthology.org/2023.findings-acl.112/_

They introduce two important terms:

- Controllability: Measured by the model’s ability to produce correct answers when the context is relevant but contradicts pretrained knowledge.
- Robustness: Measured by the model’s ability to avoid being influenced by irrelevant contexts.


They propose KAFT:

> KAFT enhances the controllability by creating counterfactual data augmentations where the answer entity in the context is swapped to a different but plausible entity, in conflict with the ground truth (and potentially the model’s world knowledge). As for enhancing robustness, KAFT requires that the model should predict its pretrained closed-book answer rather than the ground truth answer whenever the context is irrelevant.

They show that this form of training increases the robustness of the model against irrelevant context.

**How Context Affects Language Models' Factual Predictions**

_https://openreview.net/forum?id=025X0zPfn_


1. Augmentation with Retrieved Contexts:
- The core idea is to augment the input cloze questions with relevant context retrieved from an external corpus (e.g., Wikipedia).
- This retrieval is performed using an information retrieval (IR) system, such as TF-IDF, to fetch paragraphs related to the query.
- The context is then concatenated with the cloze question, using special tokens (e.g., [SEP] in BERT) to distinguish between the question and the context.
  
2. Contextual Input Handling:
- For BERT, different segment embeddings are used to indicate the question and the context separately. This allows BERT to utilize its Next Sentence Prediction (NSP) capability to assess the relevance of the context.
- For RoBERTa, which lacks segment embeddings, the end-of-sentence (eos) token is used to separate the question and context.
  
3. Next Sentence Prediction (NSP):
- BERT’s NSP pre-training helps determine whether the context is relevant to the question. If the context is deemed irrelevant, the model can effectively ignore it, thereby improving robustness against noisy or misleading contexts.
- This unsupervised mechanism relies on the self-supervised learning that BERT undergoes during pre-training, where it learns to distinguish between contiguous (related) and randomly sampled (unrelated) text segments.
  
4. Oracle and Retrieved Contexts:
- Oracle contexts, which are known to be relevant, are used to establish an upper performance bound.
- retrieved contexts from the IR system, though not perfect, often contain related information that can still aid in answering the question correctly.
    
https://aclanthology.org/2020.acl-main.698/

https://aclanthology.org/2021.emnlp-main.119/

https://aclanthology.org/2023.eacl-main.213/

https://arxiv.org/abs/2302.00093

maybe: https://openreview.net/forum?id=fcO9Cgn-X-R