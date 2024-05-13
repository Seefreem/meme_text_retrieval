# meme_text_retrieval

Group members: 
Shiling Deng, Qing Li and Yuwei Shen

## Overview
This project is the final exam assignment of the course HIOK0003FU Computational Cognitive Science 2 at the University of Copenhagen.  
We will explore the effectiveness of cross-modal embedding models, specifically focusing on their application in meme-text retrieval scenarios. The primary `research question` addresses how well a model trained on general image-text data can adapt to the specific demands of interpreting memes. Given the unique characteristics of memes, which often rely heavily on metaphorical and cultural context, this presents a challenging yet intriguing problem in the field of machine learning.

## Objective
The objective of this project is to evaluate the performance of models, e.g. ALIGN, on specialized meme-text datasets, e.g. MemeCap.

## Hypothesis
We hypothesize that the cross-modal embedding model's performance might degrade when applied to meme-text data compared to its performance on general datasets. This potential degradation could be due to the model's interpretation of text in memes not just as labels or descriptors, but as integral, often symbolic elements of the images.

## Current Status
This project is currently in progress. 

## zero-shot evaluation
- Dataset: the test set of MemeCap. 558 samples.

### ALIGN
The ALIGN model is a global alignment algorithm.
- R@1: 0.5394265232974911,
- R@5: 0.7258064516129032,
- R@10: 0.7741935483870968,
- R_mean: 0.6798088410991637

![pic](/pictures/txt2img_align.png "retrieval probabilities matrix")

This hot map shows the first 100 text-image similarity scores of the ALIGN model. The y-axis is the indexes of meme captions and the x-axis is the meme indexes. There are about 54% meme-text pairs that have the highest similarity on the right indexes, which means the model performance significantly better than random behavior, but there are still rooms for improvement.

### CLIP
The CLIP model is a global alignment algorithm.
- R@1: 0.45698924731182794
- R@5: 0.6648745519713262
- R@10: 0.7329749103942652
- R_mean: 0.6182795698924731

![pic](/pictures/txt2img_clip.png "retrieval probabilities matrix")

This hot map shows the first 100 text-image similarity scores of the CLIP model. The y-axis is the indexes of meme captions and the x-axis is the meme indexes. There are about 46% meme-text pairs that have the highest similarity on the right indexes. There are some light spots spreading over the plot, suggesting that those text embeddings are similar to memes that are not matched with the query texts. 

### UNITER
The UNITER model is a local-involved alignment algorithm.

### ALBEF
The ALBEF model is a local-involved alignment algorithm.

### SGRAF
The SGRAF model is a relation-aware local-involved alignment algorithm.


# License
TODO