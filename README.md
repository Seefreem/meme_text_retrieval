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

## Preliminary Result
![pic](/pictures/retrieval_probabilities_matrix.png "retrieval probabilities matrix")

This hot map shows the image-text similarity score, where the x-axis is the indexes of meme captions and the y-axis is the meme indexes. There are 60% meme-text pairs that have the highest similarity on the right indexes, which means the model performance significantly better than random behavior, but there are still rooms for improvement.


# License
TODO