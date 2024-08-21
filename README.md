# Hubness Precision Recall For Generative Image
project page: https://byronliang8.github.io/Hubness_Precision_Recall_page/

## Abstract
Despite impressive results, deep generative models require massive datasets for training. As dataset size increases, effective evaluation metrics like precision and recall (P\&R) become computationally infeasible on commodity hardware. 
In this paper, we address this challenge by proposing efficient P\&R (eP\&R) metrics that give almost identical results as the original P\&R but with much lower computational costs. 
Specifically, we identify two redundancies in the original P\&R: i) redundancy in ratio computation and ii) redundancy in manifold inside/outside identification. We find both can be effectively removed via hubness-aware sampling, which extracts representative elements from synthetic/real image samples based on their hubness values, \ie, the number of times a sample becomes a $k$-nearest neighbor to others in the feature space. 
Thanks to the insensitivity of hubness-aware sampling to exact $k$-nearest neighbor ($k$-NN) results, we further improve the efficiency of our eP\&R metrics by using approximate $k$-NN methods.
Extensive experiments show that our eP\&R matches the original P\&R but is far more efficient in time and space. 