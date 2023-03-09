# Retail Recommender with sBERT and GCN
Third place submission for 2023 Daisy Intelligence Hackathon

## Purpose
40% of a businesses revenue comes from repeat customers [1]. This is especially important for small businesses with significantly smaller consumers, so our goal is to grow the number of regular customers by appealing to niche product categories. We decided to do this by utilizing AI to recommend these products using a custom recommender system we created. According to Nvidia, the average recommender system increases the conversion rate from clicks to purchases by 22.66%, introducing more products to customers [2]. We hope that the introduction of these products will incentivise consumers to purchase more and therefore increase our small businesses profit. 

## sBERT and GCN Model
- **Metric**: Recall = $\frac{\text{True Positives (TPR)}}{\text{TPR + False Negatives}}$ which gives us the percentage of valid products that we recommend. 
- **Dataset**: UCI Online Retail dataset, 80-20 Temporal Split, changed data to be numerical torch compatible and removed invalid entries
- **Trained on**: Purchase Description through sBERT, Quantity and Unit price for edge weighting in our graph network
<img src="https://github.com/JuliaLWang8/julialwang8.github.io/blob/master/src/media/GCN.png" width=600>

sBERT takes in item descriptions in text format, then outputs numeric embeddings to use as input feature to GCN. This encodes semantic knowledge while dealing with new entries (weakness of CBRS). GCN then computes graph-aware embeddings of users and items, where it uses these generated embeddings to predict item-user “similarity” as a proxy for items to recommend.


### Why sBERT & GCN
- **sBERT** is a transformer-based model that balances the computation efficiency with the results we obtain. It allows us to encode semantics and meaning, while also being able to adapt further through transfer learning to continue learning. 
- **Why not GPT?** GPT decodes from pre-trained embeddings. This means that it doesn’t learn and does not effectively provide user-specific tasks. 
- Using a **GCN** takes into account the full topology of the user-item space, meanwhile, other options such as clustering can only  effectively cover user-space *xor* item space. By using a graph, our model also takes into account how much people liked certain items using the weighting on the edges of nodes. 

