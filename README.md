# [IEEE Access] Modeling of Negative Feedback Refined via LLM in Recommender Systems
[![View Paper](https://img.shields.io/badge/View%20Paper-PDF-red?logo=adobeacrobatreader)](https://doi.org/10.1109/ACCESS.2025.3599176)

This study presents the **ReFINe++** model, an extension of the **ReFINe** framework designed to address negative feedback in recommender systems.  
[This paper](https://ieeexplore.ieee.org/document/11126013) has been accepted for publication in [IEEE Access](https://ieeexplore.ieee.org/xpl/RecentIssue.jsp?punumber=6287639).  
The previous work, **ReFINe** model, with its corresponding [paper](https://ieeexplore.ieee.org/document/11126013) and [code](https://github.com/Chanwoo-Jeong-2000/ReFINe_plus), is publicly available.  

![ReFINe++](https://github.com/user-attachments/assets/1e30de1e-a93c-48bf-ada0-a7a865b0522a)

# Requirements
python 3.9.20, cuda 11.8, and the following installations:
```
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install torch_geometric
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.5.0+cu118.html
pip install pandas
pip install scikit-learn
pip install transformers
pip install ptyprocess ipykernel pyzmq -U --force-reinstall
```

# Run
##### ML-100K
```
python main.py --dataset ML-100K
```
##### ML-1M
```
python main.py --dataset ML-1M
```
##### Netflix-1M
```
python main.py --dataset Netflix-1M
```
##### Amazon-Music
```
python main.py --dataset Amazon_Digital_Music
```
##### Amazon-Book
```
python main.py --dataset Amazon_Books --layers 2
```

# Settings
We adopt the same experimental configuration as in [ReFINe](https://github.com/Chanwoo-Jeong-2000/ReFINe).
The model is built using PyTorch Geometric with a 64-dimensional embedding size and a batch size of 1024.
A 4-layer GCN is used for all datasets, except for Amazon-Book, where a 2-layer GCN is used LightGCN.
The learning rate is set to 1e-3, and training is performed for up to 1000 epochs with early stopping, which is applied if Recall@20 on the validation set does not improve for 50 consecutive epochs.
Following ReFINe, the sampling weight γ for confirmed negative feedback is fixed at 1.5.
The regularization coefficient λ₁ for the L_{RW_SSM} term is set to 1e-7, and λ₂ for the L_{AE} term is set to 1e-5.
The autoencoder for capturing dispreference has one hidden layer and follows the architecture [ |I| -> 600 -> 64 -> 600 -> |I| ].
In Re-Weighted SSM, we do not treat the number of negative samples as a tunable hyperparameter.
All experiments are conducted on a single NVIDIA RTX A6000 GPU.

# Datasets
Download **ml-100k.inter** and **ml-100k.item** from [here](https://recbole.s3-accelerate.amazonaws.com/ProcessedDatasets/MovieLens/ml-100k.zip).

Download **ml-1m.inter** and **ml-1m.item** from [here](https://recbole.s3-accelerate.amazonaws.com/ProcessedDatasets/MovieLens/ml-1m.zip).

Download **netflix.inter** from [here](https://recbole.s3-accelerate.amazonaws.com/ProcessedDatasets/Netflix/netflix.zip).

Download **movie_titles.csv** from [here](https://www.kaggle.com/datasets/netflix-inc/netflix-prize-data?select=movie_titles.csv).

Download **netflix_genres.csv** from [here](https://github.com/tommasocarraro/netflix-prize-with-genres).

Download **Amazon_Digital_Music.inter** and **Amazon_Digital_Music.item** from [here](https://recbole.s3-accelerate.amazonaws.com/ProcessedDatasets/Amazon_ratings/Amazon_Digital_Music.zip).

Download **Amazon_Books.inter** and **Amazon_Books.item** from [here](https://recbole.s3-accelerate.amazonaws.com/ProcessedDatasets/Amazon_ratings/Amazon_Books.zip).

# Compare with our results
The **results4comparison** folder contains the results of our experiment.
Each file includes the loss and performance metrics for every epoch, as well as the hyperparameters, dataset statistics, and training time.
You can compare our results with your own reproduced results.

# Citation
If you find ReFINe++ useful for your research or development, please cite the following our papers:
```
@article{jeong2025modeling,
  title={Modeling of Negative Feedback Refined via LLM in Recommender Systems},
  author={Jeong, Chanwoo and Cho, Yoon-Sik},
  journal={IEEE Access},
  year={2025},
  publisher={IEEE}
}
```

# Acknowledgments
This work was supported by the Institute of Information \& Communications Technology Planning \& Evaluation (IITP) grant funded by the Korea government (MSIT) [RS-2021-II211341, Artificial Intelligence Graduate School Program (Chung-Ang University)], [IITP-2025-RS-2024-00438056, ITRC(Information Technology Research Center) Support Program]. This research was also supported by the Chung-Ang University Research Scholarship Grants in 2023. 
