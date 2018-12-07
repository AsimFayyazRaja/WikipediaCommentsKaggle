# WikipediaCommentsKaggle
Classification of wikipedia comments using LSTM model by making converting comments to vectors using word2vec model.

## Usage
- Download dataset from [Comments CLassification Kaggle](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge/submit)
- Extract data in the current folder with all files
- Execute toxicity_Word2vec.py to get training data converted to vectors.
- Execute LSTM_model.py to train the model
- Execute get_test_data.py to generate test data's encoding from same model of Word2vec
- Run test_LSTM.py to check result's quality and for making submission file.

## Results
- Got 98% training and 94% validation accuracy.
- Got 0.79 Kaggle score where as highest was 0.96

## License
This is a free code to use and experiment
