# Sentiment-Analysis-using-LSTM
This project implements a deep learning-based sentiment analysis model using LSTM (Long Short-Term Memory) networks to classify the sentiment of movie reviews as positive or negative. The project uses the IMDb dataset, but it can be extended to any custom dataset.

Overview
The goal of this project is to classify movie reviews as having either positive or negative sentiment based on the text. We utilize an LSTM-based deep learning model along with word embeddings to capture the sentiment of the reviews. The project achieves an accuracy of 88% on the IMDb dataset.

Technologies Used
Python: The programming language used for developing the project.
TensorFlow/Keras: Used for building the LSTM model.
NumPy: Used for numerical computations.
Pandas: Used for handling data processing.
IMDb Dataset: Pre-processed dataset containing 50,000 movie reviews labeled as either positive or negative.
Project Structure
The main parts of the project include:

Loading and preprocessing the dataset.
Building the LSTM model for sentiment classification.
Training the model.
Evaluating the model's performance.
Files in the Project
sentiment_analysis_lstm.py: The main Python script that loads data, builds the model, trains, and evaluates it.
README.md: This file that contains the project overview and instructions for setting it up.
Requirements
To run this project, you will need to have the following packages installed:

tensorflow
numpy
pandas
scikit-learn
You can install the required packages by running:

bash
Copy code
pip install tensorflow numpy pandas scikit-learn
Dataset
IMDb Dataset
This project uses the IMDb dataset available directly via the Keras API. The dataset contains 50,000 highly polar movie reviews, split into 25,000 training and 25,000 testing samples. Each review is pre-labeled as positive or negative.

If you would like to use your own dataset, ensure that it is in CSV format, with one column containing the movie reviews and another column containing the sentiment labels (0 for negative, 1 for positive).

Usage
Clone the Repository

Clone this repository to your local machine:

bash
Copy code
git clone https://github.com/your-repository/sentiment-analysis-lstm.git
cd sentiment-analysis-lstm
Run the Sentiment Analysis

If you're using the IMDb dataset provided by Keras, simply run the Python script:

bash
Copy code
python sentiment_analysis_lstm.py
Using a Custom Dataset

To use a custom dataset:

Prepare your data in a CSV format with two columns: review and label.

Update the sentiment_analysis_lstm.py script to load your dataset instead of the IMDb dataset. For example:

python
Copy code
df = pd.read_csv('path_to_your_dataset.csv')
Ensure that your CSV file is placed in the correct directory, and the paths are updated in the script.

Model Training

The model will automatically start training for a default of 5 epochs with a batch size of 64. You can modify these parameters in the script if desired.

Evaluate the Model

Once the training is complete, the model will be evaluated on the test dataset, and the accuracy will be printed in the terminal.

bash
Copy code
Test Accuracy: 0.88
Make Predictions

You can modify the script to make predictions on new or unseen reviews using the trained model.

Customization
Hyperparameters: You can adjust hyperparameters like batch_size, epochs, max_len, and embedding_dim in the script to experiment with the model's performance.
Embedding Layer: You can also experiment with pre-trained word embeddings such as GloVe or Word2Vec by replacing the embedding layer with pre-trained weights.
Future Enhancements
Some possible future improvements to the project:

Implementing additional LSTM or GRU layers for better performance.
Using more advanced pre-trained embeddings like GloVe or FastText.
Hyperparameter tuning with Keras Tuner or GridSearchCV.
Adding support for multi-class sentiment analysis (e.g., classifying reviews into more than two categories).
Results
The model achieves an accuracy of 88% on the IMDb test dataset. With further tuning and advanced embeddings, the performance could be improved even more.

License
This project is open-source and available under the MIT License.

Contact
For any inquiries, please reach out to:

Timmirishetty Vivek
LinkedIn
Email: timmirishettyvivek@gmail.com
