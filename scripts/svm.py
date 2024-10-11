import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score
import joblib

def process_data(dataframe):
	""" Process text """
	dataframe['text'] = dataframe['text'].str.lower()
	dataframe['text'] = dataframe['text'].astype(str)
	
	# Text, labels
	X = dataframe['text']
	y = dataframe['sentiment']
	# Split
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
	
	return X_train, X_test, y_train, y_test

def main():
	# Load dataset
	df = pd.read_csv('../data/social_media/sentiment_analysis.csv')
	sent_df = df[['text', 'sentiment']]
	
	X_train, X_test, y_train, y_test = process_data(sent_df)
	
	# Vectorize text
	vectorizer = TfidfVectorizer()
	X_train_vectors = vectorizer.fit_transform(X_train)
	X_test_vectors = vectorizer.transform(X_test)
	
	# Model definition and training
	model_svc = SVC()
	model_svc.fit(X_train_vectors, y_train)
	
	# Save the model using joblib
	joblib.dump(model_svc, 'svc_model.pkl')
	joblib.dump(vectorizer, 'tfidf_vectorizer.pkl')
	
	# Testing
	y_predict = model_svc.predict(X_test_vectors)
	
	print(f"Classification Report: {classification_report(y_test, y_predict)}")
	print(f"Accuracy Score: {accuracy_score(y_test, y_predict)}")

if __name__ == '__main__':
	main()
