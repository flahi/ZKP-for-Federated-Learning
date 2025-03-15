import socket
import threading
import numpy as np
import time
import pandas as pd
import os
import json
import copy
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from imblearn.over_sampling import SMOTE
from sklearn.metrics import confusion_matrix
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

class global_mod:
	def __init__(self, port):
		self.port = port
		self.local_ports = []
		self.model = RandomForestClassifier(
			random_state = 9,
			n_estimators = 100,
			max_depth = 12,
			class_weight = 'balanced',
			oob_score = True
		)
		self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
		self.server_socket.bind(('localhost', self.port))
		self.server_socket.listen()
		threading.Thread(target=self.listen_for_data, daemon=True).start()
	def listen_for_data(self):
		print(f"Global hospital listening on port {self.port}")
		while True:
			client_socket, _ = self.server_socket.accept()
			threading.Thread(target=self.handle_data, args=(client_socket, _), daemon=True).start()
	def handle_data(self, client_socket, client_address):
		data_from_client = client_socket.recv(8192).decode()
		data = json.loads(data_from_client)
		client_socket.close()
		print(data)
		if (data["request"]==1):
			limits = {"request":2, "ranges":self.ranges, "port":self.port}
			limits_encoded = json.dumps(limits).encode()
			with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
				try:
					s.connect(('localhost', data["port"]))
					s.sendall(limits_encoded)
				except ConnectionRefusedError:
					print(f"Node {self.node_id} could not connect to Node on port {port}")
	def train(self, x_train, x_test, y_train, y_test):
		self.model.fit(x_train, y_train)
		y_pred_proba = self.model.predict_proba(x_test)[:, 1]
		threshold = 0.75
		y_pred = (y_pred_proba >= threshold).astype(int)
		accuracy = accuracy_score(y_test, y_pred)
		print(f"Accuracy for hospital: {accuracy:.2f}")
		print(classification_report(y_test, y_pred))
		cm = confusion_matrix(y_test, y_pred)
		print(f"Confusion Matrix for hospital:\n{cm}")
		feature_importances = self.get_feature_importances()
		self.ranges = [[i-1000, i+1000] for i in feature_importances]
	def get_feature_importances(self):
		feature_importances = self.model.feature_importances_.tolist()
		feature_importances = [int(i*(10**6)) for i in feature_importances]
		return feature_importances

def load_data(path):
	if not os.path.exists(path):
		print(f"Error: File not found at {path}")
		return None, None
	dataset = pd.read_csv(path)
	print("File loaded successfully")
	print(dataset.head())
	X = dataset.iloc[:, :-1].values
	Y = dataset.iloc[:, -1].values
	return X, Y

def preprocess_data(X, Y):
	print("\nData preprocessing...")
	imputer=SimpleImputer(missing_values=np.nan,strategy='mean')
	imputer.fit(X[:, 1:4])
	X[:, 1:4] = imputer.transform(X[:, 1:4])
	imputer.fit(X[:, 5:])
	X[:, 5:] = imputer.transform(X[:, 5:])
	print("Missing data handled.")
	
	print("Unique values before change: ",np.unique(X[:,4]))
	for i in range(X.shape[0]):
		X[i, 4] = categorize_smoking(X[i, 4])
	print("Unique values in the 'smoking_history' column after categorization:", np.unique(X[:, 4]))
	print("Regularized contents of smoking history.")
	
	gender_mapping = {'Female': 0, 'Male': 1, 'Other': 2}
	smoking_mapping = {'non-smoker': 0, 'current-smoker': 1, 'past-smoker': 2}
	for i in range(X.shape[0]):
		X[i, 0] = gender_mapping[X[i, 0]]
	for i in range(X.shape[0]):
		X[i, 4] = smoking_mapping[X[i, 4]]
	X = X.astype(float)
	print("Encoded string contents.")
	
	print("Data preprocessed successfully.")
	#print("\nPreprocessed X")
	#print(X)
	#print("\nPreprocessed Y")
	#print(Y)
	return X, Y

def split_data(X, Y, n):
	print(f"\nSplitting data into {n}")
	x_array = np.array_split(X, n)
	y_array = np.array_split(Y, n)
	#for i in range(n):
	#	print(f"Hospital {i+1} - X shape: {x_array[i].shape}, Y shape: {y_array[i].shape}")
	#	print(x_array[i])
	return x_array, y_array

def test_train(x_array, y_array, n):
	x_train, x_test, y_train, y_test = [0]*n, [0]*n, [0]*n, [0]*n
	for i in range(n):
		x_train[i], x_test[i], y_train[i], y_test[i] = train_test_split(x_array[i], y_array[i],test_size=0.2,random_state=1)
	print("Data split for training and testing.")
	return x_train, x_test, y_train, y_test

def scale_and_synthesize(x_train, x_test, y_train):
	columns_to_scale = [1, 5, 6, 7]
	scaler = StandardScaler()
	x_train[:, columns_to_scale] = scaler.fit_transform(x_train[:, columns_to_scale])
	x_test[:, columns_to_scale] = scaler.transform(x_test[:, columns_to_scale])
	smote = SMOTE(random_state=9)
	x_train, y_train = smote.fit_resample(x_train, y_train)
	return x_train, x_test, y_train

def categorize_smoking(smoking_status):
	if smoking_status in ['No Info', 'never']:
		return 'non-smoker'
	if smoking_status in ['current']:
		return 'current-smoker'
	if smoking_status in ['ever', 'not current', 'former']:
		return 'past-smoker'

n = 3
main_port = 6000
global_model = global_mod(main_port)

path = "Sample Data/diabetes_prediction_dataset.csv"
X, Y = load_data(path)

X, Y = preprocess_data(X, Y)

x_array, y_array = split_data(X, Y, n+1)

x_train, x_test, y_train, y_test = test_train(x_array, y_array, n+1)

print("Training global model...")
global_model.train(x_train[n], x_test[n], y_train[n], y_test[n])

time.sleep(1000)
