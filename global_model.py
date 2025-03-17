import socket
import threading
import numpy as np
import time
import pandas as pd
import os
import json
import copy
from ZKP import *
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from imblearn.over_sampling import SMOTE
from sklearn.metrics import confusion_matrix
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

class global_mod:
	def __init__(self, port, n):
		self.port = port
		self.local_ports = list()
		self.model = RandomForestClassifier(
			random_state = 9,
			n_estimators = 100,
			max_depth = 12,
			class_weight = 'balanced',
			oob_score = True
		)
		self.local_commitments = dict()
		self.valid_models = list()
		self.no_of_local = n
		self.no_of_local_recieved = 0
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
		data_from_client = b""
		while True:
			chunk = client_socket.recv(4098)
			if not chunk:
				break
			data_from_client += chunk
		data = json.loads(data_from_client.decode())
		deserialize_ZKP_json(data)
		client_socket.close()
		print(data)
		if (data["type"]==1):
			self.local_commitments[data["port"]] = data["commitments"]
			self.send_ranges(data["port"])
		elif (data["type"]==3):
			self.no_of_local_recieved += 1
			proofs = data["proofs"]
			self.check_validity(proofs, data["port"])
			if self.no_of_local_recieved==self.no_of_local:
				time.sleep(2)
				self.send_valid_ports()
		else:
			print("Random access recieved...")
	def send_ranges(self, local_port):
		limits = {"type":2, "ranges":self.ranges, "port":self.port}
		limits_encoded = json.dumps(limits).encode()
		with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
			try:
				s.connect(('localhost', local_port))
				s.sendall(limits_encoded)
			except ConnectionRefusedError:
				print(f"Global model could not connect to Node on port {local_port}")
	def check_validity(self, proofs, local_port):
		check = True
		for i in range(len(proofs)):
			if (self.local_commitments[local_port][i]==proofs[i]["C"]):
				print("\nProof: ", proofs[i])
				if (validate_proof(proofs[i], self.ranges[i][0], self.ranges[i][1])):
					print("Proof valid.")
				else:
					check = False
					print("ZKP proof invalid.")
			else:
				check = False
				print("proof invalid.")
		print(f"\nModel validity: {check}")
		validity = {"type": 4, "validity":check, "port":self.port}
		validity_encoded = json.dumps(validity).encode()
		if check:
			self.valid_models.append(local_port)
		with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
			try:
				s.connect(('localhost', local_port))
				s.sendall(validity_encoded)
			except ConnectionRefusedError:
				print(f"Global model could not connect to Node on port {local_port}")
	def send_valid_ports(self):
		valid_model_data = {"type":5, "port":self.port, "valid models":self.valid_models}
		valid_model_data_encoded = json.dumps(valid_model_data).encode()
		print(f"Valid model ports: {self.valid_models}")
		for port in self.valid_models:
			with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
				try:
					s.connect(('localhost', port))
					s.sendall(valid_model_data_encoded)
				except ConnectionRefusedError:
					print(f"Global model could not connect to Node on port {port}")
		self.no_of_local_recieved = 0
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
		print(feature_importances)
		percent = 0.33
		self.ranges = [[int(np.maximum(0, i-(i*percent))), int(i+(i*percent))] for i in feature_importances]
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
global_model = global_mod(main_port, n)

path = "Sample Data/diabetes_prediction_dataset.csv"
X, Y = load_data(path)

X, Y = preprocess_data(X, Y)

x_array, y_array = split_data(X, Y, n+1)

x_train, x_test, y_train, y_test = test_train(x_array, y_array, n+1)

print("Training global model...")
global_model.train(x_train[n], x_test[n], y_train[n], y_test[n])

time.sleep(1000)
