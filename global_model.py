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

np.set_printoptions(linewidth=np.inf)

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
		self.total_r = 0
		self.total_fi = list()
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
		elif (data["type"]==7):
			self.update_total_data(data)
			if self.no_of_local_recieved == len(self.valid_models):
				if self.verify_total_data(data):
					print(f"Feature summation is correctly sent.")
					print("\nAggregating...")
					time.sleep(0.5)
					self.aggregate()
				else:
					print(f"Feature summation is incorrect.")
				self.no_of_local = 0
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
		print("\n\nChecking validity of proofs of local model {local_port - self.port}...")
		check = True
		for i in range(len(proofs)):
			if (self.local_commitments[local_port][i]==proofs[i]["C"]):
				#print("\nProof {i} for {local_port - self.port}: ", proofs[i])
				if (validate_proof(proofs[i], self.ranges[i][0], self.ranges[i][1])):
					print(f"Proof {i} for local model {local_port - self.port} is valid.")
				else:
					check = False
					print(f"Proof {i} for local model {local_port - self.port} is invalid.")
			else:
				check = False
				print(f"Proof {i} for local model {local_port - self.port} is invalid.")
		print(f"\nModel {local_port - self.port} validity: {check}")
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
		print(f"\n\nValid model ports: {self.valid_models}\n")
		for port in self.valid_models:
			with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
				try:
					s.connect(('localhost', port))
					s.sendall(valid_model_data_encoded)
				except ConnectionRefusedError:
					print(f"Global model could not connect to Node on port {port}")
		self.no_of_local_recieved = 0
	def update_total_data(self, data):
		self.no_of_local_recieved += 1
		self.total_r += data["partial r"]
		for i in range(len(self.total_fi)):
			self.total_fi[i] += data["partial fi"][i]
	def verify_total_data(self, data):
		print("\nChecking if recieved values are correct...")
		print(f"Total r: {self.total_r}")
		print(f"Total fi: {self.total_fi}")
		valid_commitments = self.get_valid_commitments()
		C_total_LHS = []
		C_total_RHS = []
		time.sleep(0.5)
		for i in range(len(valid_commitments)):
			C_total = add_commitments(valid_commitments[i])
			C_total_LHS.append(C_total)
			C_calculated = pedersen_commit(self.total_fi[i], self.total_r, G, H)
			C_total_RHS.append(C_calculated)
			if (C_total!=C_calculated):
				return False
		#print(f"C LHS = {C_total_LHS}")
		#print(f"C RHS = {C_total_RHS}")
		return True
	def train(self, x_train, x_test, y_train, y_test, sw=None):
		self.model.fit(x_train, y_train, sample_weight=sw)
		y_pred_proba = self.model.predict_proba(x_test)[:, 1]
		threshold = 0.75
		y_pred = (y_pred_proba >= threshold).astype(int)
		accuracy = accuracy_score(y_test, y_pred)
		print(f"Accuracy for hospital: {accuracy:.2f}")
		print(classification_report(y_test, y_pred))
		cm = confusion_matrix(y_test, y_pred)
		print(f"Confusion Matrix for hospital:\n{cm}")
		feature_importances = self.get_feature_importances()
		self.total_fi = [0]*len(feature_importances)
		percent = 0.33
		self.ranges = [[int(np.maximum(0, i-(i*percent))), int(i+(i*percent))] for i in feature_importances]
		print(f"\nFeature importances: {self.model.feature_importances_.tolist()}")
	def get_feature_importances(self):
		feature_importances = self.model.feature_importances_.tolist()
		feature_importances = [int(i*(10**6)) for i in feature_importances]
		return feature_importances
	def get_valid_commitments(self):
		valid_commitments = []
		for i in range(len(self.get_feature_importances())):
			feature = []
			for j in range(len(self.valid_models)):
				feature.append(self.local_commitments[self.valid_models[j]][i])
			valid_commitments.append(feature)
		return valid_commitments
	def aggregate(self):
		aggregated_feature_importances = self.calculate_aggregated_fi()
		print(f"Aggregated feature importances: {aggregated_feature_importances}")
		print(f"\n\nRetraining global model...\n")
		sample_weights = np.dot(x_train[n], aggregated_feature_importances)
		sample_weights = (sample_weights - np.min(sample_weights)) / (np.max(sample_weights) - np.min(sample_weights))
		self.train(x_train[n], x_test[n], y_train[n], y_test[n], sample_weights)
	def calculate_aggregated_fi(self):
		sum_array = [i/((10**6)*len(self.valid_models)) for i in self.total_fi]
		aggregated = np.array(sum_array)
		aggregated /= np.sum(aggregated)
		return aggregated

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

def add_commitments(commitments):
	val = commitments[0]
	for i in range(1, len(commitments)):
		val = add(val, commitments[i])
	return val

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
print("Training complete.")
print("\nWaiting for local models...")

time.sleep(60)
