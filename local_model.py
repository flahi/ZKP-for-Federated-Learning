import socket
import threading
import numpy as np
import time
import pandas as pd
import os
import json
import copy
import random
from ZKP import *
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from imblearn.over_sampling import SMOTE
from sklearn.metrics import confusion_matrix
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

class local_mod:
	def __init__(self, id, port, other_ports):
		self.id = id
		self.port = port
		self.other_ports = other_ports
		self.model = RandomForestClassifier(
			random_state = id,
			n_estimators = 100,
			max_depth = 12,
			class_weight = 'balanced',
			oob_score = True
		)
		threading.Thread(target=self.listen_for_data, daemon=True).start()
	def listen_for_data(self):
		server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
		server_socket.bind(('localhost', self.port))
		server_socket.listen()
		print(f"Local hospital {self.id} listening on port {self.port}")
		while True:
			client_socket, _ = server_socket.accept()
			threading.Thread(target=self.handle_data, args=(client_socket,), daemon=True).start()
	def handle_data(self, client_socket):
		data_from_client = b""
		while True:
			chunk = client_socket.recv(4098)
			if not chunk:
				break
			data_from_client += chunk
		data = json.loads(data_from_client.decode())
		data = json.loads(data_from_client)
		client_socket.close()
		if (data["type"]==2):
			self.range_from_global = data["ranges"]
		elif (data["type"]==4):
			print(data["validity"])
		else:
			print("Random access recieved.")
	def train(self, x_train, x_test, y_train, y_test):
		self.model.fit(x_train, y_train)
		y_pred_proba = self.model.predict_proba(x_test)[:, 1]
		threshold = 0.75
		y_pred = (y_pred_proba >= threshold).astype(int)
		accuracy = accuracy_score(y_test, y_pred)
		print(f"Accuracy for Hospital {self.id}: {accuracy:.2f}")
		print(classification_report(y_test, y_pred))
		cm = confusion_matrix(y_test, y_pred)
		print(f"Confusion Matrix for Hospital {self.id}:\n{cm}")
		print(self.get_feature_importances())
	def get_feature_importances(self):
		feature_importances = self.model.feature_importances_.tolist()
		feature_importances = [int(i*(10**6)) for i in feature_importances]
		return feature_importances
	def generate_proof(self, port):
		feature_importances = self.get_feature_importances()
		commitments = []
		r = randbelow(curve_order)
		C_list = []
		print(f"\nGenerating commitments for hospital {self.id}...")
		for i in range(len(feature_importances)):
			commitment = pedersen_commit(feature_importances[i], r, G, H)
			commitments.append(commitment)
		print("Commitments\n", commitments)
		commitment_transaction = {"port":self.port, "type": 1, "commitments":commitments}
		commitment_transaction_serialized = copy.deepcopy(commitment_transaction)
		serialize_ZKP_json(commitment_transaction_serialized)
		commitment_transaction_data = json.dumps(commitment_transaction_serialized).encode()
		print("Sending commitments...")
		with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
			try:
				s.connect(('localhost', port))
				s.sendall(commitment_transaction_data)
				print("Commitments sent successfully.")
			except ConnectionRefusedError:
				print(f"Local model {self.id} could not connect to Node on port {port}")
		time.sleep(2)
		print(f"\nRanges recieved from global hospitals.")
		proofs = []
		print(f"\nGenerating proofs for hospital {self.id}...")
		for i in range(len(feature_importances)):
			print("\nRange: ",self.range_from_global[i])
			print("FI: ", feature_importances[i])
			proof = create_proof(feature_importances[i], r, self.range_from_global[i][0], self.range_from_global[i][1], commitments[i], G, H)
			print(proof)
			proofs.append(proof)
		print("Proofs",proofs)
		proof_transaction = {"port":self.port, "type": 3, "proofs":proofs}
		proof_transaction_serialized = copy.deepcopy(proof_transaction)
		serialize_ZKP_json(proof_transaction_serialized)
		proof_transaction_data = json.dumps(proof_transaction_serialized).encode()
		print("Sending proofs...")
		with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
			try:
				s.connect(('localhost', port))
				s.sendall(proof_transaction_data)
				print("Proofs sent successfully.")
			except ConnectionRefusedError:
				print(f"Local model {self.id} could not connect to Node on port {port}")
		time.sleep(2)

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

def create_hospitals(num, base_port=5000):
	hospitals = []
	ports = [base_port+i+1 for i in range(num)]
	for i in range(num):
		other = ports[:i] + ports[i+1:]
		hospital = local_mod(i+1, ports[i], other)
		hospitals.append(hospital)
	return hospitals

def train_local_hospitals(local_hospitals, x_train, x_test, y_train, y_test, n):
	print()
	for i in range(n):
		x_train[i], x_test[i], y_train[i] = scale_and_synthesize(x_train[i], x_test[i], y_train[i])
		print(f"\nTraining local model {i+1}")
		local_hospitals[i].train(x_train[i], x_test[i], y_train[i], y_test[i])

def categorize_smoking(smoking_status):
	if smoking_status in ['No Info', 'never']:
		return 'non-smoker'
	if smoking_status in ['current']:
		return 'current-smoker'
	if smoking_status in ['ever', 'not current', 'former']:
		return 'past-smoker'

def split_value(val, n):
	if n==1:
		return [val]
	breakpoints = sorted(random.sample(range(1, val), n - 1))
	parts = [breakpoints[0]] + [breakpoints[i] - breakpoints[i - 1] for i in range(1, n - 1)] + [val - breakpoints[-1]]
	return parts

n = 3
main_port = 6000
local_hospitals = create_hospitals(n, main_port)
time.sleep(0.5)

path = "Sample Data/diabetes_prediction_dataset.csv"
X, Y = load_data(path)

X, Y = preprocess_data(X, Y)

x_array, y_array = split_data(X, Y, n+1)

x_train, x_test, y_train, y_test = test_train(x_array, y_array, n)

train_local_hospitals(local_hospitals, x_train, x_test, y_train, y_test, n)

for i in local_hospitals:
	i.generate_proof(main_port)

time.sleep(100)
