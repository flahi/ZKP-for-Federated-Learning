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
from imblearn.pipeline import Pipeline
from imblearn.under_sampling import RandomUnderSampler  
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score,classification_report
from imblearn.over_sampling import SMOTE
from sklearn.metrics import confusion_matrix
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

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
		self.commitments_rounds_log = []
		self.commitments_lock = threading.Lock()

		self.valid_models = list()
		self.no_of_local = n
		self.no_of_local_recieved = 0
		self.no_of_proofs_received = 0  # Track proof submissions
		self.total_r = 0
		self.total_fi = list()
		self.validity_map = dict()
		
		self.ranges = []
		self.feature_importances_map = dict()
		self.current_round = 1 # Track current partition round
		self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
		self.server_socket.bind(('localhost', self.port))
		self.server_socket.listen()
		self.global_x_train=None
		self.global_x_test=None
		self.global_y_train=None
		self.global_y_test=None
		self.received_partial_sums=set()
		threading.Thread(target=self.listen_for_data, daemon=True).start()

	def set_global_data(self, x_train, x_test, y_train, y_test):
		self.global_x_train = x_train.copy()
		self.global_x_test = x_test.copy()
		self.global_y_train = y_train.copy()
		self.global_y_test = y_test.copy()
		print("Global training and test data stored for aggregation")
		
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

		if data.get("port") == 6001:  # H1's port
			print(f"🔍 [H1-DEBUG] Type {data['type']} from H1 (6001) in round {self.current_round}")
			print(f"🔍 [H1-DEBUG] Current commitments: {list(self.local_commitments.keys())}")
		
		# FIXED: Check for type 10 (what local hospitals are sending) and handle it properly
		if data["type"] == 10:
			port = data.get("port")
			feature_importances = data.get("feature_importances", [])
			current_round = data.get("round", self.current_round)
			print(f"📥 Global: Received feature importances from port {port} for round {current_round}")

			with self.commitments_lock:  # For thread safety
				self.feature_importances_map[port] = feature_importances

				# Check if all local nodes have sent their feature importances
				if len(self.feature_importances_map) == self.no_of_local:
					print("🎯 Global: All feature importances received, starting aggregation...")
					self.aggregate()
					self.feature_importances_map.clear()
					self.current_round += 1  # Move to next round
					print(f"✅ Global: Ready for round {self.current_round}")

		else:
			print(f"🔍 Global: Received message type {data['type']} - not handling")

	def reset_for_next_partition(self):
		"""Reset state variables for the next partition"""
		with self.commitments_lock:  # Ensure thread safety
			self.valid_models = list()
			self.no_of_local_recieved = 0
			# Don't reset no_of_proofs_received here
			self.total_r = 0
			self.total_fi = [0] * len(self.get_feature_importances()) if hasattr(self, 'model') and hasattr(self.model, 'feature_importances_') else []
			self.validity_map = dict()
			self.feature_importances_map = dict()
			self.received_partial_sums = set()
        
			# Only reset commitments if we've received all proofs
			if self.no_of_proofs_received >= self.no_of_local and self.no_of_local_recieved >=len(self.valid_models):
				# self.local_commitments = dict()
				self.no_of_proofs_received = 0  # Reset it here instead
            
		self.current_round += 1  # Increment round counter
		print(f"✅ Global: Reset complete for Round {self.current_round}")
		print(f"✅ Global: Ready for next partition")

	def send_valid_ports(self):
		valid_model_data = {"type":5, "port":self.port, "valid models":self.valid_models}
		valid_model_data_encoded = json.dumps(valid_model_data).encode()
		print(f"\n\nValid model ports for partition {self.current_round }: {self.valid_models}\n")
		if not self.valid_models:  # Explicit empty list check
			print("[WARNING] No valid models available.")
			self.train(self.global_x_train, self.global_x_test, self.global_y_train, self.global_y_test)
			time.sleep(3)
			return 

		for port in self.valid_models:
			with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
				try:
					s.connect(('localhost', port))
					s.sendall(valid_model_data_encoded)
				except ConnectionRefusedError:
					print(f"Global model could not connect to Node on port {port}")

	def train(self, x_train, x_test, y_train, y_test, sw=None, check=False):
		self.model.fit(x_train, y_train, sample_weight=sw)
		y_pred_proba = self.model.predict_proba(x_test)[:, 1]
		threshold = 0.75
		y_pred = (y_pred_proba >= threshold).astype(int)
		accuracy = accuracy_score(y_test, y_pred)
		tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
		actual_positives=tp+fn
		actual_negatives=tn+fp
		recall = recall_score(y_test, y_pred)
		precision = precision_score(y_test, y_pred)  # Missing in your original code!
		f1 = f1_score(y_test, y_pred)  # Directly use f1_score instead of manual calculation
    	
		# Print results
		print(f"Accuracy: {accuracy:.4f}")
		print(f"True Positives (TP): {tp}")
		print(f"True Negatives (TN): {tn}")
		print(f"False Positives (FP): {fp}")
		print(f"False Negatives (FN): {fn}")
		print(f"Recall: {recall:.4f}")
		print(f"Actual Positive {actual_positives}")
		print(f"Acutal Negative {actual_negatives}")
		print(f"Precision: {precision:.4f}")
		print(f"F1-Score: {f1:.4f}")
		feature_importances = self.get_feature_importances()
		self.total_fi = [0]*len(feature_importances)
	
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
		print(f"\n\nRetraining global model after round {self.current_round}...\n")
		
		# Use the global test data for retraining
		scaled_importances = aggregated_feature_importances * 100
		sample_weights = np.dot(self.global_x_train, scaled_importances)
		sample_weights = (sample_weights - np.min(sample_weights)) / (np.max(sample_weights) - np.min(sample_weights))
		self.train(self.global_x_train, self.global_x_test, self.global_y_train, self.global_y_test, sample_weights)

	def calculate_aggregated_fi(self):
		# Calculate average of feature importances from all local models
		all_feature_importances = list(self.feature_importances_map.values())
		
		# Convert to numpy array for easier computation
		fi_array = np.array(all_feature_importances)
		
		# Calculate mean across all hospitals
		mean_fi = np.mean(fi_array, axis=0)
		
		# Convert back to the same scale as used in training (divide by 10^6)
		aggregated = mean_fi / (10**6)
		
		# Normalize to sum to 1
		aggregated = aggregated / np.sum(aggregated)
		
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
	return X, Y

def split_data(X, Y, n):
	print(f"\nSplitting data into {n}")
	x_array = np.array_split(X, n)
	y_array = np.array_split(Y, n)
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
	smote = Pipeline([
	('smote', SMOTE(random_state=9)),
	('undersample', RandomUnderSampler(random_state=9))
])
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
global_model.set_global_data(x_train[n],x_test[n],y_train[n],y_test[n])

print("Training initial global model...")
global_model.train(x_train[n], x_test[n], y_train[n], y_test[n])
print("Initial training complete.")
print("\nWaiting for local models...")

time.sleep(1000)