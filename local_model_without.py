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
from imblearn.pipeline import Pipeline
from imblearn.under_sampling import RandomUnderSampler  
from imblearn.over_sampling import SMOTE
from sklearn.metrics import confusion_matrix
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from threading import Lock

class local_mod:
	def __init__(self, id, port, other_ports, global_port):
		self.lock=threading.Lock()
		self.id = id
		self.port = port
		self.other_ports = other_ports
		self.global_port = global_port
		self.blinding_factor = 1
		self.valid_models = list()
		self.partial_r = 0
		self.partial_fi = list()
		self.partial_no = 0
		self.last_accuracy = 0
		
		self.proof_sent = {}
		self.partial_sum_sent={}
		self.range_from_global = []
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
		client_socket.close()

		print(f"🔍 LOCAL Hospital {self.id} received message type {data['type']} from port {data.get('port', 'unknown')}")

		
		if (data["type"]==4):
			print(f'\nModel {self.id} validity: {data["validity"]}\n\n')
			print(f'🔍 Hospital {self.id}: Received validity: {data["validity"]}')
		# elif (data["type"]==5):
		# 	print(f"🔍 Hospital {self.id}: Received valid ports - STARTING MPC")
		# 	print(f"🔍 Hospital {self.id}: Valid models = {data['valid models']}")
		# 	self.partial_r=0
		# 	self.partial_fi =[0]*len(self.get_feature_importances())
		# 	self.partial_no=0
		# 	self.split_and_send_data(data)
		# elif (data["type"]==6):
		# 	print(f"🔥 Hospital {self.id} Recieved partial data from port {data['port']}")
		# 	if not hasattr(self, 'valid_models') or len(self.valid_models) == 0:
		# 		print(f"[WARNING] Hospital {self.id} received partial data but no valid models set")
		# 		return
		# 	self.update_partial_sum(data)
		# 	print(f"🔥 Hospital {self.id}: partial_no = {self.partial_no}/{len(self.valid_models)}")
			
		# 	if self.partial_no == len(self.valid_models):
		# 		# FIX: Check if we already sent partial sum for this round
		# 		if self.partial_sum_sent.get(self.current_round, False):
		# 			print(f"[WARNING] Hospital {self.id} already sent partial sum for round {self.current_round}")
		# 			return		

		# 		print(f"Partial r for local hospital {self.id}: {self.partial_r}")
		# 		print(f"Partial feature importances for local hospital {self.id}: {self.partial_fi}")
		# 		self.send_partial_sum(self.current_round)
		# 		self.partial_sum_sent[self.current_round] = True
				
		# 		# Reset after sending
		# 		self.partial_r = 0
		# 		self.partial_fi = [0]*len(self.get_feature_importances())
		# 		self.partial_no = 0
        
				
		else:
			print("Random access recieved.")

	def send_feature_importances(self, current_round):
			feature_importances = self.get_feature_importances()

			if current_round >= 4 and self.id in [1, 2]:
				print(f"🚨 MALICIOUS: Hospital {self.id} is manipulating feature importances in round {current_round}")
				feature_importances = self.apply_malicious_modifications(feature_importances, current_round)


			data = {
			"port": self.port,
			"type": 10,  # Use a new type number (say 10) to represent feature importances sent without proofs
			"feature_importances": feature_importances,
			"round": current_round
			}
			data_encoded = json.dumps(data).encode()
			print(f"Sending feature importances for round {current_round} from hospital {self.id}...")
			with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
				try:
					s.connect(('localhost', self.global_port))
					s.sendall(data_encoded)
					print(f"Feature importances sent successfully for round {current_round}.")
				except ConnectionRefusedError:
					print(f"Local model {self.id} could not connect to Node on port {self.global_port}")

	def apply_malicious_modifications(self, feature_importances, current_round):
		"""Apply malicious modifications to feature importances - Set certain columns to 0"""
		original_fi = feature_importances.copy()
		
		
		if self.id == 1:  # Hospital 1 malicious behavior
			print(f"🚨 H1 MALICIOUS: Original FI = {original_fi}")
			
			# Set columns 0, 2, and 4 to zero
			malicious_columns = [0,2,4,6,7]
			for col in malicious_columns:
				if col < len(feature_importances):
					feature_importances[col] = 1005
			
			print(f"🚨 H1 MALICIOUS: Set columns {malicious_columns} to 1005")
			
			print(f"🚨 H1 MALICIOUS: Modified FI = {feature_importances}")
			
		elif self.id == 2:  # Hospital 2 malicious behavior
			print(f"🚨 H2 MALICIOUS: Original FI = {original_fi}")
			
			# Set columns 1, 3, and 5 to zero
			malicious_columns = [0,2,4,6,7]
			for col in malicious_columns:
				if col < len(feature_importances):
					feature_importances[col] = 1005
			
			print(f"🚨 H2 MALICIOUS: Set columns {malicious_columns} to 0")
			print(f"🚨 H2 MALICIOUS: Modified FI = {feature_importances}")
			
		return feature_importances



	def train(self, x_train, x_test, y_train, y_test):
	# Safety check: reshape if single sample passed
				if x_test.ndim == 1:
					x_test = x_test.reshape(1, -1)
				if x_train.ndim == 1:
						x_train = x_train.reshape(1, -1)
						y_train = np.array([y_train])  # Ensure 1D

				self.model.fit(x_train, y_train)
				y_pred_proba = self.model.predict_proba(x_test)[:, 1]
				threshold = 0.5
				y_pred = (y_pred_proba >= threshold).astype(int)
				accuracy = accuracy_score(y_test, y_pred)
				self.last_accuracy = accuracy
				print(f"Accuracy for Hospital {self.id}: {accuracy:.2f}")
				print(classification_report(y_test, y_pred))
				cm = confusion_matrix(y_test, y_pred)
				print(f"Confusion Matrix for Hospital {self.id}:\n{cm}")
				print(f"\nFeature importances: {self.model.feature_importances_.tolist()}")
                
	def get_feature_importances(self):
		feature_importances = self.model.feature_importances_.tolist()
		feature_importances = [int(i*(10**6)) for i in feature_importances]
		return feature_importances
	# def generate_proof(self,current_round):  
	# 	if self.port == 6001:
	# 		print(f"🔍 [H1-LOCAL-DEBUG] Starting generate_proof for round {current_round}")
	# 		print(f"🔍 [H1-LOCAL-DEBUG] Current self.current_round: {getattr(self, 'current_round', 'NOT_SET')}")
	# 		print(f"🔍 [H1-LOCAL-DEBUG] Proof already sent for this round: {self.proof_sent.get(current_round, False)}")
    	
	# 	if self.proof_sent.get(current_round, False):
	# 		if self.port == 6001:
	# 			print(f"🔍 [Hospital {self.id}] Already sent proof for round {current_round}")
	# 			return  
		
	
	# 	feature_importances=self.get_feature_importances()
	# 	# r = randbelow(curve_order)
	# 	# self.blinding_factor = r
	# 	# commitments=[]

	# 	# print(f"\nGenerating commitments for hospital {self.id} (Round {current_round})...")
	# 	# for i in range(len(feature_importances)):
	# 	# 	commitment = pedersen_commit(feature_importances[i], r, G, H)
	# 	# 	commitments.append(commitment)
	# 	# while len(self.commitments_log) <= current_round:
	# 	# 	self.commitments_log.append([])
	# 	# self.commitments_log[current_round] = commitments

	# 	# print("Commitments\n", commitments)
	# 	# self.proof_sent[current_round] = True
	# 	# commitment_transaction = {"port":self.port, "type": 1, "commitments":commitments,"round":current_round}
	# 	# commitment_transaction_serialized = copy.deepcopy(commitment_transaction)
	# 	#serialize_ZKP_json(commitment_transaction_serialized)
	# 	#commitment_transaction_data = json.dumps(commitment_transaction_serialized).encode()
	# 	# print(f"Commitment transaction data {commitment_transaction_data}")
	# 	# print(f"Sending commitments for Round {current_round}.....")
		
		
	# 	proofs = []
	# 	print(f"\nGenerating proofs for hospital {self.id} (Round {current_round})...")
	# 	for i in range(len(feature_importances)):
	# 		print(f"\nProof for feature importance {i+1}")
			
	# 		print("FI: ", feature_importances[i])
	# 		proof = create_proof(feature_importances[i], r, self.range_from_global[i][0], self.range_from_global[i][1], commitments[i], G, H)
	# 		#print(proof)
	# 		print(f"Proof {i+1} generated.")
	# 		proofs.append(proof)
	# 	#print("Proofs",proofs)
	# 	proof_transaction = {"port":self.port, "type": 3, "proofs":proofs}
	# 	proof_transaction_serialized = copy.deepcopy(proof_transaction)
	# 	serialize_ZKP_json(proof_transaction_serialized)
	# 	proof_transaction_data = json.dumps(proof_transaction_serialized).encode()
	# 	print("Sending proofs...")
	# 	with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
	# 		try:
	# 			s.connect(('localhost', self.global_port))
	# 			s.sendall(proof_transaction_data)
	# 			print(f"Proofs sent successfully for Round {current_round}.")
	# 		except ConnectionRefusedError:
	# 			print(f"Local model {self.id} could not connect to Node on port {self.global_port}")
	# 	time.sleep(2)
	# # def split_and_send_data(self, data):
	# # 	with self.lock:
	# # 	    self.valid_models = data["valid models"]
	# # 	    n = len(data["valid models"])
	# # 	    print(f"Length of valid models {n}")

	# # 	    if n==0:
	# # 	        print(f"[ERROR] No valid models for hospital {self.id}")
	# # 	        return
	# # 	    elif n==1:
	# # 	        r_list = [self.blinding_factor]
	# # 	        fi_list = [[fi] for fi in self.get_feature_importances()]
	# # 	    else:
	# # 	        r_list = split_value(self.blinding_factor, n)
	# # 	        fi_list = [split_value(i, n) for i in self.get_feature_importances()]
	# # 	    for i in range(n):
	# # 		    MPC_data = {"type":6, "port": self.port, "r":r_list[i], "feature importance":[fi[i] for fi in fi_list]}
	# # 		    MPC_data_encoded = json.dumps(MPC_data).encode()
	# # 		    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
	# # 			    try:
	# # 				    s.connect(('localhost', data["valid models"][i]))
	# # 				    s.sendall(MPC_data_encoded)
	# # 			    except ConnectionRefusedError:
	# # 				    print(f"Local model {self.id} could not connect to Node on port {data['valid models'][i]}")
	# # 	    print("\nSplitting and sending data...")
	# # 	    print(f"Valid models: {self.valid_models}")
	# # 	    print(f"Number of valid models: {n}")
	# # 	    print(f"Blinding factor splits: {r_list}")
	# # 	    print(f"Feature importance splits: {fi_list}")
	# # 	    print("Data sent successfully.")
	def update_partial_sum(self, data):
		self.partial_no += 1
		self.partial_r += data["r"]
		for i in range(len(self.partial_fi)):
			self.partial_fi[i] += data["feature importance"][i]
	def send_partial_sum(self,round_number):
		partial_sum = {"type":7, "port":self.port, "partial r": self.partial_r, "partial fi":self.partial_fi,"round":round_number}
		partial_sum_encoded = json.dumps(partial_sum).encode()
		with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
			try:
				s.connect(('localhost', self.global_port))
				s.sendall(partial_sum_encoded)
				print(f"✅ Hospital {self.id}: Successfully sent partial sum to global")
			except ConnectionRefusedError:
				print(f"Local model {self.id} could not connect to Node on port {self.global_port}")
				print(f"❌ Hospital {self.id}: Could not connect to global port {self.global_port}")
			except Exception as e:
				print(f"❌ Hospital {self.id}: Error sending partial sum: {e}")

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
	smote = Pipeline([
	('smote', SMOTE(random_state=9)),
	('undersample', RandomUnderSampler(random_state=9))
])
	x_train, y_train = smote.fit_resample(x_train, y_train)
	return x_train, x_test, y_train

def get_partial_data(X, Y, fraction=0.2):
    total = len(X)
    count = int(total * fraction)
    indices = np.random.choice(total, count, replace=False)
    return X[indices], Y[indices]


def create_hospitals(num, base_port=5000):
	hospitals = []
	ports = [base_port+i+1 for i in range(num)]
	for i in range(num):
		other = ports[:i] + ports[i+1:]
		hospital = local_mod(i+1, ports[i], other, base_port)
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
	if n<=1:
		return [val]

	if val<=n:
		raise ValueError(f"Cannot split {value} into {n} parts. Too small.")
	split_points = sorted(random.randint(0, val) for _ in range(n - 1))
	parts = [split_points[0]]
	for i in range(1, len(split_points)):
		parts.append(split_points[i] - split_points[i - 1])
	parts.append(val - split_points[-1])
	return parts

n = 3
main_port = 6000
local_hospitals = create_hospitals(n, main_port)
time.sleep(0.5)

path = "Sample Data/diabetes_prediction_dataset.csv"
X, Y = load_data(path)

X, Y = preprocess_data(X, Y)
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.25,random_state=1)#75 percent training and 20 percent training



x_array, y_array = split_data(x_train, y_train, n) # Split the training data for 3 local hospitals into 3 parts
x_test_array, y_test_array = split_data(x_test, y_test, n)


T = 5  # Number of partitions
n = len(x_array)  # Number of hospitals

# Step 1: Pre-split each hospital’s data
all_local_parts_x = []
all_local_parts_y = []

for i in range(n):
    parts_x = np.array_split(x_array[i], T)
    parts_y = np.array_split(y_array[i], T)
    all_local_parts_x.append(parts_x)
    all_local_parts_y.append(parts_y)

# Step 2: Iterate over each partition number (P1 to P5)
for part_no in range(T):
    print(f"\n=== ROUND {part_no + 1} — Collecting data from Partition P{part_no + 1} ===")
    
    

    hospital_sample_map=[] # used when training each part to know where it came from. Check the for loop below containing the idx

    # Step 3: For each hospital i, take its current partition part_no
    for i in range(n):
        # Step 3: For each hospital i, get its current partition
        part_x = all_local_parts_x[i][part_no]
        part_y = all_local_parts_y[i][part_no]

        print(f"Hospital {i+1} contributes {len(part_x)} samples to P{part_no + 1}")

        # Step 4: Optional preprocessing (scaling, SMOTE)
        part_x, _, part_y = scale_and_synthesize(part_x, part_x, part_y)

        # Step 5a: Train local model on its full partition
        print(f"Training Hospital {i+1} on P{part_no + 1}...")
        local_hospitals[i].train(part_x, part_x, part_y, part_y)
        local_hospitals[i].current_round = part_no+1


        # Step 5b: Generate ZKP proof for that trained model
        print(f"Generating feature importances for Hospital {i+1} on P{part_no + 1}.	..")
        local_hospitals[i].send_feature_importances(part_no+1)
        time.sleep(3)
        

time.sleep(10)


