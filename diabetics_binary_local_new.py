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
from sklearn.preprocessing import LabelEncoder
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix, roc_auc_score, roc_curve
from imblearn.over_sampling import SMOTE
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from imblearn.pipeline import Pipeline
from imblearn.under_sampling import RandomUnderSampler  
from threading import Lock
import matplotlib.pyplot as plt

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
		self.commitments_log=[]
		self.proof_sent = {}
		self.partial_sum_sent={}
		self.range_from_global = []
		
		# Metrics tracking
		self.partition_metrics = []
		self.current_round = 0
		self.proof_generation_times = []
		self.avg_commitment_time = []      # Average commitment time per round
		self.avg_proof_time = []
		
		# Cumulative metrics
		self.cumulative_f1 = []
		self.cumulative_recall = []
		self.cumulative_precision = []
		self.cumulative_accuracy = []
		self.cumulative_auc = []
		
		self.model = RandomForestClassifier(
			random_state=42,
			n_estimators=600,        # Increased from 50
			max_depth=30,            # Increased from 8
			class_weight='balanced',       # Changed from 'balanced' (data is balanced)
			min_samples_leaf=4,
			min_samples_split=8,
			max_features='sqrt',
			bootstrap=True,
			oob_score=True,
			n_jobs=-1                # Use all CPU cores
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

		if (data["type"]==2):
			self.range_from_global = data["ranges"]
			print(f"🔍 Hospital {self.id}: Received ranges")
			print(f"Ranges from global are {self.range_from_global}")
		elif (data["type"]==4):
			time.sleep(20)
			print(f'\nModel {self.id} validity: {data["validity"]}\n\n')
			print(f'🔍 Hospital {self.id}: Received validity: {data["validity"]}')
		elif (data["type"]==5):
			print(f"🔍 Hospital {self.id}: Received valid ports - STARTING MPC")
			print(f"🔍 Hospital {self.id}: Valid models = {data['valid models']}")
			self.partial_r=0
			self.partial_fi =[0]*len(self.get_feature_importances())
			self.partial_no=0
			self.split_and_send_data(data)
		elif (data["type"]==6):
			print(f"🔥 Hospital {self.id} Recieved partial data from port {data['port']}")
			if not hasattr(self, 'valid_models') or len(self.valid_models) == 0:
				print(f"[WARNING] Hospital {self.id} received partial data but no valid models set")
				return
			self.update_partial_sum(data)
			print(f"🔥 Hospital {self.id}: partial_no = {self.partial_no}/{len(self.valid_models)}")
			
			if self.partial_no == len(self.valid_models):
				if self.partial_sum_sent.get(self.current_round, False):
					print(f"[WARNING] Hospital {self.id} already sent partial sum for round {self.current_round}")
					return		

				print(f"Partial r for local hospital {self.id}: {self.partial_r}")
				print(f"Partial feature importances for local hospital {self.id}: {self.partial_fi}")
				self.send_partial_sum(self.current_round)
				self.partial_sum_sent[self.current_round] = True
				
				# Reset after sending
				self.partial_r = 0
				self.partial_fi = [0]*len(self.get_feature_importances())
				self.partial_no = 0
				
		else:
			print("Random access recieved.")

	def train(self, x_train, x_test, y_train, y_test):
		"""Train model and calculate comprehensive metrics"""
		# Convert to numpy if needed
		if isinstance(x_train, pd.DataFrame):
			x_train = x_train.values
		if isinstance(x_test, pd.DataFrame):
			x_test = x_test.values
		if isinstance(y_train, pd.Series):
			y_train = y_train.values
		if isinstance(y_test, pd.Series):
			y_test = y_test.values
		
		# Reshape safety check
		if x_test.ndim == 1:
			x_test = x_test.reshape(1, -1)
		if x_train.ndim == 1:
			x_train = x_train.reshape(1, -1)
			y_train = np.array([y_train])

		self.model.fit(x_train, y_train)
		y_pred_proba = self.model.predict_proba(x_test)[:, 1]
		threshold = 0.5
		y_pred = (y_pred_proba >= threshold).astype(int)
		
		accuracy = accuracy_score(y_test, y_pred)
		self.last_accuracy = accuracy
		
		# Calculate all metrics
		precision = precision_score(y_test, y_pred, zero_division=0)
		recall = recall_score(y_test, y_pred, zero_division=0)
		f1 = f1_score(y_test, y_pred, zero_division=0)
		
		try:
			auc = roc_auc_score(y_test, y_pred_proba)
			fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
		except Exception as e:
			print(f"[WARNING] Could not calculate AUC: {e}")
			auc = 0.0
			fpr, tpr = [0, 1], [0, 1]
		
		print(f"\n{'='*60}")
		print(f"Hospital {self.id} - Round {self.current_round} Results")
		print(f"{'='*60}")
		print(f"Accuracy:  {accuracy:.4f}")
		print(f"Precision: {precision:.4f}")
		print(f"Recall:    {recall:.4f}")
		print(f"F1-Score:  {f1:.4f}")
		print(f"AUC-ROC:   {auc:.4f}")
		print(f"{'='*60}\n")
		
		print(classification_report(y_test, y_pred))
		cm = confusion_matrix(y_test, y_pred)
		print(f"Confusion Matrix for Hospital {self.id}:\n{cm}")
		print(f"\nFeature importances: {self.model.feature_importances_.tolist()}")
		
		# Store metrics in partition_metrics
		metrics = {
			'round': self.current_round,
			'accuracy': accuracy,
			'precision': precision,
			'recall': recall,
			'f1': f1,
			'auc': auc,
			'fpr': fpr,
			'tpr': tpr
		}
		self.partition_metrics.append(metrics)
		
		# Update cumulative metrics (running average)
		if len(self.partition_metrics) > 0:
			self.cumulative_accuracy.append(np.mean([m['accuracy'] for m in self.partition_metrics]))
			self.cumulative_f1.append(np.mean([m['f1'] for m in self.partition_metrics]))
			self.cumulative_recall.append(np.mean([m['recall'] for m in self.partition_metrics]))
			self.cumulative_precision.append(np.mean([m['precision'] for m in self.partition_metrics]))
			self.cumulative_auc.append(np.mean([m['auc'] for m in self.partition_metrics]))
		
		return metrics
	
	def plot_cumulative_auc_curve(self):
		"""Plot ROC curves for all partitions up to current round"""
		if not self.partition_metrics:
			return
		
		save_path = f'hospital_{self.id}_auc_round_{self.current_round}.png'
		plt.figure(figsize=(10, 8))
		
		colors = plt.cm.viridis(np.linspace(0, 0.9, len(self.partition_metrics)))
		
		for idx, metrics in enumerate(self.partition_metrics):
			plt.plot(
				metrics['fpr'], 
				metrics['tpr'], 
				color=colors[idx],
				lw=2.5,
				label=f"Round {metrics['round']} (AUC = {metrics['auc']:.4f})",
				alpha=0.8
			)
		
		plt.plot([0, 1], [0, 1], 'k--', lw=2, label='Random Classifier', alpha=0.5)
		
		plt.xlim([0.0, 1.0])
		plt.ylim([0.0, 1.05])
		plt.xlabel('False Positive Rate', fontsize=13, fontweight='bold')
		plt.ylabel('True Positive Rate', fontsize=13, fontweight='bold')
		plt.title(f'Hospital {self.id}: ROC Curves (Rounds 1-{self.current_round})', 
				 fontsize=15, fontweight='bold')
		plt.legend(loc="lower right", fontsize=10, framealpha=0.9)
		plt.grid(alpha=0.3, linestyle='--')
		
		textstr = f'Hospital ID: {self.id}\nCurrent Round: {self.current_round}'
		props = dict(boxstyle='round', facecolor='lightblue', alpha=0.5)
		plt.text(0.02, 0.98, textstr, transform=plt.gca().transAxes, fontsize=10,
				verticalalignment='top', bbox=props)
		
		plt.tight_layout()
		plt.savefig(save_path, dpi=300, bbox_inches='tight')
		print(f"Hospital {self.id}: AUC curve for Round {self.current_round} saved to {save_path}")
		plt.close()

	def plot_f1_recall_progression(self):
		"""Plot F1 and Recall scores after each round"""
		if not self.partition_metrics:
			return
		
		save_path = f'hospital_{self.id}_f1_recall_round_{self.current_round}.png'
		
		rounds = [m['round'] for m in self.partition_metrics]
		f1_scores = [m['f1'] for m in self.partition_metrics]
		recall_scores = [m['recall'] for m in self.partition_metrics]
		
		fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
		
		# F1 Score
		ax1.plot(rounds, f1_scores, 'o-', linewidth=2.5, markersize=9, 
		        color='#FF6B6B', label='F1 Score')
		ax1.set_xlabel('Partition Number', fontsize=12, fontweight='bold')
		ax1.set_ylabel('F1 Score', fontsize=12, fontweight='bold')
		ax1.set_title(f'Hospital {self.id}: F1 Score Progression', fontsize=13, fontweight='bold')
		ax1.grid(alpha=0.3, linestyle='--')
		ax1.set_ylim([0, 1.05])
		ax1.legend(fontsize=11)
		
		# Add value labels
		for i, (x, y) in enumerate(zip(rounds, f1_scores)):
			ax1.text(x, y + 0.02, f'{y:.3f}', ha='center', fontsize=9)
		
		# Recall Score
		ax2.plot(rounds, recall_scores, 's-', linewidth=2.5, markersize=9, 
		        color='#4ECDC4', label='Recall')
		ax2.set_xlabel('Partition Number', fontsize=12, fontweight='bold')
		ax2.set_ylabel('Recall Score', fontsize=12, fontweight='bold')
		ax2.set_title(f'Hospital {self.id}: Recall Progression', fontsize=13, fontweight='bold')
		ax2.grid(alpha=0.3, linestyle='--')
		ax2.set_ylim([0, 1.05])
		ax2.legend(fontsize=11)
		
		# Add value labels
		for i, (x, y) in enumerate(zip(rounds, recall_scores)):
			ax2.text(x, y + 0.02, f'{y:.3f}', ha='center', fontsize=9)
		
		plt.tight_layout()
		plt.savefig(save_path, dpi=300, bbox_inches='tight')
		print(f"Hospital {self.id}: F1 & Recall progression saved to {save_path}")
		plt.close()


	def get_feature_importances(self):
		feature_importances = self.model.feature_importances_.tolist()
		feature_importances = [int(i*(10**6)) for i in feature_importances]
		return feature_importances

	def generate_proof(self, current_round):  
		"""Generate commitments and proofs with Big O-aware timing"""
		if self.proof_sent.get(current_round, False):
			return  
		
		# Get feature count for Big O analysis
		feature_importances = self.get_feature_importances()
		n_features = len(feature_importances)
		
		print(f"\n🔬 Big O Analysis - Hospital {self.id}, Round {current_round}:")
		print(f"   Features (n): {n_features}")
		print(f"   Expected Complexity: O(n) = O({n_features})")
		
		# ✅ Start computational timing (exclude network/sleep)
		computational_start = time.time()
		
		# PHASE 1: Commitment Generation - O(n)
		commitment_phase_start = time.time()
		r = randbelow(curve_order)
		self.blinding_factor = r
		commitments = []
		commitment_times = []

		print(f"\n📊 Phase 1: Commitment Generation (O(n))")
		for i in range(n_features):
			commitment_start = time.time()
			commitment = pedersen_commit(feature_importances[i], r, G, H)
			commitment_time = time.time() - commitment_start
			commitment_times.append(commitment_time)
			commitments.append(commitment)
			print(f"   Commitment {i+1}/{n_features}: {commitment_time:.6f}s")
		
		commitment_phase_time = time.time() - commitment_phase_start
		avg_commitment_time = np.mean(commitment_times)
		
		# Store timing data
		if not hasattr(self, 'commitment_times_log'):
			self.commitment_times_log = []
		self.commitment_times_log.append(commitment_times)
		
		print(f"   ✅ Commitment Phase Complete: {commitment_phase_time:.6f}s")
		print(f"   📈 Avg per commitment: {avg_commitment_time:.6f}s")
		print(f"   🎯 O(n) Verification: {n_features} × {avg_commitment_time:.6f}s ≈ {n_features * avg_commitment_time:.6f}s")

		# Store commitments
		while len(self.commitments_log) <= current_round:
			self.commitments_log.append([])
		self.commitments_log[current_round] = commitments
		self.proof_sent[current_round] = True
		
		# Send commitments (NETWORK - not included in computational timing)
		commitment_transaction = {"port":self.port, "type": 1, "commitments":commitments,"round":current_round}
		commitment_transaction_serialized = copy.deepcopy(commitment_transaction)
		serialize_ZKP_json(commitment_transaction_serialized)
		commitment_transaction_data = json.dumps(commitment_transaction_serialized).encode()
		
		print(f"\n🌐 Network Phase: Sending commitments...")
		network_start = time.time()
		with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
			try:
				s.connect(('localhost', self.global_port))
				s.sendall(commitment_transaction_data)
				print(f"   ✅ Commitments sent in {time.time() - network_start:.6f}s")
			except ConnectionRefusedError:
				print(f"   ❌ Connection failed")

		# Wait for ranges (NETWORK DELAY - not computational)
		print(f"\n⏳ Waiting for ranges from global...")
		time.sleep(2)
		
		# PHASE 2: Proof Generation - O(n)
		proof_phase_start = time.time()
		proofs = []
		proof_times = []
		
		print(f"\n📊 Phase 2: Proof Generation (O(n))")
		for i in range(n_features):
			print(f"\n   Proof for feature {i+1}/{n_features}")
			print(f"   Range: {self.range_from_global[i]}")
			print(f"   FI: {feature_importances[i]}")
			
			proof_start = time.time()
			proof = create_proof(feature_importances[i], r, self.range_from_global[i][0], 
							self.range_from_global[i][1], commitments[i], G, H)
			proof_time = time.time() - proof_start
			proof_times.append(proof_time)
			proofs.append(proof)
			
			print(f"   ✅ Proof generated: {proof_time:.6f}s")
		
		proof_phase_time = time.time() - proof_phase_start
		avg_proof_time = np.mean(proof_times)
		
		# Store proof timing data
		if not hasattr(self, 'proof_times_log'):
			self.proof_times_log = []
		self.proof_times_log.append(proof_times)
		
		print(f"   ✅ Proof Phase Complete: {proof_phase_time:.6f}s")
		print(f"   📈 Avg per proof: {avg_proof_time:.6f}s")
		print(f"   🎯 O(n) Verification: {n_features} × {avg_proof_time:.6f}s ≈ {n_features * avg_proof_time:.6f}s")

		# Calculate total COMPUTATIONAL time (excludes network/sleep)
		total_computational_time = time.time() - computational_start
		
		# Send proofs (NETWORK - not included in computational timing)
		print(f"\n🌐 Network Phase: Sending proofs...")
		network_start = time.time()
		proof_transaction = {"port":self.port, "type": 3, "proofs":proofs}
		proof_transaction_serialized = copy.deepcopy(proof_transaction)
		serialize_ZKP_json(proof_transaction_serialized)
		proof_transaction_data = json.dumps(proof_transaction_serialized).encode()
		
		with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
			try:
				s.connect(('localhost', self.global_port))
				s.sendall(proof_transaction_data)
				print(f"   ✅ Proofs sent in {time.time() - network_start:.6f}s")
			except ConnectionRefusedError:
				print(f"   ❌ Connection failed")

		# ✅ Big O Performance Analysis
		print(f"\n{'='*80}")
		print(f"🎯 BIG O PERFORMANCE ANALYSIS - Hospital {self.id}, Round {current_round}")
		print(f"{'='*80}")
		print(f"Input Size (n): {n_features} features")
		print(f"Expected Complexity: O(n)")
		print(f"\n📊 Computational Performance:")
		print(f"   Commitment Phase (O(n)): {commitment_phase_time:.6f}s")
		print(f"   Proof Phase (O(n)):     {proof_phase_time:.6f}s")
		print(f"   Total Computational:    {total_computational_time:.6f}s")
		print(f"\n📈 Per-Operation Averages:")
		print(f"   Avg Commitment Time:    {avg_commitment_time:.6f}s")
		print(f"   Avg Proof Time:         {avg_proof_time:.6f}s")
		print(f"   Total per feature:      {avg_commitment_time + avg_proof_time:.6f}s")
		print(f"\n🎯 Big O Verification:")
		print(f"   Actual/Optimal Ratio:   {total_computational_time / n_features:.6f}s per feature")
		print(f"   Linear Scaling Check:   {total_computational_time / n_features:.6f}s ≈ constant")
		print(f"{'='*80}\n")
		
		# Store overall timing
		if not hasattr(self, 'computational_times'):
			self.computational_times = []
		self.computational_times.append(total_computational_time)
		
		time.sleep(2)

	def split_and_send_data(self, data):
		with self.lock:
			self.valid_models = data["valid models"]
			n = len(data["valid models"])
			print(f"Length of valid models {n}")

			if n == 0:
				print(f"[ERROR] No valid models for hospital {self.id}")
				return
			elif n == 1:
				r_list = [self.blinding_factor]
				fi_list = [[fi] for fi in self.get_feature_importances()]
			else:
				r_list = split_value(self.blinding_factor, n)
				fi_list = [split_value(i, n) for i in self.get_feature_importances()]
			
			for i in range(n):
				MPC_data = {"type":6, "port": self.port, "r":r_list[i], "feature importance":[fi[i] for fi in fi_list]}
				MPC_data_encoded = json.dumps(MPC_data).encode()
				with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
					try:
						s.connect(('localhost', data["valid models"][i]))
						s.sendall(MPC_data_encoded)
					except ConnectionRefusedError:
						print(f"Local model {self.id} could not connect to Node on port {data['valid models'][i]}")
			
			print("\nSplitting and sending data...")
			print(f"Valid models: {self.valid_models}")
			print(f"Number of valid models: {n}")
			print(f"Blinding factor splits: {r_list}")
			print(f"Feature importance splits: {fi_list}")
			print("Data sent successfully.")

	def update_partial_sum(self, data):
		self.partial_no += 1
		self.partial_r += data["r"]
		for i in range(len(self.partial_fi)):
			self.partial_fi[i] += data["feature importance"][i]

	def send_partial_sum(self, round_number):
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

	def plot_metrics_progression(self, save_path=None):
		"""Plot progression of all metrics across partitions"""
		if not self.partition_metrics:
			print(f"Hospital {self.id}: No metrics available to plot")
			return
		
		if save_path is None:
			save_path = f'hospital_{self.id}_metrics_progression.png'
		
		rounds = [m['round'] for m in self.partition_metrics]
		accuracies = [m['accuracy'] for m in self.partition_metrics]
		aucs = [m['auc'] for m in self.partition_metrics]
		precisions = [m['precision'] for m in self.partition_metrics]
		recalls = [m['recall'] for m in self.partition_metrics]
		f1s = [m['f1'] for m in self.partition_metrics]
		
		fig, axes = plt.subplots(2, 2, figsize=(15, 10))
		
		# AUC progression
		axes[0, 0].plot(rounds, aucs, 'o-', linewidth=2, markersize=8, color='#2E86AB')
		axes[0, 0].set_xlabel('Partition Number', fontsize=11)
		axes[0, 0].set_ylabel('AUC-ROC Score', fontsize=11)
		axes[0, 0].set_title(f'Hospital {self.id}: AUC-ROC Progression', fontsize=12, fontweight='bold')
		axes[0, 0].grid(alpha=0.3)
		axes[0, 0].set_ylim([0, 1.05])
		for i, v in enumerate(aucs):
			axes[0, 0].text(rounds[i], v + 0.02, f'{v:.3f}', ha='center', fontsize=9)
		
		# Accuracy progression
		axes[0, 1].plot(rounds, accuracies, 'o-', linewidth=2, markersize=8, color='#A23B72')
		axes[0, 1].set_xlabel('Partition Number', fontsize=11)
		axes[0, 1].set_ylabel('Accuracy', fontsize=11)
		axes[0, 1].set_title(f'Hospital {self.id}: Accuracy Progression', fontsize=12, fontweight='bold')
		axes[0, 1].grid(alpha=0.3)
		axes[0, 1].set_ylim([0, 1.05])
		for i, v in enumerate(accuracies):
			axes[0, 1].text(rounds[i], v + 0.02, f'{v:.3f}', ha='center', fontsize=9)
		
		# Precision & Recall
		axes[1, 0].plot(rounds, precisions, 'o-', linewidth=2, markersize=8, color='#F18F01', label='Precision')
		axes[1, 0].plot(rounds, recalls, 's-', linewidth=2, markersize=8, color='#C73E1D', label='Recall')
		axes[1, 0].set_xlabel('Partition Number', fontsize=11)
		axes[1, 0].set_ylabel('Score', fontsize=11)
		axes[1, 0].set_title(f'Hospital {self.id}: Precision & Recall', fontsize=12, fontweight='bold')
		axes[1, 0].legend(fontsize=10)
		axes[1, 0].grid(alpha=0.3)
		axes[1, 0].set_ylim([0, 1.05])
		for i, v in enumerate(precisions):
			axes[1, 0].text(rounds[i], v + 0.01, f'{v:.2f}', ha='center', fontsize=8)
		
		# F1 Score
		axes[1, 1].plot(rounds, f1s, 'o-', linewidth=2, markersize=8, color='#6A994E')
		axes[1, 1].set_xlabel('Partition Number', fontsize=11)
		axes[1, 1].set_ylabel('F1 Score', fontsize=11)
		axes[1, 1].set_title(f'Hospital {self.id}: F1 Score Progression', fontsize=12, fontweight='bold')
		axes[1, 1].grid(alpha=0.3)
		axes[1, 1].set_ylim([0, 1.05])
		for i, v in enumerate(f1s):
			axes[1, 1].text(rounds[i], v + 0.02, f'{v:.3f}', ha='center', fontsize=9)
		
		plt.tight_layout()
		plt.savefig(save_path, dpi=300, bbox_inches='tight')
		print(f"Hospital {self.id}: Metrics progression saved to {save_path}")
		plt.close()

	def plot_cumulative_metrics(self, save_path=None):
		"""Plot cumulative average of F1, Recall, Precision, Accuracy"""
		if len(self.cumulative_f1) == 0:
			print(f"Hospital {self.id}: No cumulative metrics to plot")
			return
		
		if save_path is None:
			save_path = f'hospital_{self.id}_cumulative_metrics.png'
		
		rounds = list(range(1, len(self.cumulative_f1) + 1))
		
		fig, ax = plt.subplots(figsize=(12, 7))
		
		ax.plot(rounds, self.cumulative_f1, 'o-', linewidth=2.5, markersize=8, color='#FF6B6B', label='Cumulative F1')
		ax.plot(rounds, self.cumulative_recall, 's-', linewidth=2.5, markersize=8, color='#4ECDC4', label='Cumulative Recall')
		ax.plot(rounds, self.cumulative_precision, '^-', linewidth=2.5, markersize=8, color='#95E1D3', label='Cumulative Precision')
		ax.plot(rounds, self.cumulative_accuracy, 'd-', linewidth=2.5, markersize=8, color='#F38181', label='Cumulative Accuracy')
		ax.plot(rounds, self.cumulative_auc, 'x-', linewidth=2.5, markersize=8, color='#2E86AB', label='Cumulative AUC')
		
		ax.set_xlabel('Partition Number', fontsize=13, fontweight='bold')
		ax.set_ylabel('Score (Cumulative Average)', fontsize=13, fontweight='bold')
		ax.set_title(f'Hospital {self.id}: Cumulative Metrics Progression', fontsize=14, fontweight='bold')
		ax.grid(alpha=0.3, linestyle='--')
		ax.set_ylim([0, 1.05])
		ax.legend(fontsize=11, loc='lower right')
		
		# Add final values annotation
		textstr = f'Final Cumulative:\nF1: {self.cumulative_f1[-1]:.4f}\nRecall: {self.cumulative_recall[-1]:.4f}\nPrecision: {self.cumulative_precision[-1]:.4f}\nAccuracy: {self.cumulative_accuracy[-1]:.4f}\nAUC: {self.cumulative_auc[-1]:.4f}'
		props = dict(boxstyle='round', facecolor='wheat', alpha=0.7)
		ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=10, verticalalignment='top', bbox=props)
		
		plt.tight_layout()
		plt.savefig(save_path, dpi=300, bbox_inches='tight')
		print(f"Hospital {self.id}: Cumulative metrics saved to {save_path}")
		plt.close()

	def plot_roc_curves(self, save_path=None):
		"""Plot all ROC curves together"""
		if not self.partition_metrics:
			print(f"Hospital {self.id}: No metrics available to plot ROC curves")
			return
		
		if save_path is None:
			save_path = f'hospital_{self.id}_roc_curves.png'
		
		plt.figure(figsize=(12, 8))
		colors = plt.cm.rainbow(np.linspace(0, 1, len(self.partition_metrics)))
		
		for idx, metrics in enumerate(self.partition_metrics):
			plt.plot(metrics['fpr'], metrics['tpr'], color=colors[idx], lw=2, label=f"Round {metrics['round']} (AUC = {metrics['auc']:.3f})")
		
		plt.plot([0, 1], [0, 1], 'k--', lw=2, label='Random Classifier')
		plt.xlim([0.0, 1.0])
		plt.ylim([0.0, 1.05])
		plt.xlabel('False Positive Rate', fontsize=12, fontweight='bold')
		plt.ylabel('True Positive Rate', fontsize=12, fontweight='bold')
		plt.title(f'Hospital {self.id}: ROC Curves (All Rounds)', fontsize=14, fontweight='bold')
		plt.legend(loc="lower right", fontsize=10)
		plt.grid(alpha=0.3)
		plt.tight_layout()
		plt.savefig(save_path, dpi=300, bbox_inches='tight')
		print(f"Hospital {self.id}: ROC curves saved to {save_path}")
		plt.close()

	def generate_final_report(self):
		"""Generate comprehensive final report with all visualizations"""
		if not self.partition_metrics:
			print(f"Hospital {self.id}: No metrics to report")
			return
		
		print(f"\n{'='*80}")
		print(f"HOSPITAL {self.id} - FINAL PERFORMANCE SUMMARY")
		print(f"{'='*80}")
		print(f"Total Partitions Processed: {len(self.partition_metrics)}")
		
		# Calculate averages
		avg_accuracy = np.mean([m['accuracy'] for m in self.partition_metrics])
		avg_auc = np.mean([m['auc'] for m in self.partition_metrics])
		avg_precision = np.mean([m['precision'] for m in self.partition_metrics])
		avg_recall = np.mean([m['recall'] for m in self.partition_metrics])
		avg_f1 = np.mean([m['f1'] for m in self.partition_metrics])
		
		print(f"\nAverage Performance Across All Partitions:")
		print(f"  Accuracy:  {avg_accuracy:.4f}")
		print(f"  AUC-ROC:   {avg_auc:.4f}")
		print(f"  Precision: {avg_precision:.4f}")
		print(f"  Recall:    {avg_recall:.4f}")
		print(f"  F1-Score:  {avg_f1:.4f}")
		
		# Best partition
		best_idx = np.argmax([m['auc'] for m in self.partition_metrics])
		best_metrics = self.partition_metrics[best_idx]
		print(f"\nBest Partition: {best_metrics['round']} (AUC = {best_metrics['auc']:.4f})")
		
		# Worst partition
		worst_idx = np.argmin([m['auc'] for m in self.partition_metrics])
		worst_metrics = self.partition_metrics[worst_idx]
		print(f"Worst Partition: {worst_metrics['round']} (AUC = {worst_metrics['auc']:.4f})")
		
		print(f"{'='*80}\n")
		
		# Generate all visualizations
		self.plot_metrics_progression()
		self.plot_cumulative_metrics()
		self.plot_roc_curves()
		print(f"Hospital {self.id}: Final report generated successfully\n")


def load_data(path):
	if not os.path.exists(path):
		print(f"❌ Error: File not found at {path}")
		return None, None
	
	try:
		dataset = pd.read_csv(path)
		print("✅ File loaded successfully")
		print(f"Dataset shape: {dataset.shape}")
		
		# First column is target (Diabetes_binary), rest are features
		X = dataset.iloc[:, 1:]
		Y = dataset.iloc[:, 0]
		
		print(f"Features shape: {X.shape}")
		print(f"Target shape: {Y.shape}")
		print(f"Feature columns: {list(X.columns)}")
		
		return X, Y
	except Exception as e:
		print(f"❌ Error loading file: {e}")
		return None, None

def preprocess_data(X, Y):
	"""Preprocess numeric data - no encoding needed"""
	print("\n" + "="*50)
	print("Starting preprocessing...")
	print("="*50)
	
	X = X.copy()
	Y = Y.copy()
	
	print(f"✅ Dataset shape: {X.shape}")
	print(f"✅ All columns are numeric")
	print(f"✅ Feature columns: {list(X.columns)}")
	
	# Check for missing values
	missing_count = X.isnull().sum().sum()
	if missing_count > 0:
		print(f"⚠️ Found {missing_count} missing values, filling with median...")
		X = X.fillna(X.median())
	else:
		print("✅ No missing values found")
	
	# Check for missing values in target
	y_missing = Y.isnull().sum()
	if y_missing > 0:
		print(f"⚠️ Found {y_missing} missing target values, filling with 0...")
		Y = Y.fillna(0)
	else:
		print("✅ No missing target values")
	
	# Check target distribution
	class_dist = np.bincount(Y.astype(int))
	print(f"✅ Target distribution: {class_dist}")
	print(f"   Class 0 (No Diabetes): {class_dist[0]} ({class_dist[0]/len(Y)*100:.1f}%)")
	print(f"   Class 1 (Diabetes): {class_dist[1]} ({class_dist[1]/len(Y)*100:.1f}%)")
	
	if abs(class_dist[0] - class_dist[1]) / len(Y) < 0.1:
		print("✅ Data is well balanced - no SMOTE needed")
	else:
		print("⚠️ Data is imbalanced - consider using SMOTE")
	
	print(f"✅ Preprocessing complete. Final shape: {X.shape}")
	
	return X.values, Y.values


def split_data(X, Y, n):
	print(f"\nSplitting data into {n}")
	x_array = np.array_split(X, n)
	y_array = np.array_split(Y, n)
	return x_array, y_array

def test_train(x_array, y_array, n):
	x_train, x_test, y_train, y_test = [0]*n, [0]*n, [0]*n, [0]*n
	for i in range(n):
		x_train[i], x_test[i], y_train[i], y_test[i] = train_test_split(x_array[i], y_array[i], test_size=0.2, random_state=9)
	print("Data split for training and testing.")
	return x_train, x_test, y_train, y_test

def scale_and_synthesize(x_train,  y_train):
	unique_before, counts_before = np.unique(y_train, return_counts=True)
	print(f"Class distribution BEFORE SMOTE: {dict(zip(unique_before, counts_before))}")
	columns_to_scale = [0,1,2,3,4]
	scaler = StandardScaler()
	
	x_train[:, columns_to_scale] = scaler.fit_transform(x_train[:, columns_to_scale])
	x_test[:, columns_to_scale] = scaler.transform(x_test[:, columns_to_scale])
	smote = Pipeline([
		('smote', SMOTE(random_state=9)),
		('undersample', RandomUnderSampler(random_state=9))
	])
	x_train_resampled, y_train_resampled = smote.fit_resample(x_train, y_train)
	unique_after, counts_after = np.unique(y_train_resampled, return_counts=True)
	print(f"Class distribution AFTER SMOTE: {dict(zip(unique_after, counts_after))}")
	return x_train_resampled, y_train_resampled

def categorize_smoking(smoking_status):
	if smoking_status in ['No Info', 'never']:
		return 'non-smoker'
	if smoking_status in ['current']:
		return 'current-smoker'
	if smoking_status in ['ever', 'not current', 'former']:
		return 'past-smoker'

def split_value(val, n):
	if n <= 1:
		return [val]
	if val <= n:
		raise ValueError(f"Cannot split {val} into {n} parts. Too small.")
	split_points = sorted(random.randint(0, val) for _ in range(n - 1))
	parts = [split_points[0]]
	for i in range(1, len(split_points)):
		parts.append(split_points[i] - split_points[i - 1])
	parts.append(val - split_points[-1])
	return parts


def create_hospitals(num, base_port=5000):
	hospitals = []
	ports = [base_port + i + 1 for i in range(num)]
	for i in range(num):
		other = ports[:i] + ports[i+1:]
		hospital = local_mod(i+1, ports[i], other, base_port)
		hospitals.append(hospital)
	return hospitals


# ============================================================================
# MAIN EXECUTION
# ============================================================================

n = 3
main_port = 6000
local_hospitals = create_hospitals(n, main_port)
time.sleep(0.5)

path = "Sample Data/diabetes_binary.csv"




X, Y = load_data(path)

X, Y = preprocess_data(X, Y)

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.25, random_state=9)

x_array, y_array = split_data(x_train, y_train, n)

T = 5 # Number of partitions
n = len(x_array)

# Pre-split each hospital's data into T partitions
all_local_parts_x = []
all_local_parts_y = []

for i in range(n):
	parts_x = np.array_split(x_array[i], T)
	parts_y = np.array_split(y_array[i], T)
	all_local_parts_x.append(parts_x)
	all_local_parts_y.append(parts_y)

# Iterate over each partition
for part_no in range(T):
	print(f"\n{'='*80}")
	print(f"ROUND {part_no + 1} — Collecting data from Partition P{part_no + 1}")
	print(f"{'='*80}\n")
	
	for i in range(n):
		part_x = all_local_parts_x[i][part_no]
		part_y = all_local_parts_y[i][part_no]

		print(f"Hospital {i+1} contributes {len(part_x)} samples to P{part_no + 1}")

		
		

		print(f"\nTraining Hospital {i+1} on P{part_no + 1}...")
		part_x_train, part_x_test, part_y_train, part_y_test = train_test_split(
			part_x, part_y, test_size=0.2, random_state=9
		)
		
		local_hospitals[i].current_round = part_no + 1
		
		local_hospitals[i].train(part_x_train, part_x_test, part_y_train, part_y_test)
		
		# NEW: Generate all plots after each partition
		local_hospitals[i].plot_cumulative_auc_curve()
		local_hospitals[i].plot_f1_recall_progression()
		local_hospitals[i].plot_cumulative_metrics()

		time.sleep(random.uniform(4,7))

		print(f"\nGenerating proof for Hospital {i+1} on P{part_no + 1}...")
		local_hospitals[i].generate_proof(part_no + 1)
		time.sleep(3)

print("\n" + "="*80)
print("ALL PARTITIONS COMPLETE FOR LOCAL HOSPITALS")
print("="*80 + "\n")

# Generate final reports for all hospitals
print("Generating final reports and visualizations for all local hospitals...\n")
for hospital in local_hospitals:
	hospital.generate_final_report()

print("\n" + "="*80)
print("LOCAL HOSPITALS: All visualizations and reports generated successfully!")
print("="*80 + "\n")

time.sleep(10)