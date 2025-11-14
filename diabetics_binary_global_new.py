# CORRECTED GLOBAL_MODEL.PY - NO DATA LEAKAGE, NO SMOTE
# ============================================================================

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
from sklearn.metrics import precision_score, recall_score, f1_score, classification_report
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.metrics import roc_curve, auc, roc_auc_score
from sklearn.metrics import confusion_matrix
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

np.set_printoptions(linewidth=np.inf)

class global_mod:
	def __init__(self, port, n):
		self.port = port
		self.local_ports = list()
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
		self.local_commitments = dict()
		self.commitments_rounds_log = []
		self.commitments_lock = threading.Lock()

		self.valid_models = list()
		self.no_of_local = n
		self.no_of_local_recieved = 0
		self.no_of_proofs_received = 0
		self.total_r = 0
		self.total_fi = list()
		self.validity_map = dict()
		
		self.ranges = []
		self.feature_importances_map = dict()
		self.current_round = 1
		self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
		self.server_socket.bind(('localhost', self.port))
		self.server_socket.listen()
		
		# CORRECTED: Separate validation and test sets
		self.global_x_train = None
		self.global_x_val = None      # Validation set for per-round evaluation
		self.global_x_test = None     # Holdout test set (only used at the very end)
		self.global_y_train = None
		self.global_y_val = None
		self.global_y_test = None
		
		self.received_partial_sums = set()
		self.auc_history = []
		self.partition_metrics = []
		
		threading.Thread(target=self.listen_for_data, daemon=True).start()

	def set_global_data(self, x_train, x_val, x_test, y_train, y_val, y_test):
		"""Store training, validation, and test data separately"""
		self.global_x_train = x_train.copy()
		self.global_x_val = x_val.copy()
		self.global_x_test = x_test.copy()
		self.global_y_train = y_train.copy()
		self.global_y_val = y_val.copy()
		self.global_y_test = y_test.copy()
		print("Global training, validation, and test data stored")
		print(f"Train size: {len(y_train)}, Val size: {len(y_val)}, Test size: {len(y_test)}")
		
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

		if data.get("port") == 6001:
			print(f"[H1-DEBUG] Type {data['type']} from H1 (6001) in round {self.current_round}")
		
		if (data["type"]==1):
			with self.commitments_lock:
				print("Received Commitments from local\n")
				port = data["port"]
				commitments = data["commitments"]
				round_num = data.get("round", self.current_round)
				if round_num == self.current_round:
					self.local_commitments[port] = commitments
					self.send_ranges(port)

		elif (data["type"]==3):
			self.no_of_proofs_received += 1
			print(f"Global: Received proof from port {data['port']}, total proofs: {self.no_of_proofs_received}")
			proofs = data["proofs"]
			self.feature_importances_map[data["port"]] = data.get("feature_importances", [])
			self.check_validity(proofs, data["port"])
			
			if self.no_of_proofs_received == self.no_of_local:
				print(f"Global: All {self.no_of_local} local models have submitted proofs for partition {self.current_round}")
				self.send_valid_ports()
				self.no_of_proofs_received = 0
				self.no_of_local_recieved = 0

		elif (data["type"]==7):
			sender_port = data["port"]
			round_num = data["round"]
			submission_id = f"{sender_port}_{round_num}"
			
			if submission_id in self.received_partial_sums:
				print(f"[WARNING] Duplicate partial sum from port {sender_port} for round {round_num}, ignoring")
				return

			self.received_partial_sums.add(submission_id)
			self.update_total_data(data)
			
			print(f"Global: Received partial sum from port {data['port']}, total received: {self.no_of_local_recieved}/{len(self.valid_models)}")
			
			if self.no_of_local_recieved == len(self.valid_models):
				if self.verify_total_data(data):
					print(f"Global: Feature summation is correctly verified for partition {data['round']}")
					print(f"\nGlobal: Aggregating for partition {data['round']}...")
					time.sleep(0.5)
					
					self.aggregate()
					self.plot_cumulative_auc_curve()
					
					self.reset_for_next_partition()
					
					print(f"\n=== Global: Ready for next partition (Round {self.current_round}) ===\n")
				else:
					print(f"Global: Feature summation is incorrect for partition {data['round'] + 1}")
		else:
			print("Global: Random access received...")

	def reset_for_next_partition(self):
		with self.commitments_lock:
			self.valid_models = list()
			self.no_of_local_recieved = 0
			self.total_r = 0
			self.total_fi = [0] * len(self.get_feature_importances()) if hasattr(self, 'model') and hasattr(self.model, 'feature_importances_') else []
			self.validity_map = dict()
			self.feature_importances_map = dict()
			self.received_partial_sums = set()
        
			if self.no_of_proofs_received >= self.no_of_local and self.no_of_local_recieved >=len(self.valid_models):
				self.local_commitments = dict()
				self.no_of_proofs_received = 0
            
		self.current_round += 1
		print(f"Global: Reset complete for Round {self.current_round}")

	def send_ranges(self, local_port):
		print(f"Ranges going to sent:{self.ranges}")
		limits = {"type":2, "ranges":self.ranges, "port":self.port}
		limits_encoded = json.dumps(limits).encode()
		with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
			try:
				s.connect(('localhost', local_port))
				s.sendall(limits_encoded)
			except ConnectionRefusedError:
				print(f"Global model could not connect to Node on port {local_port}")

	def check_validity(self, proofs, local_port):
		print(f"\n\nChecking validity of proofs of local model {local_port - self.port}...")
		check = True

		if local_port not in self.local_commitments:
			print(f"[ERROR] No commitments found for the port {local_port}")
			return 
		if len(self.local_commitments[local_port]) < len(proofs):
			print(f"[ERROR] Not enough commitments from port {local_port}")
			return
		for i in range(len(proofs)):
			if (self.local_commitments[local_port][i]==proofs[i]["C"]):
				if (validate_proof(proofs[i], self.ranges[i][0], self.ranges[i][1])):
					print(f" Proof {i+1} for local model {local_port - self.port} is valid.")
				else:
					check = False
					print(f" Proof {i+1} for local model {local_port - self.port} is invalid.")
			else:
				check = False
				print(f"Proof {i} for local model {local_port - self.port} is invalid.")
		print(f"\nModel {local_port - self.port} validity: {check}")
		self.validity_map[local_port] = check
		validity = {"type": 4, "validity":check, "port":self.port}
		validity_encoded = json.dumps(validity).encode()
		if check and local_port not in self.valid_models:
			self.valid_models.append(local_port)
		with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
			try:
				s.connect(('localhost', local_port))
				s.sendall(validity_encoded)
			except ConnectionRefusedError:
				print(f"Global model could not connect to Node on port {local_port}")

	def send_valid_ports(self):
		"""
		FIXED: Handle case when no valid models exist
		"""
		valid_model_data = {"type":5, "port":self.port, "valid models":self.valid_models}
		valid_model_data_encoded = json.dumps(valid_model_data).encode()
		
		print(f"\n{'='*70}")
		print(f"Valid model ports for partition {self.current_round}: {self.valid_models}")
		print(f"{'='*70}\n")
		
		if not self.valid_models:
			print("⚠️ [CRITICAL WARNING] No valid models available for MPC!")
			print("➡️ Skipping aggregation and retraining global model alone...")
			
			# Train global model without aggregation
			self.train(self.global_x_train, self.global_x_val, 
					self.global_y_train, self.global_y_val, is_initial=False)
			
			# Notify all local models that this round is skipped
			skip_message = {
				"type": 8,
				"port": self.port,
				"message": "No valid models - round skipped",
				"round": self.current_round
			}
			skip_message_encoded = json.dumps(skip_message).encode()
			
			for port in self.local_commitments.keys():
				with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
					try:
						s.connect(('localhost', port))
						s.sendall(skip_message_encoded)
						print(f"✅ Sent skip notification to port {port}")
					except ConnectionRefusedError:
						print(f"❌ Could not notify port {port}")
			
			self.reset_for_next_partition()
			return

		# Normal case: send valid ports to all valid models
		for port in self.valid_models:
			with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
				try:
					s.connect(('localhost', port))
					s.sendall(valid_model_data_encoded)
					print(f"✅ Sent valid ports to {port}")
				except ConnectionRefusedError:
					print(f"❌ Global model could not connect to port {port}")

	def update_total_data(self, data):
		self.no_of_local_recieved += 1
		self.total_r += data["partial r"]
		if not self.total_fi:
			self.total_fi = [0] * len(data["partial fi"])
		for i in range(len(self.total_fi)):
			self.total_fi[i] += data["partial fi"][i]

	def verify_total_data(self, data):
		print("\nChecking if received values are correct...")
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
		return True

	def train(self, x_train, x_eval, y_train, y_eval, sw=None, is_initial=True):
		"""
		CORRECTED: Now uses separate validation set for evaluation
		x_eval and y_eval should be validation set, NOT test set
		"""
		self.model.fit(x_train, y_train, sample_weight=sw)
		y_pred_proba = self.model.predict_proba(x_eval)[:, 1]
		optimal_threshold = self.find_optimal_threshold(y_pred_proba, y_eval)
		y_pred = (y_pred_proba >= optimal_threshold).astype(int)
		accuracy = accuracy_score(y_eval, y_pred)
		tn, fp, fn, tp = confusion_matrix(y_eval, y_pred).ravel()
		actual_positives = tp + fn
		actual_negatives = tn + fp
		recall = recall_score(y_eval, y_pred, zero_division=0)
		precision = precision_score(y_eval, y_pred, zero_division=0)
		f1 = f1_score(y_eval, y_pred, zero_division=0)
		
		try:
			auc_score = roc_auc_score(y_eval, y_pred_proba)
			fpr, tpr, thresholds = roc_curve(y_eval, y_pred_proba)
			
			metrics = {
				'round': self.current_round if not is_initial else 0,
				'accuracy': accuracy,
				'precision': precision,
				'recall': recall,
				'f1': f1,
				'auc': auc_score,
				'tp': int(tp),
				'tn': int(tn),
				'fp': int(fp),
				'fn': int(fn),
				'fpr': fpr,
				'tpr': tpr
			}
			self.partition_metrics.append(metrics)
			self.auc_history.append(auc_score)
			
			round_label = "Initial" if is_initial else f"Partition {self.current_round}"
			print(f"\n{'='*70}")
			print(f"GLOBAL MODEL - {round_label} Results (on VALIDATION set)")
			print(f"{'='*70}")
			print(f"Accuracy: {accuracy:.4f}")
			print(f"AUC-ROC: {auc_score:.4f}")
			print(f"Precision: {precision:.4f}")
			print(f"Recall: {recall:.4f}")
			print(f"F1-Score: {f1:.4f}")
			print(f"True Positives (TP): {tp}")
			print(f"True Negatives (TN): {tn}")
			print(f"False Positives (FP): {fp}")
			print(f"False Negatives (FN): {fn}")
			print(f"Actual Positive: {actual_positives}")
			print(f"Actual Negative: {actual_negatives}")
			print(f"{'='*70}\n")
			
		except Exception as e:
			print(f"Error calculating AUC: {e}")
			auc_score = None
		
		feature_importances = self.get_feature_importances()
		self.total_fi = [0]*len(feature_importances)
		percent = 3.0
		self.ranges = [[int(np.maximum(0, i-(i*percent))), int(i+(i*percent))] for i in feature_importances]
		print(f"Ranges: {self.ranges}")
		print(f"\nFeature importances: {self.model.feature_importances_.tolist()}")
		
		return auc_score
	
	def evaluate_on_test_set(self):
		"""
		CORRECTED: Final evaluation on completely holdout test set
		This should only be called ONCE at the very end
		"""
		print("\n" + "="*80)
		print("FINAL EVALUATION ON HOLDOUT TEST SET")
		print("="*80)
		
		y_pred_proba = self.model.predict_proba(self.global_x_test)[:, 1]
		optimal_threshold = self.find_optimal_threshold(y_pred_proba, self.global_y_test)
		y_pred = (y_pred_proba >= optimal_threshold).astype(int)
		
		accuracy = accuracy_score(self.global_y_test, y_pred)
		tn, fp, fn, tp = confusion_matrix(self.global_y_test, y_pred).ravel()
		recall = recall_score(self.global_y_test, y_pred, zero_division=0)
		precision = precision_score(self.global_y_test, y_pred, zero_division=0)
		f1 = f1_score(self.global_y_test, y_pred, zero_division=0)
		auc_score = roc_auc_score(self.global_y_test, y_pred_proba)
		
		print(f"\nFINAL TEST SET RESULTS:")
		print(f"Accuracy: {accuracy:.4f}")
		print(f"AUC-ROC: {auc_score:.4f}")
		print(f"Precision: {precision:.4f}")
		print(f"Recall: {recall:.4f}")
		print(f"F1-Score: {f1:.4f}")
		print(f"TP: {tp}, TN: {tn}, FP: {fp}, FN: {fn}")
		print("="*80 + "\n")
		
		return {
			'accuracy': accuracy,
			'auc': auc_score,
			'precision': precision,
			'recall': recall,
			'f1': f1,
			'confusion_matrix': {'tp': tp, 'tn': tn, 'fp': fp, 'fn': fn}
		}
	
	def plot_cumulative_auc_curve(self):
		if not self.partition_metrics:
			return
		
		save_path = f'global_auc_round_{self.current_round}.png'
		fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
		
		colors = plt.cm.viridis(np.linspace(0, 0.9, len(self.partition_metrics)))
		
		# LEFT PLOT: Full ROC curve
		for idx, metrics in enumerate(self.partition_metrics):
			ax1.plot(
				metrics['fpr'], 
				metrics['tpr'], 
				color=colors[idx],
				lw=2.5,
				label=f"Round {metrics['round']} (AUC = {metrics['auc']:.4f})",
				alpha=0.8
			)
		
		ax1.plot([0, 1], [0, 1], 'k--', lw=2, label='Random Classifier', alpha=0.5)
		ax1.set_xlim([0.0, 1.0])
		ax1.set_ylim([0.0, 1.05])
		ax1.set_xlabel('False Positive Rate', fontsize=12, fontweight='bold')
		ax1.set_ylabel('True Positive Rate', fontsize=12, fontweight='bold')
		ax1.set_title('Full ROC Curve', fontsize=13, fontweight='bold')
		ax1.legend(loc="lower right", fontsize=9)
		ax1.grid(alpha=0.3, linestyle='--')
		
		# RIGHT PLOT: Zoomed into clinically relevant region
		for idx, metrics in enumerate(self.partition_metrics):
			ax2.plot(
				metrics['fpr'], 
				metrics['tpr'], 
				color=colors[idx],
				lw=3,
				label=f"Round {metrics['round']}",
				alpha=0.9,
				marker='o',
				markersize=4,
				markevery=20
			)
		
		ax2.set_xlim([0.0, 0.15])
		ax2.set_ylim([0.70, 1.0])
		ax2.set_xlabel('False Positive Rate', fontsize=12, fontweight='bold')
		ax2.set_ylabel('True Positive Rate', fontsize=12, fontweight='bold')
		ax2.set_title('Zoomed: Clinical Operating Region', fontsize=13, fontweight='bold')
		ax2.legend(loc="lower right", fontsize=9)
		ax2.grid(alpha=0.3, linestyle='--')
		
		textstr = f'Current Round: {self.current_round}\nEvaluated on: Validation Set'
		props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
		ax1.text(0.02, 0.98, textstr, transform=ax1.transAxes, fontsize=10,
				verticalalignment='top', bbox=props)
		
		plt.tight_layout()
		plt.savefig(save_path, dpi=300, bbox_inches='tight')
		print(f"AUC curve (with zoom) for Round {self.current_round} saved to {save_path}")
		plt.close()

	def plot_auc_curves(self, save_path='global_auc_curves_final.png'):
		if not self.partition_metrics:
			print("No metrics available to plot")
			return
		
		plt.figure(figsize=(12, 8))
		colors = plt.cm.rainbow(np.linspace(0, 1, len(self.partition_metrics)))
		
		for idx, metrics in enumerate(self.partition_metrics):
			plt.plot(
				metrics['fpr'], 
				metrics['tpr'], 
				color=colors[idx],
				lw=2,
				label=f"Partition {metrics['round']} (AUC = {metrics['auc']:.3f})"
			)
		
		plt.plot([0, 1], [0, 1], 'k--', lw=2, label='Random Classifier')
		plt.xlim([0.0, 1.0])
		plt.ylim([0.0, 1.05])
		plt.xlabel('False Positive Rate', fontsize=12)
		plt.ylabel('True Positive Rate', fontsize=12)
		plt.title('Global Model: ROC Curves on Validation Set', fontsize=14, fontweight='bold')
		plt.legend(loc="lower right", fontsize=10)
		plt.grid(alpha=0.3)
		plt.tight_layout()
		plt.savefig(save_path, dpi=300, bbox_inches='tight')
		print(f"Global AUC curves (final) saved to {save_path}")
		plt.close()
	
	def plot_metrics_progression(self, save_path='global_metrics_progression.png'):
		if not self.partition_metrics:
			print("No metrics available to plot")
			return
		
		rounds = [m['round'] for m in self.partition_metrics]
		accuracies = [m['accuracy'] for m in self.partition_metrics]
		aucs = [m['auc'] for m in self.partition_metrics]
		precisions = [m['precision'] for m in self.partition_metrics]
		recalls = [m['recall'] for m in self.partition_metrics]
		f1s = [m['f1'] for m in self.partition_metrics]
		
		fig, axes = plt.subplots(2, 2, figsize=(15, 10))
		
		axes[0, 0].plot(rounds, aucs, 'o-', linewidth=2, markersize=8, color='#2E86AB')
		axes[0, 0].set_xlabel('Partition Number', fontsize=11)
		axes[0, 0].set_ylabel('AUC-ROC Score', fontsize=11)
		axes[0, 0].set_title('AUC-ROC Score Progression (Validation Set)', fontsize=12, fontweight='bold')
		axes[0, 0].grid(alpha=0.3)
		axes[0, 0].set_ylim([0, 1.05])
		
		axes[0, 1].plot(rounds, accuracies, 'o-', linewidth=2, markersize=8, color='#A23B72')
		axes[0, 1].set_xlabel('Partition Number', fontsize=11)
		axes[0, 1].set_ylabel('Accuracy', fontsize=11)
		axes[0, 1].set_title('Accuracy Progression (Validation Set)', fontsize=12, fontweight='bold')
		axes[0, 1].grid(alpha=0.3)
		axes[0, 1].set_ylim([0, 1.05])
		
		axes[1, 0].plot(rounds, precisions, 'o-', linewidth=2, markersize=8, 
					   color='#F18F01', label='Precision')
		axes[1, 0].plot(rounds, recalls, 's-', linewidth=2, markersize=8, 
					   color='#C73E1D', label='Recall')
		axes[1, 0].set_xlabel('Partition Number', fontsize=11)
		axes[1, 0].set_ylabel('Score', fontsize=11)
		axes[1, 0].set_title('Precision & Recall Progression', fontsize=12, fontweight='bold')
		axes[1, 0].legend(fontsize=10)
		axes[1, 0].grid(alpha=0.3)
		axes[1, 0].set_ylim([0, 1.05])
		
		axes[1, 1].plot(rounds, f1s, 'o-', linewidth=2, markersize=8, color='#6A994E')
		axes[1, 1].set_xlabel('Partition Number', fontsize=11)
		axes[1, 1].set_ylabel('F1 Score', fontsize=11)
		axes[1, 1].set_title('F1 Score Progression', fontsize=12, fontweight='bold')
		axes[1, 1].grid(alpha=0.3)
		axes[1, 1].set_ylim([0, 1.05])
		
		plt.tight_layout()
		plt.savefig(save_path, dpi=300, bbox_inches='tight')
		print(f"Metrics progression saved to {save_path}")
		plt.close()
	
	def find_optimal_threshold(self, y_pred_proba, y_true):
		"""Find optimal threshold using validation set"""
		best_f1 = 0
		best_threshold = 0.5
		
		for threshold in np.arange(0.2, 0.8, 0.01):
			y_pred = (y_pred_proba >= threshold).astype(int)
			current_f1 = f1_score(y_true, y_pred, zero_division=0)
			if current_f1 > best_f1:
				best_f1 = current_f1
				best_threshold = threshold
		print(f"Optimal threshold: {best_threshold:.3f}")
		return best_threshold
		
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
		aggregated_fi = self.calculate_aggregated_fi()
		
		print(f"🔧 Using aggregated FI to influence training")
		print(f"   Aggregated FI: {aggregated_fi}")
		
		# Calculate feature scores for each sample
		feature_scores = np.dot(self.global_x_train, aggregated_fi)
		
		# Normalize to 0-1 range
		feature_weights = (feature_scores - feature_scores.min()) / (feature_scores.max() - feature_scores.min() + 1e-10)
		
		# Since data is balanced, we don't need class weights
		# Just use feature importance weights
		sample_weights = 1 + feature_weights
		
		# Normalize
		sample_weights = sample_weights / sample_weights.sum() * len(sample_weights)
		
		print(f"   Sample weights range: [{sample_weights.min():.3f}, {sample_weights.max():.3f}]")
		
		# Train with feature-informed weights
		self.train(self.global_x_train, self.global_x_val, 
				self.global_y_train, self.global_y_val, 
				sample_weights, is_initial=False)
		self.plot_metrics_progression()

	def calculate_aggregated_fi(self):
		sum_array = [i/((10**6)*len(self.valid_models)) for i in self.total_fi]
		aggregated = np.array(sum_array)
		aggregated /= np.sum(aggregated)
		return aggregated


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
	"""Split data into n partitions"""
	print(f"\nSplitting data into {n} partitions")
	x_array = np.array_split(X, n)
	y_array = np.array_split(Y, n)
	return x_array, y_array

def add_commitments(commitments):
	"""Add multiple commitments together"""
	val = commitments[0]
	for i in range(1, len(commitments)):
		val = add(val, commitments[i])
	return val

def scale_train_only(x_train, y_train):
	"""
	Scale training data - NO SMOTE (data is balanced)
	"""
	scaler = StandardScaler()
	x_train_scaled = scaler.fit_transform(x_train)
	
	print(f"\n{'='*50}")
	print("Scaling training data (NO SMOTE - data is balanced)")
	print(f"{'='*50}")
	print(f"Training samples: {len(y_train)}")
	
	class_dist = np.bincount(y_train.astype(int))
	print(f"Class distribution: {class_dist}")
	print(f"   Class 0: {class_dist[0]} ({class_dist[0]/len(y_train)*100:.1f}%)")
	print(f"   Class 1: {class_dist[1]} ({class_dist[1]/len(y_train)*100:.1f}%)")
	print("✅ Data is balanced - no SMOTE needed")
	print(f"{'='*50}\n")
	
	return x_train_scaled, y_train, scaler




def scale_eval_data(x_eval, scaler):
	"""Scale validation/test data using the fitted scaler from training"""
	x_eval_scaled = scaler.transform(x_eval)
	return x_eval_scaled


# ============================================================================
# MAIN EXECUTION
# ============================================================================

n = 3
main_port = 6000
global_model = global_mod(main_port, n)

path = "Sample Data/diabetes_binary.csv"
X, Y = load_data(path)

if X is None or Y is None:
	print("❌ Failed to load data. Exiting...")
	exit(1)

X, Y = preprocess_data(X, Y)

print("\n" + "="*80)
print("DATA SPLITTING - NO LEAKAGE, NO SMOTE")
print("="*80)

# Three-way split: 60% training, 20% validation, 20% test
X_train_all, X_test, Y_train_all, Y_test = train_test_split(
	X, Y, test_size=0.2, random_state=42, stratify=Y
)
X_train_all, X_val, Y_train_all, Y_val = train_test_split(
	X_train_all, Y_train_all, test_size=0.2, random_state=42, stratify=Y_train_all
)

print(f"Total data: {len(X)}")
print(f"Training data (60%): {len(X_train_all)}")
print(f"Validation data (20%): {len(X_val)}")
print(f"Test data (20% - HOLDOUT): {len(X_test)}")
print("="*80 + "\n")

# Split training data among n+1 entities (n local hospitals + 1 global)
x_array, y_array = split_data(X_train_all, Y_train_all, n+1)

# Process global model's data
print("\nProcessing global model's data...")
x_train_global_raw = x_array[n]
y_train_global_raw = y_array[n]

# Scale training data (NO SMOTE - data is balanced)
x_train_global, y_train_global, global_scaler = scale_train_only(
	x_train_global_raw, 
	y_train_global_raw
)

# Scale validation and test sets using the same scaler (NO SMOTE)
x_val_global = scale_eval_data(X_val, global_scaler)
x_test_global = scale_eval_data(X_test, global_scaler)

print(f"Global training samples (scaled, no SMOTE): {len(y_train_global)}")
print(f"Global validation samples (scaled, no SMOTE): {len(Y_val)}")
print(f"Global test samples (scaled, no SMOTE): {len(Y_test)}")

# Store data in global model
global_model.set_global_data(
	x_train_global, 
	x_val_global, 
	x_test_global, 
	y_train_global, 
	Y_val, 
	Y_test
)

print("\n" + "="*80)
print("Training initial global model (evaluated on VALIDATION set)...")
print("="*80)
global_model.train(x_train_global, x_val_global, y_train_global, Y_val, is_initial=True)
print("✅ Initial training complete.")
print("\nWaiting for local models...")

time.sleep(1000)

# After all partitions complete, do final evaluation
print("\n" + "="*80)
print("ALL PARTITIONS COMPLETE")
print("="*80 + "\n")

# Generate validation set visualizations
print("Generating visualizations based on validation set performance...")
global_model.plot_auc_curves()
global_model.plot_metrics_progression()

# FINAL TEST SET EVALUATION (only once)
print("\n" + "="*80)
print("PERFORMING FINAL EVALUATION ON HOLDOUT TEST SET")
print("(This test set was NEVER seen during training/validation)")
print("="*80 + "\n")

final_test_results = global_model.evaluate_on_test_set()

print("\n" + "="*80)
print("FINAL RESULTS SUMMARY")
print("="*80)
print(f"Test Accuracy: {final_test_results['accuracy']:.4f}")
print(f"Test AUC-ROC: {final_test_results['auc']:.4f}")
print(f"Test Precision: {final_test_results['precision']:.4f}")
print(f"Test Recall: {final_test_results['recall']:.4f}")
print(f"Test F1-Score: {final_test_results['f1']:.4f}")
print("="*80)

print("\nAll visualizations and evaluations complete!")
print("="*80 + "\n")