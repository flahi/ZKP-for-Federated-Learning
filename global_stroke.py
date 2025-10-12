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
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import classification_report, roc_curve, auc, roc_auc_score
from imblearn.pipeline import Pipeline
from imblearn.under_sampling import RandomUnderSampler  
from imblearn.over_sampling import SMOTE
from sklearn.metrics import confusion_matrix
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, LabelEncoder
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

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
		
		# Separate validation and test sets
		self.global_x_train = None
		self.global_x_val = None      
		self.global_x_test = None     
		self.global_y_train = None
		self.global_y_val = None
		self.global_y_test = None
		
		self.received_partial_sums = set()
		self.auc_history = []
		self.partition_metrics = []
		
		# NEW: Cumulative metrics tracking
		self.cumulative_f1 = []
		self.cumulative_recall = []
		self.cumulative_precision = []
		self.cumulative_accuracy = []
		
		threading.Thread(target=self.listen_for_data, daemon=True).start()

	def set_global_data(self, x_train, x_val, x_test, y_train, y_val, y_test):
		"""Store training, validation, and test data separately"""
		self.global_x_train = x_train.copy()
		self.global_x_val = x_val.copy()
		self.global_x_test = x_test.copy()
		self.global_y_train = y_train.copy()
		self.global_y_val = y_val.copy()
		self.global_y_test = y_test.copy()
		print("✅ Global training, validation, and test data stored")
		print(f"   Train: {len(y_train)}, Val: {len(y_val)}, Test: {len(y_test)}")
		
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
			print(f"🔍 [H1-DEBUG] Type {data['type']} from H1 (6001) in round {self.current_round}")
		
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
			print(f"✅ Global: Received proof from port {data['port']}, total proofs: {self.no_of_proofs_received}")
			proofs = data["proofs"]
			self.feature_importances_map[data["port"]] = data.get("feature_importances", [])
			self.check_validity(proofs, data["port"])
			
			if self.no_of_proofs_received == self.no_of_local:
				print(f"🎉 Global: All {self.no_of_local} local models submitted proofs for partition {self.current_round}")
				self.send_valid_ports()
				self.no_of_proofs_received = 0
				self.no_of_local_recieved = 0

		elif (data["type"]==7):
			sender_port = data["port"]
			round_num = data["round"]
			submission_id = f"{sender_port}_{round_num}"
			
			if submission_id in self.received_partial_sums:
				print(f"[WARNING] Duplicate partial sum from port {sender_port} for round {round_num}")
				return

			self.received_partial_sums.add(submission_id)
			self.update_total_data(data)
			
			print(f"📊 Global: Received partial sum from port {data['port']}, total: {self.no_of_local_recieved}/{len(self.valid_models)}")
			
			if self.no_of_local_recieved == len(self.valid_models):
				if self.verify_total_data(data):
					print(f"✅ Global: Feature summation verified for partition {data['round']}")
					print(f"\n🔄 Global: Aggregating for partition {data['round']}...")
					time.sleep(0.5)
					
					self.aggregate()
					self.plot_cumulative_auc_curve()
					self.plot_f1_recall_progression()  # NEW
					self.plot_cumulative_metrics()      # NEW
					self.reset_for_next_partition()
					
					print(f"\n=== Global: Ready for next partition (Round {self.current_round}) ===\n")
				else:
					print(f"❌ Global: Feature summation incorrect for partition {data['round'] + 1}")
		elif (data["type"]==8):
			print(f"⚠️ Hospital {self.id}: Received skip notification for round {data['round']}")
			print(f"   Reason: {data['message']}")
		else:
			print("🔍 Global: Random access received...")

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
		print(f"✅ Global: Reset complete for Round {self.current_round}")

	def send_ranges(self, local_port):
		print(f"Sending ranges: {self.ranges}")
		limits = {"type":2, "ranges":self.ranges, "port":self.port}
		limits_encoded = json.dumps(limits).encode()
		with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
			try:
				s.connect(('localhost', local_port))
				s.sendall(limits_encoded)
			except ConnectionRefusedError:
				print(f"Global model could not connect to Node on port {local_port}")

	def check_validity(self, proofs, local_port):
		print(f"\nChecking validity of proofs from local model {local_port - self.port}...")
		check = True
		
		total_verification_time = 0
		individual_times = []
		
		if local_port not in self.local_commitments:
			print(f"[ERROR] No commitments found for port {local_port}")
			return 
		if len(self.local_commitments[local_port]) < len(proofs):
			print(f"[ERROR] Not enough commitments from port {local_port}")
			return
			
		for i in range(len(proofs)):
			if (self.local_commitments[local_port][i]==proofs[i]["C"]):
				start_time = time.time()
				is_valid = validate_proof(proofs[i], self.ranges[i][0], self.ranges[i][1])
				end_time = time.time()
				verification_time = (end_time - start_time) * 1000
				
				total_verification_time += verification_time
				individual_times.append(verification_time)
				
				if is_valid:
					print(f" ✔ Proof {i+1} valid ({verification_time:.2f} ms)")
				else:
					check = False
					print(f" ❌ Proof {i+1} invalid ({verification_time:.2f} ms)")
			else:
				check = False
				print(f"❌ Proof {i} invalid - commitment mismatch")
		
		print(f"\n📊 VERIFICATION TIME SUMMARY for Hospital {local_port - self.port}:")
		print(f"   Total verification time: {total_verification_time:.2f} ms")
		print(f"   Average per proof: {total_verification_time/len(proofs):.2f} ms")
		print(f"   Individual times: {[f'{t:.2f}ms' for t in individual_times]}")
		print(f"   Model validity: {check}")
		
		if not hasattr(self, 'verification_times'):
			self.verification_times = []
		
		self.verification_times.append({
			'hospital_id': local_port - self.port,
			'round': self.current_round,
			'total_time_ms': total_verification_time,
			'avg_time_ms': total_verification_time/len(proofs),
			'individual_times': individual_times,
			'proofs_count': len(proofs),
			'is_valid': check
		})
		
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
		valid_model_data = {"type":5, "port":self.port, "valid models":self.valid_models}
		valid_model_data_encoded = json.dumps(valid_model_data).encode()
		
		print(f"\n{'='*70}")
		print(f"Valid model ports for partition {self.current_round}: {self.valid_models}")
		print(f"{'='*70}\n")
		
		if not self.valid_models:
			print("⚠️ [CRITICAL WARNING] No valid models available for MPC!")
			print("➡️ Skipping aggregation and retraining global model alone...")
			
			self.train(self.global_x_train, self.global_x_val, 
					self.global_y_train, self.global_y_val, is_initial=False)
			
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
		print("\nVerifying received values...")
		valid_commitments = self.get_valid_commitments()
		time.sleep(0.5)
		for i in range(len(valid_commitments)):
			C_total = add_commitments(valid_commitments[i])
			C_calculated = pedersen_commit(self.total_fi[i], self.total_r, G, H)
			if (C_total!=C_calculated):
				return False
		return True

	def train(self, x_train, x_eval, y_train, y_eval, sw=None, is_initial=True):
		"""Train model and evaluate on validation set"""
		self.model.fit(x_train, y_train, sample_weight=sw)
		y_pred_proba = self.model.predict_proba(x_eval)[:, 1]
		threshold = 0.75
		y_pred = (y_pred_proba >= threshold).astype(int)
		
		accuracy = accuracy_score(y_eval, y_pred)
		tn, fp, fn, tp = confusion_matrix(y_eval, y_pred).ravel()
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
			
			# NEW: Track cumulative metrics
			if not is_initial:
				self.cumulative_f1.append(np.mean([m['f1'] for m in self.partition_metrics]))
				self.cumulative_recall.append(np.mean([m['recall'] for m in self.partition_metrics]))
				self.cumulative_precision.append(np.mean([m['precision'] for m in self.partition_metrics]))
				self.cumulative_accuracy.append(np.mean([m['accuracy'] for m in self.partition_metrics]))
			
			round_label = "Initial" if is_initial else f"Partition {self.current_round}"
			print(f"\n{'='*70}")
			print(f"GLOBAL MODEL - {round_label} Results (VALIDATION SET)")
			print(f"{'='*70}")
			print(f"Accuracy: {accuracy:.4f}")
			print(f"AUC-ROC: {auc_score:.4f}")
			print(f"Precision: {precision:.4f}")
			print(f"Recall: {recall:.4f}")
			print(f"F1-Score: {f1:.4f}")
			print(f"TP: {tp}, TN: {tn}, FP: {fp}, FN: {fn}")
			
			# NEW: Print cumulative metrics
			if not is_initial and len(self.cumulative_f1) > 0:
				print(f"\n📊 CUMULATIVE METRICS (Average up to Round {self.current_round}):")
				print(f"   Cumulative F1: {self.cumulative_f1[-1]:.4f}")
				print(f"   Cumulative Recall: {self.cumulative_recall[-1]:.4f}")
				print(f"   Cumulative Precision: {self.cumulative_precision[-1]:.4f}")
				print(f"   Cumulative Accuracy: {self.cumulative_accuracy[-1]:.4f}")
			print(f"{'='*70}\n")
			
		except Exception as e:
			print(f"Error calculating AUC: {e}")
			auc_score = None
		
		feature_importances = self.get_feature_importances()
		self.total_fi = [0]*len(feature_importances)
		percent = 2.0
		self.ranges = [[int(np.maximum(0, i-(i*percent))), int(i+(i*percent))] for i in feature_importances]
		
		return auc_score
	
	def evaluate_on_test_set(self):
		"""Final evaluation on holdout test set"""
		print("\n" + "="*80)
		print("FINAL EVALUATION ON HOLDOUT TEST SET")
		print("="*80)
		
		y_pred_proba = self.model.predict_proba(self.global_x_test)[:, 1]
		threshold = 0.75
		y_pred = (y_pred_proba >= threshold).astype(int)
		
		accuracy = accuracy_score(self.global_y_test, y_pred)
		tn, fp, fn, tp = confusion_matrix(self.global_y_test, y_pred).ravel()
		recall = recall_score(self.global_y_test, y_pred, zero_division=0)
		precision = precision_score(self.global_y_test, y_pred, zero_division=0)
		f1 = f1_score(self.global_y_test, y_pred, zero_division=0)
		auc_score = roc_auc_score(self.global_y_test, y_pred_proba)
		
		print(f"\nFINAL TEST RESULTS:")
		print(f"Accuracy: {accuracy:.4f}")
		print(f"AUC-ROC: {auc_score:.4f}")
		print(f"Precision: {precision:.4f}")
		print(f"Recall: {recall:.4f}")
		print(f"F1-Score: {f1:.4f}")
		print(f"TP: {tp}, TN: {tn}, FP: {fp}, FN: {fn}")
		print("="*80 + "\n")
		
		return {'accuracy': accuracy, 'auc': auc_score, 'precision': precision, 
		        'recall': recall, 'f1': f1}
	
	def plot_cumulative_auc_curve(self):
		"""Plot ROC curves after each round"""
		if not self.partition_metrics:
			return
		
		save_path = f'global_auc_round_{self.current_round}.png'
		fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
		
		colors = plt.cm.viridis(np.linspace(0, 0.9, len(self.partition_metrics)))
		
		for idx, metrics in enumerate(self.partition_metrics):
			ax1.plot(metrics['fpr'], metrics['tpr'], color=colors[idx], lw=2.5,
			        label=f"Round {metrics['round']} (AUC={metrics['auc']:.4f})", alpha=0.8)
		
		ax1.plot([0, 1], [0, 1], 'k--', lw=2, label='Random', alpha=0.5)
		ax1.set_xlim([0.0, 1.0])
		ax1.set_ylim([0.0, 1.05])
		ax1.set_xlabel('False Positive Rate', fontsize=12, fontweight='bold')
		ax1.set_ylabel('True Positive Rate', fontsize=12, fontweight='bold')
		ax1.set_title('Full ROC Curve', fontsize=13, fontweight='bold')
		ax1.legend(loc="lower right", fontsize=9)
		ax1.grid(alpha=0.3)
		
		for idx, metrics in enumerate(self.partition_metrics):
			ax2.plot(metrics['fpr'], metrics['tpr'], color=colors[idx], lw=3,
			        label=f"Round {metrics['round']}", alpha=0.9, marker='o', markersize=4, markevery=20)
		
		ax2.set_xlim([0.0, 0.15])
		ax2.set_ylim([0.70, 1.0])
		ax2.set_xlabel('False Positive Rate', fontsize=12, fontweight='bold')
		ax2.set_ylabel('True Positive Rate', fontsize=12, fontweight='bold')
		ax2.set_title('Zoomed: Clinical Region', fontsize=13, fontweight='bold')
		ax2.legend(loc="lower right", fontsize=9)
		ax2.grid(alpha=0.3)
		
		plt.tight_layout()
		plt.savefig(save_path, dpi=300, bbox_inches='tight')
		print(f"✅ AUC curve saved to {save_path}")
		plt.close()

	# NEW: F1 and Recall progression plot
	def plot_f1_recall_progression(self):
		"""Plot F1 and Recall scores after each round"""
		if len(self.partition_metrics) < 2:
			return
		
		save_path = f'global_f1_recall_round_{self.current_round}.png'
		
		rounds = [m['round'] for m in self.partition_metrics if m['round'] > 0]
		f1_scores = [m['f1'] for m in self.partition_metrics if m['round'] > 0]
		recall_scores = [m['recall'] for m in self.partition_metrics if m['round'] > 0]
		
		fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
		
		# F1 Score
		ax1.plot(rounds, f1_scores, 'o-', linewidth=2.5, markersize=9, 
		        color='#FF6B6B', label='F1 Score')
		ax1.set_xlabel('Partition Number', fontsize=12, fontweight='bold')
		ax1.set_ylabel('F1 Score', fontsize=12, fontweight='bold')
		ax1.set_title('Global Model: F1 Score Progression', fontsize=13, fontweight='bold')
		ax1.grid(alpha=0.3, linestyle='--')
		ax1.set_ylim([0, 1.05])
		ax1.legend(fontsize=11)
		
		# Add value labels on points
		for i, (x, y) in enumerate(zip(rounds, f1_scores)):
			ax1.text(x, y + 0.02, f'{y:.3f}', ha='center', fontsize=9)
		
		# Recall Score
		ax2.plot(rounds, recall_scores, 's-', linewidth=2.5, markersize=9, 
		        color='#4ECDC4', label='Recall')
		ax2.set_xlabel('Partition Number', fontsize=12, fontweight='bold')
		ax2.set_ylabel('Recall Score', fontsize=12, fontweight='bold')
		ax2.set_title('Global Model: Recall Progression', fontsize=13, fontweight='bold')
		ax2.grid(alpha=0.3, linestyle='--')
		ax2.set_ylim([0, 1.05])
		ax2.legend(fontsize=11)
		
		# Add value labels on points
		for i, (x, y) in enumerate(zip(rounds, recall_scores)):
			ax2.text(x, y + 0.02, f'{y:.3f}', ha='center', fontsize=9)
		
		plt.tight_layout()
		plt.savefig(save_path, dpi=300, bbox_inches='tight')
		print(f"✅ F1 & Recall progression saved to {save_path}")
		plt.close()

	# NEW: Cumulative metrics plot
	def plot_cumulative_metrics(self):
		"""Plot cumulative average of F1, Recall, Precision, Accuracy"""
		if len(self.cumulative_f1) == 0:
			return
		
		save_path = f'global_cumulative_metrics_round_{self.current_round}.png'
		
		rounds = list(range(1, len(self.cumulative_f1) + 1))
		
		fig, ax = plt.subplots(figsize=(12, 7))
		
		ax.plot(rounds, self.cumulative_f1, 'o-', linewidth=2.5, markersize=8, 
		       color='#FF6B6B', label='Cumulative F1')
		ax.plot(rounds, self.cumulative_recall, 's-', linewidth=2.5, markersize=8, 
		       color='#4ECDC4', label='Cumulative Recall')
		ax.plot(rounds, self.cumulative_precision, '^-', linewidth=2.5, markersize=8, 
		       color='#95E1D3', label='Cumulative Precision')
		ax.plot(rounds, self.cumulative_accuracy, 'd-', linewidth=2.5, markersize=8, 
		       color='#F38181', label='Cumulative Accuracy')
		
		ax.set_xlabel('Partition Number', fontsize=13, fontweight='bold')
		ax.set_ylabel('Score (Cumulative Average)', fontsize=13, fontweight='bold')
		ax.set_title('Global Model: Cumulative Metrics Progression', fontsize=14, fontweight='bold')
		ax.grid(alpha=0.3, linestyle='--')
		ax.set_ylim([0, 1.05])
		ax.legend(fontsize=11, loc='lower right')
		
		# Add final values annotation
		textstr = f'Final Cumulative:\nF1: {self.cumulative_f1[-1]:.4f}\nRecall: {self.cumulative_recall[-1]:.4f}\nPrecision: {self.cumulative_precision[-1]:.4f}\nAccuracy: {self.cumulative_accuracy[-1]:.4f}'
		props = dict(boxstyle='round', facecolor='wheat', alpha=0.7)
		ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=10,
		       verticalalignment='top', bbox=props)
		
		plt.tight_layout()
		plt.savefig(save_path, dpi=300, bbox_inches='tight')
		print(f"✅ Cumulative metrics saved to {save_path}")
		plt.close()

	def plot_metrics_progression(self, save_path='global_metrics_progression.png'):
		"""Plot progression of all metrics"""
		if not self.partition_metrics:
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
		axes[0, 0].set_title('AUC-ROC Progression', fontsize=12, fontweight='bold')
		axes[0, 0].grid(alpha=0.3)
		axes[0, 0].set_ylim([0, 1.05])
		
		axes[0, 1].plot(rounds, accuracies, 'o-', linewidth=2, markersize=8, color='#A23B72')
		axes[0, 1].set_xlabel('Partition Number', fontsize=11)
		axes[0, 1].set_ylabel('Accuracy', fontsize=11)
		axes[0, 1].set_title('Accuracy Progression', fontsize=12, fontweight='bold')
		axes[0, 1].grid(alpha=0.3)
		axes[0, 1].set_ylim([0, 1.05])
		
		axes[1, 0].plot(rounds, precisions, 'o-', linewidth=2, markersize=8, 
					   color='#F18F01', label='Precision')
		axes[1, 0].plot(rounds, recalls, 's-', linewidth=2, markersize=8, 
					   color='#C73E1D', label='Recall')
		axes[1, 0].set_xlabel('Partition Number', fontsize=11)
		axes[1, 0].set_ylabel('Score', fontsize=11)
		axes[1, 0].set_title('Precision & Recall', fontsize=12, fontweight='bold')
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
		print(f"✅ Metrics progression saved to {save_path}")
		plt.close()
		
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
		print(f"Aggregated FI: {aggregated_feature_importances}")
		print(f"\nRetraining global model after partition {self.current_round}...\n")
		
		scaled_importances = aggregated_feature_importances * 100
		sample_weights = np.dot(self.global_x_train, scaled_importances)
		sample_weights = (sample_weights - np.min(sample_weights)) / (np.max(sample_weights) - np.min(sample_weights))
		
		self.train(self.global_x_train, self.global_x_val, self.global_y_train, self.global_y_val, sample_weights, is_initial=False)

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
		dataset = dataset.sample(frac=1, random_state=42).reset_index(drop=True)
		X = dataset.iloc[:, :-1]
		Y = dataset.iloc[:, -1]
		return X, Y
	except Exception as e:
		print(f"❌ Error loading file: {e}")
		return None, None


def preprocess_data(X, Y):
	print("\n" + "="*70)
	print("PATIENT DATASET PREPROCESSING")
	print("="*70)
	print(f"Initial shape: {X.shape}")
	
	X = X.copy()
	
	if 'id' in X.columns or X.columns[0] == 'id':
		X = X.drop(columns=['id'])
		print(f"✅ Dropped 'id' column. New shape: {X.shape}")
	
	if 'bmi' in X.columns:
		X['bmi'] = X['bmi'].replace('N/A', np.nan)
		X['bmi'] = pd.to_numeric(X['bmi'], errors='coerce')
		print(f"✅ Cleaned 'bmi' column")

	print(f"\n📋 COLUMNS AFTER DROPPING ID:")
	for idx, col in enumerate(X.columns):
		print(f"   Index {idx}: {col}")
	
	numeric_cols = ['age', 'hypertension', 'heart_disease', 'avg_glucose_level', 'bmi']
	numeric_cols = [col for col in numeric_cols if col in X.columns]
	
	categorical_cols = ['gender', 'ever_married', 'work_type', 'Residence_type', 'smoking_status']
	categorical_cols = [col for col in categorical_cols if col in X.columns]
	
	print(f"\n📊 Numeric: {numeric_cols}")
	print(f"📊 Categorical: {categorical_cols}")
	
	if numeric_cols:
		imputer = SimpleImputer(strategy='median')
		X[numeric_cols] = imputer.fit_transform(X[numeric_cols])
		print(f"✅ Imputed missing values")
	
	for col in categorical_cols:
		le = LabelEncoder()
		X[col] = X[col].fillna('Unknown')
		X[col] = le.fit_transform(X[col].astype(str))
		print(f"✅ Encoded '{col}'")
	
	leY = LabelEncoder()
	Y = Y.fillna(0)
	Y = leY.fit_transform(Y.astype(str))
	
	print(f"\n✅ Final shape: {X.shape}")
	print(f"Target distribution: {dict(zip(*np.unique(Y, return_counts=True)))}")
	print("="*70 + "\n")
	
	return X.values, Y


def split_data(X, Y, n):
	print(f"\nSplitting data into {n} partitions")
	x_array = np.array_split(X, n)
	y_array = np.array_split(Y, n)
	return x_array, y_array


def scale_and_synthesize_train_only(x_train, y_train):
	columns_to_scale = [1,7,8]
	
	scaler = StandardScaler()
	x_train_scaled = x_train.copy()
	x_train_scaled[:, columns_to_scale] = scaler.fit_transform(x_train[:, columns_to_scale])
	
	unique_before, counts_before = np.unique(y_train, return_counts=True)
	print(f"Class BEFORE SMOTE: {dict(zip(unique_before, counts_before))}")
	
	min_count = np.min(counts_before)
	k_neighbors = min(5, min_count - 1) if min_count > 1 else 1
	
	smote = SMOTE(random_state=8, k_neighbors=k_neighbors)
	x_train_resampled, y_train_resampled = smote.fit_resample(x_train_scaled, y_train)
	
	unique_after, counts_after = np.unique(y_train_resampled, return_counts=True)
	print(f"Class AFTER SMOTE: {dict(zip(unique_after, counts_after))}")
	
	return x_train_resampled, y_train_resampled, scaler


def scale_eval_data(x_eval, scaler):
	columns_to_scale = [1,7,8]
	x_eval_scaled = x_eval.copy()
	x_eval_scaled[:, columns_to_scale] = scaler.transform(x_eval[:, columns_to_scale])
	return x_eval_scaled


def add_commitments(commitments):
	val = commitments[0]
	for i in range(1, len(commitments)):
		val = add(val, commitments[i])
	return val


if __name__ == "__main__":
	print("\n" + "="*70)
	print("ZERO-KNOWLEDGE PROOF FEDERATED LEARNING SYSTEM")
	print("WITH F1 & RECALL TRACKING + CUMULATIVE METRICS")
	print("="*70 + "\n")
	
	n = 3
	main_port = 6000
	
	print(f"Initializing global model on port {main_port}...")
	global_model = global_mod(main_port, n)
	print("✅ Global model initialized\n")
	
	path = r"Sample Data\stroke_data.csv"
	print(f"Loading data from: {path}")
	X, Y = load_data(path)
	
	if X is None or Y is None:
		print("\n❌ Exiting due to data load failure...")
		exit(1)
	
	X, Y = preprocess_data(X, Y)
	
	print("\n" + "="*70)
	print("DATA SPLITTING - NO LEAKAGE")
	print("="*70)
	
	X_train_all, X_test, Y_train_all, Y_test = train_test_split(
		X, Y, test_size=0.2, random_state=1, stratify=Y
	)
	
	X_train_all, X_val, Y_train_all, Y_val = train_test_split(
		X_train_all, Y_train_all, test_size=0.25, random_state=1, stratify=Y_train_all
	)
	
	print(f"Total data: {len(X)}")
	print(f"Training (60%): {len(X_train_all)}")
	print(f"Validation (20%): {len(X_val)}")
	print(f"Test (20% - HOLDOUT): {len(X_test)}")
	print("="*70 + "\n")
	
	x_array, y_array = split_data(X_train_all, Y_train_all, n+1)
	
	print("\nProcessing global model's data...")
	x_train_global_raw = x_array[n]
	y_train_global_raw = y_array[n]
	
	x_train_global, y_train_global, global_scaler = scale_and_synthesize_train_only(
		x_train_global_raw, 
		y_train_global_raw
	)
	
	x_val_global = scale_eval_data(X_val, global_scaler)
	x_test_global = scale_eval_data(X_test, global_scaler)
	
	print(f"Global train after SMOTE: {len(y_train_global)}")
	print(f"Global validation (no SMOTE): {len(Y_val)}")
	print(f"Global test (no SMOTE): {len(Y_test)}")
	
	global_model.set_global_data(
		x_train_global, 
		x_val_global, 
		x_test_global, 
		y_train_global, 
		Y_val, 
		Y_test
	)
	
	print("\n" + "="*70)
	print("Training initial global model (on VALIDATION set)...")
	print("="*70)
	global_model.train(x_train_global, x_val_global, y_train_global, Y_val, is_initial=True)
	print("✅ Initial training complete.")
	
	print("\n" + "="*70)
	print("Waiting for local models to connect...")
	print("="*70 + "\n")
	
	time.sleep(1000)