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
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import classification_report, roc_curve, auc, roc_auc_score
from imblearn.over_sampling import SMOTE
from sklearn.metrics import confusion_matrix
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from imblearn.pipeline import Pipeline
from imblearn.under_sampling import RandomUnderSampler  
from sklearn.preprocessing import LabelEncoder
from threading import Lock
import matplotlib.pyplot as plt

np.set_printoptions(linewidth=np.inf)

class local_mod:
	def __init__(self, id, port, other_ports, global_port):
		self.lock = threading.Lock()
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
		self.commitments_log = []
		self.proof_sent = {}
		self.partial_sum_sent = {}
		self.range_from_global = []
		
		# Metrics tracking
		self.partition_metrics = []
		self.auc_history = []
		self.current_round = 0
		
		# NEW: Cumulative metrics
		self.cumulative_f1 = []
		self.cumulative_recall = []
		self.cumulative_precision = []
		self.cumulative_accuracy = []
		
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

		print(f"LOCAL Hospital {self.id} received message type {data['type']} from port {data.get('port', 'unknown')}")

		if (data["type"]==2):
			self.range_from_global = data["ranges"]
			print(f"Hospital {self.id}: Received ranges from global")
			print(f"Ranges: {self.range_from_global}")
		elif (data["type"]==4):
			print(f'\nModel {self.id} validity: {data["validity"]}\n')
		elif (data["type"]==5):
			print(f"Hospital {self.id}: Received valid ports - STARTING MPC")
			print(f"Hospital {self.id}: Valid models = {data['valid models']}")
			self.partial_r = 0
			self.partial_fi = [0]*len(self.get_feature_importances())
			self.partial_no = 0
			self.split_and_send_data(data)
		elif (data["type"]==6):
			print(f"Hospital {self.id} Received partial data from port {data['port']}")
			if not hasattr(self, 'valid_models') or len(self.valid_models) == 0:
				print(f"[WARNING] Hospital {self.id} received partial data but no valid models set")
				return
			self.update_partial_sum(data)
			print(f"Hospital {self.id}: partial_no = {self.partial_no}/{len(self.valid_models)}")
			
			if self.partial_no == len(self.valid_models):
				if self.partial_sum_sent.get(self.current_round, False):
					print(f"[WARNING] Hospital {self.id} already sent partial sum for round {self.current_round}")
					return		

				print(f"Partial r for local hospital {self.id}: {self.partial_r}")
				print(f"Partial feature importances for local hospital {self.id}: {self.partial_fi}")
				self.send_partial_sum(self.current_round)
				self.partial_sum_sent[self.current_round] = True
				
				self.partial_r = 0
				self.partial_fi = [0]*len(self.get_feature_importances())
				self.partial_no = 0
		else:
			print("Random access received.")

	def train(self, x_train, x_test, y_train, y_test):
		"""Enhanced training with comprehensive metrics tracking"""
		if x_test.ndim == 1:
			x_test = x_test.reshape(1, -1)
		if x_train.ndim == 1:
			x_train = x_train.reshape(1, -1)
			y_train = np.array([y_train])

		self.model.fit(x_train, y_train)
		y_pred_proba = self.model.predict_proba(x_test)[:, 1]
		
		threshold = 0.75
		y_pred = (y_pred_proba >= threshold).astype(int)
		
		accuracy = accuracy_score(y_test, y_pred)
		tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
		actual_positives = tp + fn
		actual_negatives = tn + fp
		
		recall = recall_score(y_test, y_pred, zero_division=0)
		precision = precision_score(y_test, y_pred, zero_division=0)
		f1 = f1_score(y_test, y_pred, zero_division=0)
		
		try:
			auc_score = roc_auc_score(y_test, y_pred_proba)
			fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
			
			metrics = {
				'round': self.current_round,
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
			
			# NEW: Calculate cumulative metrics
			self.cumulative_f1.append(np.mean([m['f1'] for m in self.partition_metrics]))
			self.cumulative_recall.append(np.mean([m['recall'] for m in self.partition_metrics]))
			self.cumulative_precision.append(np.mean([m['precision'] for m in self.partition_metrics]))
			self.cumulative_accuracy.append(np.mean([m['accuracy'] for m in self.partition_metrics]))
			
			print(f"\n{'='*70}")
			print(f"HOSPITAL {self.id} - Partition {self.current_round} Results")
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
			
			# NEW: Print cumulative metrics
			print(f"\n📊 CUMULATIVE METRICS (Average up to Round {self.current_round}):")
			print(f"   Cumulative F1: {self.cumulative_f1[-1]:.4f}")
			print(f"   Cumulative Recall: {self.cumulative_recall[-1]:.4f}")
			print(f"   Cumulative Precision: {self.cumulative_precision[-1]:.4f}")
			print(f"   Cumulative Accuracy: {self.cumulative_accuracy[-1]:.4f}")
			print(f"{'='*70}\n")
			
		except Exception as e:
			print(f"Error calculating AUC for Hospital {self.id}: {e}")
			auc_score = None
		
		self.last_accuracy = accuracy
		
		print(classification_report(y_test, y_pred))
		print(f"\nFeature importances: {self.model.feature_importances_.tolist()}")

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

	# NEW: F1 and Recall progression plot
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

	# NEW: Cumulative metrics plot
	def plot_cumulative_metrics(self):
		"""Plot cumulative average of F1, Recall, Precision, Accuracy"""
		if len(self.cumulative_f1) == 0:
			return
		
		save_path = f'hospital_{self.id}_cumulative_metrics_round_{self.current_round}.png'
		
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
		ax.set_title(f'Hospital {self.id}: Cumulative Metrics Progression', 
		            fontsize=14, fontweight='bold')
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
		print(f"Hospital {self.id}: Cumulative metrics saved to {save_path}")
		plt.close()

	def plot_auc_curves_final(self, save_path=None):
		"""Plot all ROC curves together - final version"""
		if not self.partition_metrics:
			print(f"Hospital {self.id}: No metrics available to plot")
			return
		
		if save_path is None:
			save_path = f'hospital_{self.id}_auc_curves_final.png'
		
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
		plt.title(f'Hospital {self.id}: ROC Curves (All Partitions)', fontsize=14, fontweight='bold')
		plt.legend(loc="lower right", fontsize=10)
		plt.grid(alpha=0.3)
		plt.tight_layout()
		plt.savefig(save_path, dpi=300, bbox_inches='tight')
		print(f"Hospital {self.id}: Final AUC curves saved to {save_path}")
		plt.close()

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
		
		# Accuracy progression
		axes[0, 1].plot(rounds, accuracies, 'o-', linewidth=2, markersize=8, color='#A23B72')
		axes[0, 1].set_xlabel('Partition Number', fontsize=11)
		axes[0, 1].set_ylabel('Accuracy', fontsize=11)
		axes[0, 1].set_title(f'Hospital {self.id}: Accuracy Progression', fontsize=12, fontweight='bold')
		axes[0, 1].grid(alpha=0.3)
		axes[0, 1].set_ylim([0, 1.05])
		
		# Precision & Recall
		axes[1, 0].plot(rounds, precisions, 'o-', linewidth=2, markersize=8, 
					   color='#F18F01', label='Precision')
		axes[1, 0].plot(rounds, recalls, 's-', linewidth=2, markersize=8, 
					   color='#C73E1D', label='Recall')
		axes[1, 0].set_xlabel('Partition Number', fontsize=11)
		axes[1, 0].set_ylabel('Score', fontsize=11)
		axes[1, 0].set_title(f'Hospital {self.id}: Precision & Recall', fontsize=12, fontweight='bold')
		axes[1, 0].legend(fontsize=10)
		axes[1, 0].grid(alpha=0.3)
		axes[1, 0].set_ylim([0, 1.05])
		
		# F1 Score
		axes[1, 1].plot(rounds, f1s, 'o-', linewidth=2, markersize=8, color='#6A994E')
		axes[1, 1].set_xlabel('Partition Number', fontsize=11)
		axes[1, 1].set_ylabel('F1 Score', fontsize=11)
		axes[1, 1].set_title(f'Hospital {self.id}: F1 Score Progression', fontsize=12, fontweight='bold')
		axes[1, 1].grid(alpha=0.3)
		axes[1, 1].set_ylim([0, 1.05])
		
		plt.tight_layout()
		plt.savefig(save_path, dpi=300, bbox_inches='tight')
		print(f"Hospital {self.id}: Metrics progression saved to {save_path}")
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
		self.plot_auc_curves_final()
		self.plot_metrics_progression()
		self.plot_f1_recall_progression()  # NEW
		self.plot_cumulative_metrics()      # NEW
		print(f"Hospital {self.id}: Final report generated successfully\n")

	def get_feature_importances(self):
		feature_importances = self.model.feature_importances_.tolist()
		feature_importances = [int(i*(10**6)) for i in feature_importances]
		return feature_importances

	def generate_proof(self, current_round):  
		if self.port == 6001:
			print(f"[H1-LOCAL-DEBUG] Starting generate_proof for round {current_round}")
			print(f"[H1-LOCAL-DEBUG] Current self.current_round: {getattr(self, 'current_round', 'NOT_SET')}")
			print(f"[H1-LOCAL-DEBUG] Proof already sent: {self.proof_sent.get(current_round, False)}")
    	
		if self.proof_sent.get(current_round, False):
			if self.port == 6001:
				print(f"[Hospital {self.id}] Already sent proof for round {current_round}")
			return  
	
		feature_importances = self.get_feature_importances()
		r = randbelow(curve_order)
		self.blinding_factor = r
		commitments = []

		print(f"\nGenerating commitments for hospital {self.id} (Round {current_round})...")
		for i in range(len(feature_importances)):
			commitment = pedersen_commit(feature_importances[i], r, G, H)
			commitments.append(commitment)
		
		while len(self.commitments_log) <= current_round:
			self.commitments_log.append([])
		self.commitments_log[current_round] = commitments

		print("Commitments\n", commitments)
		self.proof_sent[current_round] = True
		commitment_transaction = {"port":self.port, "type": 1, "commitments":commitments, "round":current_round}
		commitment_transaction_serialized = copy.deepcopy(commitment_transaction)
		serialize_ZKP_json(commitment_transaction_serialized)
		commitment_transaction_data = json.dumps(commitment_transaction_serialized).encode()
		print(f"Commitment transaction data {commitment_transaction_data}")
		print(f"Sending commitments for Round {current_round}...")
		
		with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
			try:
				s.connect(('localhost', self.global_port))
				print("Calling sendall")
				s.sendall(commitment_transaction_data)
				print(f"Commitments sent successfully for Round {current_round}.")
			except ConnectionRefusedError:
				print(f"Local model {self.id} could not connect to Node on port {self.global_port}")
		
		time.sleep(2)
		print(f"\nRanges received from global hospital.")
		proofs = []
		print(f"\nGenerating proofs for hospital {self.id} (Round {current_round})...")
		
		for i in range(len(feature_importances)):
			print(f"\nProof for feature importance {i+1}")
			print("Range: ", self.range_from_global[i])
			print("FI: ", feature_importances[i])
			proof = create_proof(feature_importances[i], r, self.range_from_global[i][0], 
							   self.range_from_global[i][1], commitments[i], G, H)
			print(f"Proof {i+1} generated.")
			proofs.append(proof)
		
		proof_transaction = {"port":self.port, "type": 3, "proofs":proofs}
		proof_transaction_serialized = copy.deepcopy(proof_transaction)
		serialize_ZKP_json(proof_transaction_serialized)
		proof_transaction_data = json.dumps(proof_transaction_serialized).encode()
		print("Sending proofs...")
		
		with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
			try:
				s.connect(('localhost', self.global_port))
				s.sendall(proof_transaction_data)
				print(f"Proofs sent successfully for Round {current_round}.")
			except ConnectionRefusedError:
				print(f"Local model {self.id} could not connect to Node on port {self.global_port}")
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
				MPC_data = {"type":6, "port": self.port, "r":r_list[i], 
						   "feature importance":[fi[i] for fi in fi_list]}
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
		partial_sum = {"type":7, "port":self.port, "partial r": self.partial_r, 
					  "partial fi":self.partial_fi, "round":round_number}
		partial_sum_encoded = json.dumps(partial_sum).encode()
		with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
			try:
				s.connect(('localhost', self.global_port))
				s.sendall(partial_sum_encoded)
				print(f"Hospital {self.id}: Successfully sent partial sum to global")
			except ConnectionRefusedError:
				print(f"Local model {self.id} could not connect to Node on port {self.global_port}")
			except Exception as e:
				print(f"Hospital {self.id}: Error sending partial sum: {e}")


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

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
	print(f"\nSplitting data into {n}")
	x_array = np.array_split(X, n)
	y_array = np.array_split(Y, n)
	return x_array, y_array

def test_train(x_array, y_array, n):
	x_train, x_test, y_train, y_test = [0]*n, [0]*n, [0]*n, [0]*n
	for i in range(n):
		x_train[i], x_test[i], y_train[i], y_test[i] = train_test_split(
			x_array[i], y_array[i], test_size=0.2, random_state=1)
	print("Data split for training and testing.")
	return x_train, x_test, y_train, y_test

def scale_and_synthesize(x_train, x_test, y_train):
	unique_before, counts_before = np.unique(y_train, return_counts=True)
	print(f"Class distribution BEFORE SMOTE: {dict(zip(unique_before, counts_before))}")
	
	columns_to_scale = [1,7,8]
	scaler = StandardScaler()
	
	x_train[:, columns_to_scale] = scaler.fit_transform(x_train[:, columns_to_scale])
	x_test[:, columns_to_scale] = scaler.transform(x_test[:, columns_to_scale])
	
	smote = Pipeline([
		('smote', SMOTE(random_state=8)),
		('undersample', RandomUnderSampler(random_state=8))
	])
	x_train_resampled, y_train_resampled = smote.fit_resample(x_train, y_train)
	
	unique_after, counts_after = np.unique(y_train_resampled, return_counts=True)
	print(f"Class distribution AFTER SMOTE: {dict(zip(unique_after, counts_after))}")
	
	return x_train_resampled, x_test, y_train_resampled

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


# ============================================================================
# MAIN EXECUTION
# ============================================================================

n = 3
main_port = 6000
local_hospitals = create_hospitals(n, main_port)
time.sleep(0.5)

path = "Sample Data/stroke_data.csv"
X, Y = load_data(path)

X, Y = preprocess_data(X, Y)

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.25, random_state=1)

x_array, y_array = split_data(x_train, y_train, n)

T = 5  # Number of partitions
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

		part_x, _, part_y = scale_and_synthesize(part_x, part_x, part_y)
		print("Class distribution after SMOTE:", Counter(part_y))
		unique_after_split, counts_after_split = np.unique(part_y, return_counts=True)
		class_dist_after = dict(zip(unique_after_split, counts_after_split))
		print(f"Hospital {i+1} - Class distribution AFTER SMOTE: {class_dist_after}")

		print(f"\nTraining Hospital {i+1} on P{part_no + 1}...")
		part_x_train, part_x_test, part_y_train, part_y_test = train_test_split(
			part_x, part_y, test_size=0.2, random_state=42
		)
		
		local_hospitals[i].current_round = part_no + 1
		
		local_hospitals[i].train(part_x_train, part_x_test, part_y_train, part_y_test)
		
		# NEW: Generate all plots after each partition
		local_hospitals[i].plot_cumulative_auc_curve()
		local_hospitals[i].plot_f1_recall_progression()
		local_hospitals[i].plot_cumulative_metrics()

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