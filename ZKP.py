from py_ecc.bn128 import G1, add, multiply, curve_order
from secrets import randbelow
import hashlib
import time

def pedersen_commit(v, r, G, H):
	# C = vG + rH
	return add(multiply(G, v % curve_order), multiply(H, r % curve_order))

def generate_challenge(input_data):
	hash_input = str(input_data).encode()
	return int(hashlib.sha256(hash_input).hexdigest(), 16) % curve_order

def create_proof(w, r, low, high, C, G, H):
	bit_length = (high - low).bit_length()
	w_low = w - low
	C_w_low = pedersen_commit(w_low, r, G, H)
	w_high = high - w
	C_w_high = pedersen_commit(w_high, r, G, H)
	
	bit_commitments_w_low = []
	r_bits_w_low = []
	for i in range(bit_length):
		bit_value = (w_low >> i) & 1
		r_bit = randbelow(curve_order)
		C_bit = pedersen_commit(bit_value, r_bit, G, H)
		bit_commitments_w_low.append(C_bit)
		r_bits_w_low.append(r_bit)
	
	bit_commitments_w_high = []
	r_bits_w_high = []
	for i in range(bit_length):
		bit_value = (w_high >> i) & 1
		r_bit = randbelow(curve_order)
		C_bit = pedersen_commit(bit_value, r_bit, G, H)
		bit_commitments_w_high.append(C_bit)
		r_bits_w_high.append(r_bit)
	
	hash_input = f"{C}|{bit_commitments_w_low}|{bit_commitments_w_high}".encode()
	c = generate_challenge(hash_input)
	print(f"Challenge c = {c}")
	
	zw_low = (w_low + c * w) % curve_order
	zr_low = (sum(r_bits_w_low[i] * (2 ** i) for i in range(bit_length)) + c * r) % curve_order
	
	zw_high = (w_high + c * w) % curve_order
	zr_high = (sum(r_bits_w_high[i] * (2 ** i) for i in range(bit_length)) + c * r) % curve_order
	
	print(f"\nResponses for w-low: zw_low = {zw_low}, zr_low = {zr_low}")
	print(f"Responses for high-w: zw_high = {zw_high}, zr_high = {zr_high}")
	
	proof = {
		'C': C,
		'C_w_low': C_w_low,
		'C_w_high': C_w_high,
		'bit_commitments_w_low': bit_commitments_w_low,
		'bit_commitments_w_high': bit_commitments_w_high,
		'zw_low': zw_low,
		'zr_low': zr_low,
		'zw_high': zw_high,
		'zr_high': zr_high
	}
	print(f"Proof: {proof}")
	
	return proof

def validate_proof(proof, low, high):
	C = proof['C']
	C_w_low = proof['C_w_low']
	C_w_high = proof['C_w_high']
	bit_commitments_w_low = proof['bit_commitments_w_low']
	bit_commitments_w_high = proof['bit_commitments_w_high']
	zw_low = proof['zw_low']
	zr_low = proof['zr_low']
	zw_high = proof['zw_high']
	zr_high = proof['zr_high']
	
	hash_input = f"{C}|{bit_commitments_w_low}|{bit_commitments_w_high}".encode()
	c = generate_challenge(hash_input)
	print(f"Challenge c = {c}")
	
	if add(C_w_low, multiply(G, low)) == C:
		print("\nCheck 1 passed: C = C_w_low + l.G")
	else:
		print("\nProof failed: Incorrect w' commitment!")
		return False
		
	if add(C_w_high, multiply(G, high)) == C:
		print("\nCheck 2 passed: C = C_w_high + h.G")
	else:
		print("\nProof failed: Incorrect w'' commitment!")
		return False
	
	#Check if hidden value is within given range
	lhs_low = add(multiply(G, zw_low), multiply(H, zr_low))
	rhs_low = multiply(C, c)
	for i in range(bit_length):
		rhs_low = add(rhs_low, multiply(bit_commitments_w_low[i], (2 ** i) % curve_order))
	
	lhs_high = add(multiply(G, zw_high), multiply(H, zr_high))
	rhs_high = multiply(C, c)
	for i in range(bit_length):
		rhs_high = add(rhs_high, multiply(bit_commitments_w_high[i], (2 ** i) % curve_order))
	
	if lhs_low == rhs_low and lhs_high == rhs_high:
		print(f"\nProof successful: w lies in the range [{low}, {high}].")
	else:
		print(f"\nProof failed: w does not lie in the range [{low}, {high}].")
		return False
	
	return True

# Global parameters
G = G1
secret = "secret_string"
H = multiply(G1, int(hashlib.sha256(secret.encode()).hexdigest(), 16) % curve_order)



