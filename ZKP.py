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

def serialize_ZKP_json(json_data):
	for k, v in json_data.items():
		if isinstance(v, tuple):
			json_data[k] = {"point":[int(i) for i in v]}
		elif isinstance(v, list):
			serialize_ZKP_list(v)
		elif isinstance(v, dict):
			serialize_ZKP_json(v)

def serialize_ZKP_list(list_data):
	for i in range(len(list_data)):
		if isinstance(list_data[i], tuple):
			list_data[i] = {"point":[int(j) for j in list_data[i]]}
		elif isinstance(list_data[i], list):
			serialize_ZKP_list(list_data[i])
		elif isinstance(list_data[i], dict):
			serialize_ZKP_json(list_data[i])

# Global parameters
G = G1
secret = "secret_string"
H = multiply(G1, int(hashlib.sha256(secret.encode()).hexdigest(), 16) % curve_order)



"""proof = {'C': (2681829019758089291043906137638745708186004183266120803928666737180509098779, 8641245553116585857299808797930874563378881582163105144900975711920733742786), 'C_w_low': (1783145320998007104199549117843330947385293907517374333185259538575101133512, 11029605381459877049064908240414905102567206822702820509950263419850456801548), 'C_w_high': (14798489800485762288808054575626064460014167624885025788788990177267745240524, 8383245445988534321588668550159540308512503548179923650791395691617343459141), 'bit_commitments_w_low': [(18694330278823901706300923666022471111584441951375230118320594578363653989125, 2232281443113810393471465198713421219782619298178625119340226409635706499660), (16078731677335235243267947395233768278552818913149623140042948963962962029291, 20197225293187677345850993239579485587854780360986772262219264148442908609459), (17432277700511934553871042781541797529494450777558957271484742080014131646202, 3894699664683885005520877876826417535280006232059429848806520827736625059591), (7408112493072393998614166732254169106795468895810825037276654435351889748704, 15382088397693611509659119118934004299848327808368689890239354934397927437322), (17951913387783762993642958501716661234828629246794746885522466804809936928348, 15644134044116091848556751882575631654879289052459357167807785892649276589446), (9077447493812076844799405919177588598070966606722541245253306619981861778570, 15630872839368628560713894979817687608241700710570137677813359911913036571540), (5154610189111701880604576378759104735390170869653064212725406470696892955050, 9853826136397875139180143609647222071705164501655372197780770061612977238139), (9621048997793743224715423861458173803502842176632496577280757739169528569334, 9825086717485670184024642591727892577668964250328080545019146746726074580024), (603232767456789847466514230004148349117879212878840663976101268292933204729, 15793830778332043647595066170290204706471927508490796189809624737679914651850), (982402492668188725229490593810525087369403861943050664795404597236641479310, 16274715499151536501211056020698465664858350652178071296934025864318877203793), (21576703017265616467902183489576125625734247457518531560634952482759812295323, 14237495781225408438923182735365784604138384603741992965038126560613165720702)], 'bit_commitments_w_high': [(18295255661707026587112985406157112543011057228700166163584039893967775672317, 11310871844409951607305549173211753915759609529752075363098911288856664897361), (15088322948100553615747818529147166220829739066544333053236032951998708798292, 3083465508046344012903281039286723544725254309661114729320089035964509381895), (16848623800162015996060981823429875219089978197775018648479598234445519136654, 17564906073960037480837899502141081188500435463285677908402121162057798328104), (7844989433606421319767895011952359824596702744184884515562089948125494539973, 16376849701498115705589512841416553863025140773880708157585298751838229929755), (21639777285741144201942509248905976371210321043471212242442261474410301175752, 3006373972028680518935864373938000645480303996417744972134895976922026766748), (15725870638707981974577946118265968810170896054170997592824390749200213502076, 20993305129858958578559140149069426520089400048572525331493161523438998518167), (9985025955382733071150677849662161641223719592120770740738015446762016912059, 11216680832786150061811482376976636894257117992218086457658823796816277946079), (8452298679210806154391249242792378849841814490761674127525763473831374168162, 18703229970008305915470811370441258304217818819348089521187640483796508378881), (13703199588472394305522715010043479729867121616056555668391215774351281207830, 5078361023119377576527470287864153581786259723677429730446149322015435262085), (1177158267690838931792007573748397374152401283742595659809016132124849944261, 8482379618935392404095488831987108350759959352383930941397465864794186593560), (20075422002826313967932004530653123742321070738382219804795585559460175631793, 19002727263700623700446489130321684109136469691210997793573484210204780853152)], 'zw_low': 16972347806932661584517590242743521344433401957724577059836034162246256686260, 'zr_low': 6388818167442957978198690398140672612364953787926458225920278516656815597721, 'zw_high': 16972347806932661584517590242743521344433401957724577059836034162246256725582, 'zr_high': 10416461799001844323535132372396237305295290916267396706704314502554481517700}
print(proof)
serialize_ZKP_json(proof)
print(proof)"""
