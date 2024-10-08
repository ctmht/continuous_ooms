import scipy as sp

from src.oom import *

if __name__ == '__main__':
	n_obs = 5
	src_dim = 50
	
	observables = []
	operators = []
	for uidx in range(n_obs):
		# Create observable
		name = chr(ord("a") + uidx)
		observable = Observable(name)
		observables.append(observable)
		
		# Create operator
		matrix_rep = np.asmatrix(sp.sparse.random(src_dim, src_dim, density = 1.5/src_dim, random_state=42 + uidx).toarray())
		operator = Operator(observable, src_dim, matrix_rep)
		operators.append(operator)
	
	# Create linear functional
	linear_func = LinFunctional(np.ones(src_dim))
	
	# Create starting state
	start_state = np.random.random([src_dim, 1])
	start_state = start_state/(linear_func * start_state)
	start_state = State(start_state)
	
	# Print
	# print('w0\n', start_state, '\n\ns0\n', linear_func, '\n')
	# for o in observables:
	# 	print(o.name, end=' ')
	# print('\n')
	# for op in operators:
	# 	print(op.observable.name, '\n', op.mat, end='\n\n')
	
	# Force operators to be valid
	colsum = [np.sum(op.mat, axis = 0) for op in operators]
	# print(f"Colwise sum\n", [i for i in colsum])
	
	colsum = np.sum(colsum, axis = 0).flatten()
	# print(f"Colwise sum\n", colsum)
	for col in range(src_dim):
		for op in operators:
			op.mat[:, col] /= colsum[col]
	
	# Create OOM
	big_oom = DiscreteValuedOOM(src_dim, linear_func, operators, start_state)
	
	# Generate sequence
	states, sequence = big_oom.generate(length = 5000)
	sequence = [obs.name for obs in sequence]
	sequence = "".join(sequence)
	print(sequence)
	
	# Attempt to learn OOM from sequence
	possible_obs = ["a", "b", "c", "d", "e"]
	characterisic_evts = ["a", "b", "c", "d", "e", "ab", "bc", "cd", "de", "ea", "ace", "bda", "ceb", "dab", "ebd",
					"adbe", "beca", "cadb", "dbec", "ecad"]
	indicative_evts = ["a", "b", "c", "d", "e", "ab", "bc", "cd", "de", "ea", "ace", "bda", "ceb", "dab", "ebd",
					"adbe", "beca", "cadb", "dbec", "ecad"]
	
	print(f"Linear functional: {linear_func.flatten()}\n"
		  f"V[lf] = {np.var(linear_func)}\n"
		  f"Starting state   : {start_state.flatten()}\n"
		  f"V[ss] = {np.var(start_state)}\n\n")
	
	for q in range(2, 20):
		(sig, tau, omega), F_IJ = estimate_OOM(sequence, possible_obs,
											   characterisic_evts, indicative_evts, q)
		est_ops = [Operator(Observable(possible_obs[idx]), q, mat) for idx, mat in enumerate(tau)]
		estimated_oom = DiscreteValuedOOM(q, sig, est_ops, omega)
		
		print(60*'-' + f"\nEstimation of OOM using a dimension of dim = {q}\n" + 60*'-' + '\n'
			  f"Linear functional_est: {sig.flatten()}\n"
			  f"V[lf_est] = {np.var(sig)}\n"
			  f"V[lf] - V[lf_est] = {np.var(linear_func) - np.var(sig)}\n\n"
			  f"Starting state_est   : {omega.flatten()}\n"
			  f"V[ss_est] = {np.var(omega)}\n"
			  f"V[ss] - V[ss_est] = {np.var(start_state) - np.var(omega)}\n\n")
		