import argparse
import os
import sys

sys.path.insert(0, '..')
from qsub import qsub

def main():
	parser = argparse.ArgumentParser()
	parser.add_argument('-d', '--dir', dest='dir', default='.')
	parser.add_argument('--src', dest='src', default='../src')
	
	parser.add_argument("--data-dir", dest="data_dir", default='../data/eating_detection_inertial_ubicomp2015', type=str, \
			help="Directory where the dataset is stored")
	
	parser.add_argument("--bag-size", dest="bag_size", default=-1, type=int, \
			help="If clf is an MIL classifier, bag-size specifies the size of each training bag")
	parser.add_argument("--held-out-bag-size", dest="held_out_bag_size", default=-1, type=int, \
			help=".")
	parser.add_argument("--test-bag-size", dest="test_bag_size", default=1, type=int, \
			help=".")	

	parser.add_argument("--N", dest="N", default=20, type=int, \
			help="Number of instances used for training in each LOPO iteration")
	parser.add_argument("--M", dest="M", default=100, type=int, \
			help="Number of single-instance bags used for training in each LOPO iteration")
	parser.add_argument("--K", dest="K", default=0, type=int, \
			help="Number of single-instance bags in the training data from the held-out participant.")
	
	parser.add_argument("--clf", dest="clf", default='sbMIL', type=str, \
			help="Classifier ('RF', 'SVM', 'LinearSVC', 'SIL', 'LinearSIL', 'MIForest', 'sMIL', 'sbMIL', 'misvm')")
	parser.add_argument("--eta", dest="eta", default=0.5, type=float, \
			help="Balancing parameter for sbMIL, between 0 and 1 inclusively")	
	parser.add_argument("--kernel", dest="kernel", default='linear', type=str, \
			help="Kernel type, i.e. 'linear', 'rbf', 'linear_av', etc.")
			
	parser.add_argument("--cv-method", dest="cv_method", default='randomized', type=str, \
			help="Determines how hyperparameters are learned ('grid' or 'randomized')")
	parser.add_argument("--cv", dest="cv", default=3, type=int, \
			help="Determines split for cross-validation (see GridSearchCV.cv)")
	parser.add_argument("--n-iter", dest="n_iter", default=10, type=int, \
			help="The number of iterations in randomized cross-validation (see RandomizedSearchCV.cv)")
	parser.add_argument("--n-trials", dest="n_trials", default=5, type=int, \
			help="Number of trials over which to average the performance metrics")
	parser.add_argument("--n-jobs", dest="n_jobs", default=1, type=int, \
			help="Number of threads used (default = 1). Use -1 for maximal parallelization")
	parser.add_argument("--n-jobs-per-iter", dest="n_jobs_per_iter", default=1, type=int, \
			help="Number of machines used for each run of GridSearchCV")
			
	parser.add_argument("--verbose", dest="verbose", default=1, type=int, \
			help="Indicates how much information should be reported (0=None, 1=Some, 2=Quite a bit)")	
	parser.add_argument("--desc", dest="desc", default='', type=str, \
			help="Description of the evaluation / parameter selection")

	args = parser.parse_args()

	working_dir = args.dir
	if not os.path.isdir(working_dir):
		os.mkdir(working_dir, 0755)

	log_dir = working_dir + '/log'
	if not os.path.isdir(log_dir):
		os.mkdir(log_dir, 0755)
	
	err_dir = working_dir + '/err'
	if not os.path.isdir(err_dir):
		os.mkdir(err_dir, 0755)
		
	res_dir = working_dir + '/res'
	if not os.path.isdir(res_dir):
		os.mkdir(res_dir, 0755)
		
	params_dir = working_dir + '/params'
	if not os.path.isdir(params_dir):
		os.mkdir(params_dir, 0755)
	
	arg_str = ''
	for arg in args._get_kwargs():
		if arg[0] not in {'src', 'dir', 'cv', 'n_jobs_per_iter'}:
			arg_str += ' --' + arg[0].replace('_', '-') + '="' + str(arg[1]) + '"'
		
	with open(os.path.join(params_dir, 'params.txt'), 'wb') as f:
		f.write(arg_str)
	
	#pre-load dataset once (to find number of participants)
	sys.path.insert(0, args.data_dir)
	from load_data import load_data
	dataset = load_data(args.data_dir)
	
	n_jobs_per_iter = args.n_jobs_per_iter
	if args.cv_method == 'grid':
		n_jobs_per_iter = 1
	n_participants = len(dataset['data']['Y'])
	n_iter = int(args.n_iter / n_jobs_per_iter)
	for p in range(n_participants):
		for i in range(1,n_jobs_per_iter):
			save_path = os.path.join(res_dir, 'lopo_p%d_i%d.pickle' % (p,i))
			submit_this_job = 'python %s/w_lopo.py --save=%s --test-participant=%d --cv=%d ' % (args.src, save_path, p, n_iter) + arg_str
			print submit_this_job
			job_id = 'lopo_p%d_i%d' % (p,i)
			log_file = log_dir + '/lopo_p%d_i%dlog.txt' % (p,i)
			err_file = err_dir + '/lopo_p%d_i%derr.txt' % (p,i)
			qsub(submit_this_job, job_id, log_file, err_file, n_cores=args.n_jobs)

if __name__ == '__main__':
	main()

