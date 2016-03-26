import argparse
import os
import sys

sys.path.insert(0, '../tests/')
from qsub import qsub

participants = range(20)
#bag_size = 200
#N = 5
M = range(100,1001,50)

def main():
	parser = argparse.ArgumentParser()
	parser.add_argument('-d', '--dir', dest='dir', default='.')
	parser.add_argument('--src', dest='src', default='../src')
			
	parser.add_argument("--data-dir", dest="data_dir", default='../data/', \
			help="Directory where the dataset is stored")
	parser.add_argument("--load", dest="load_pickle_path", default='../data/data.pickle', \
			help="Path from which to load the pickle file. This will significantly speed up loading the data. If 'none' (default), the data will be reloaded from --dir")	
	parser.add_argument("--save", dest="save_pickle_path", default='../data/data.pickle', \
			help="Path of the pickle file containing the data. If none (default), the data will not be pickled")	
	parser.add_argument("--N", dest="N", default=-1, \
			help="Number of instances used for training in each LOPO iteration")
	parser.add_argument("--M", dest="M", default=-1, \
			help="Number of single-instance bags used for training in each LOPO iteration")
	parser.add_argument("--clf", dest="clf_name", default='SIL', \
			help="Classifier ('RF', 'SVM', 'LinearSVC', 'SIL', 'LinearSIL', 'MIForest', 'sMIL', 'sbMIL', 'misvm')")
	parser.add_argument("--n-jobs", dest="n_jobs", default=1, type=int, \
			help="Number of threads used (default = 1). Use -1 for maximal parallelization")
	parser.add_argument("--desc", dest="description", default='SIL based on Linear SVC implementation', type=str, \
			help="Description of the evaluation and parameter selection")
	parser.add_argument("--eta", dest="eta", default=0.5, type=float, \
			help="Balancing parameter for sbMIL, between 0 and 1 inclusively")	
	parser.add_argument("--cv", dest="cv", default=3, type=int, \
			help="Determines split for cross-validation (see GridSearchCV.cv)")
	parser.add_argument("--niter", dest="n_iter", default=10, type=int, \
			help="The number of iterations in randomized cross-validation (see RandomizedSearchCV.cv)")
	parser.add_argument("--cv-method", dest="cv_method", default='randomized', type=str, \
			help="Determines how hyperparameters are learned ('grid' or 'randomized')")
	parser.add_argument("--kernel", dest="kernel", default='linear', type=str, \
			help="Kernel used in SVM / SVM-based MIL algorithm.")
	parser.add_argument("--K", dest="K", default=0, type=int, \
			help="Number of single-instance bags in the training data from the held-out participant.")
	parser.add_argument("--bag-size", dest="bag_size", default=1, type=int, \
			help="Bag size.")
	parser.add_argument("--n-trials", dest="n_trials", default=5, type=int, \
			help="Number of trials over which to average the performance metrics")

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
		
	arg_str = ' --dir=' + str(args.data_dir) + ' --load=' + str(args.load_pickle_path) + ' --save=' + str(args.save_pickle_path) \
			+ ' --K=' + str(args.K) + ' --N=' + str(args.N) + ' --clf="' + str(args.clf_name) + '" --eta=' + str(args.eta) \
			+ ' --n-jobs=' + str(args.n_jobs) + ' --desc="' + str(args.description) + '" --cv=' + str(args.cv) + ' --cv-method="' \
			+ str(args.cv_method) + '" --niter=' + str(args.n_iter) + ' --kernel="' + str(args.kernel) + '" --n-trials=' \
			+ str(args.n_trials) + ' --bag-size=' + str(args.bag_size)
			
	with open(os.path.join(params_dir, 'params.txt'), 'wb') as f:
		f.write(arg_str)

	for p in participants:
		for m in M:
			save_path = os.path.join(res_dir, 'lopo_p%d_m%d.pickle' % (p, m))
			submit_this_job = ('python %s/w_lopo.py --save_path=%s --test-participant=%d --M=%d' % (args.src, save_path, p, m)) + arg_str
			print submit_this_job
			job_id = 'lopo_p%d_m%d' % (p, m)
			log_file = log_dir + '/lopo_p%d_m%dlog.txt' % (p, m)
			err_file = err_dir + '/lopo_p%d_m%derr.txt' % (p, m)
			qsub(submit_this_job, job_id, log_file, err_file, n_cores=args.n_jobs)

if __name__ == '__main__':
	main()

