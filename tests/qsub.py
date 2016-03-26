from subprocess import Popen, PIPE

def qsub(command, job_name=None, stdout=None, stderr=None, depend=None, n_cores=None):
	"""
	depend could be either a string or a list (or tuple, etc.)
	"""
	args = ['qsub']
	if n_cores:
		args.extend(['-pe','generic',"%d"%n_cores])
	if job_name:
		args.extend(['-N', job_name])
	if stderr:
		args.extend(['-e', stderr])
	if stdout:
		args.extend(['-o', stdout])
	if depend:
		# in python3, use isinstance(depend, str) instead.
		if not isinstance(depend, basestring):
			depend = ','.join(depend)
		args.extend(['-hold_jid', depend])
	out = Popen(args, stdin=PIPE, stdout=PIPE).communicate(command + '\n')[0]
	print out.rstrip()
	job_id = out.split()[2]
	return job_id
