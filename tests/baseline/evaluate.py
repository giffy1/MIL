import sys

sys.path.insert(0, '..')
from evaluate_base import BaseEvaluator

def main():	

	base = BaseEvaluator({'N' : range(10,101,10), 'M' : range(100,1001,100), 'bag_size' : [10,25,50,100,200]})
	base.evaluate()

if __name__ == '__main__':
	main()

