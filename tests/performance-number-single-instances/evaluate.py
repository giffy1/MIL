import sys

sys.path.insert(0, '..')
from evaluate_base import BaseEvaluator

def main():	

	base = BaseEvaluator({'M' : range(100,1001,50)})
	base.evaluate()

if __name__ == '__main__':
	main()