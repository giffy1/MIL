import sys

sys.path.insert(0, '..')
from evaluate_base import BaseEvaluator

def main():	

	base = BaseEvaluator({})
	base.evaluate()

if __name__ == '__main__':
	main()

