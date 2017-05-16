## Deep Networks for Global Optimization

Paper:

### How to Run:
1. Add known data points in data/X.npy and data/Y.npy
2. Specify function evaluator in evaluate method of data/func.py
3. To run call main.py, specify:
	-i input_dimension (SCALAR)
	-o opt methd
	-ei function

## Running Demo.py

Go to 'dngo/data/func_demo.py' and specify your function. Modify 'data/generator.py' to specify number of points generated
Run 'dngo/data/generate.py', '''python generate.py'''.
Now run 'dngo/demo.py', '''python demo.py -ARGS_HERE'''
	   
### Other Notes

-  print gradients to see if alpha, beta, learning rate are reasonable
- EI performs well when both posterior mean and stdeviation are roughly no more than 1 order of magnitude apart


Test functions used for experimentation taken from:

https://en.wikipedia.org/wiki/Test_functions_for_optimization

http://www.sfu.ca/~ssurjano/optimization.html