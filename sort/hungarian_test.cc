#include <iostream>
#include "hungarian.hpp"

using namespace sort;

int main(void)
{
	vector< vector<double> > costMatrix = { { 10, 1.9, 8, 1.5, 0 }, 
										  { 1.0, 1.8, 7, 1.7, 0 }, 
										  { 1.3, 1.6, 9, 1.4, 0 }, 
										  { 1.2, 1.9, 0.8, 1.8, 0 } };

	HungarianAlgorithm HungAlgo;
	vector<int> assignment;

	double cost = HungAlgo.Solve(costMatrix, assignment);

	for (unsigned int x = 0; x < costMatrix.size(); x++)
		std::cout << x << ":" << assignment[x] << "\t";

	std::cout << "\ncost: " << cost << std::endl;

	return 0;
}
