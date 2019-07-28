#include <iostream>
#include <cmath>
#include <cstdlib>


int main()
{
	freopen("sqr.data","w", stdout);
	std::cout << "topology: 1 10 10 10 1" << std::endl;
	for (int i = 10; i > 0; i--)
	{
//		double n1 = (rand() / double(RAND_MAX));
		double n1 = 1.0/i;
//		int n2 = (int)(2.0 * rand() / double(RAND_MAX));
//		int t = !(n1 & n2);

//		std::cout << "in: " << n1 << ".0 " << n2 << ".0" << std::endl;
//		std::cout << "out: " << t << ".0 " << std::endl;
		std::cout << "in: " << n1 << " " << std::endl;
		std::cout << "out: " << n1*n1 << " " << std::endl;
	}
	
	return 0;
}
