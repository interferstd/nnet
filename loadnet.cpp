#include "net.h"

int main(int argc, char ** argv)
{
	assert(argc == 2);
	
	Net net({});
	net.getDump(argv[1]);
	
	Neuron::setLearnRate(0.005);
	Neuron::activation.setTanh();
	Neuron::setAlpha(0.9);

	std::string in;
	Data data(net.getInputNum()), result;
	while (std::getline(std::cin, in), in != "exit")
	{
		std::stringstream ss(in);
		for(auto& e: data) ss >> e;
		net.predict(data);
		net.getRecentResults(result);
		for(auto& e: result) std::cout << e << ' ';
		std::cout << '\n';
	}
	
	return 0;
}
