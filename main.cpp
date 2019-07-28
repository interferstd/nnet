#include "net.h"
#include <stdlib.h>

//void showVector(std::string label, std::vector<double> &v)
//{
//	std::cout << label << " ";
//	for (auto& e: v) std::cout << e << " ";
//	std::cout << '\n';
//}

int main(int argc, char ** argv)
{
	std::string nameSet = "pirs";
	int epoches = 1E5;
	if (argc > 1) nameSet = argv[1];
	if (argc > 2) epoches = atoi(argv[2]);

	TrainingData trainingData(nameSet);
	Topology topology;
	trainingData.getTopology(topology);
	Net net(topology);
	TrainList trainList(trainingData);

	Neuron::setLearnRate(0.005);
	Neuron::activation.setTanh();
	Neuron::setAlpha(0.9);

	std::cout << "Get dump: [" << net.getDump(nameSet) << "]\n";
	trainList.repeatData([&](TrainEpoch epoch) { net.fit(epoch.input, epoch.target); }, epoches);
	std::cout << "Set dump: [" << net.setDump(nameSet) << "]\n";
	std::cout << "Avg error: " << net.getRecentAvgError() << '\n';

#if defined(_MSC_VER) || defined(_WIN32)
	//system("PAUSE");
#endif

	return(0);
}
