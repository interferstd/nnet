#include "trainingdata.h"
#include "net.h"

void showVector(std::string label, std::vector<double> &v)
{
	std::cout << label << " ";
	for (auto& e: v) std::cout << e << " ";
	std::cout << '\n';
}

int main()
{
	TrainingData trainingData("testData.txt");
	Topology topology;
	trainingData.getTopology(topology);
	Net net(topology);
	TrainList trainList(trainingData);
	
	std::cout << "Get dump: [" << net.getDump("dump.nnet") << "]\n";
	trainList.repeatData([&](TrainEpoch epoch) { net.fit(epoch.input, epoch.target); }, 1000);
	std::cout << "Set dump: [" << net.setDump("dump.nnet") << "]\n";
	std::cout << "Avg error: " << net.getRecentAvgError() << '\n';

#if defined(_MSC_VER) || defined(_WIN32)
	//system("PAUSE");
#endif

	return(0);
}
