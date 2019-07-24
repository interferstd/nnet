#include "trainingdata.h"
#include "net.h"

void showVector(std::string label, std::vector<double> &v)
{
	std::cout << label << " ";
	for (unsigned i = 0; i < v.size(); i++)
	{
		std::cout << v[i] << " ";
	}
	std::cout << '\n';
}

int main()
{
	TrainingData trainingData("testData.txt");
	Topology topology;
	trainingData.getTopology(topology);
	TrainList trainList(trainingData);
	Net net(topology);
	
	trainList.repeatData([&](TrainEpoch epoch) { net.fit(epoch.input, epoch.target); }, 10);

//////////////////////////////////////////////////////////////////////////////////////////////////
//	Data input, target, result;
//
//	int trainingPass = 0;
//	while (trainingData.getInput(input), trainingData.getTarget(target), !trainingData.isEOF())
//	//while (trainingData.getInput(input), !trainingData.isEOF())
//	{
//		++trainingPass;
//
//		assert(input.size() == topology[0]);
//		assert(target.size() == topology.back());
//
//		net.fit(input, target);
/////////////////////////////////////////////////////////////////////////////////////////////////
//		++trainingPass;
//		//std::cout << '\n' << "Pass #" << trainingPass << '\n';
//
//		assert(input.size() == topology[0]);
//		//showVector("  Input:   ", input);
//		net.predict(input);
//
//		trainingData.getTarget(target);
//		//showVector("  Targets: ", target);
//
//		assert(target.size() == topology.back());
//		//net.getRecentResults(result);
//		//showVector("  Outputs: ", result);
//
//		net.fit(target);
//
//		//std::cout << "Avg error: " << net.getRecentAvgError() << '\n';
/////////////////////////////////////////////////////////////////////////////////////////////////
//	}
/////////////////////////////////////////////////////////////////////////////////////////////////

	std::cout << "Avg error: " << net.getRecentAvgError() << '\n';

#if defined(_MSC_VER) || defined(_WIN32)
	//system("PAUSE");
#endif

	return(0);
}
