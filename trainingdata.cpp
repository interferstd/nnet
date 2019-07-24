#include "trainingdata.h"

TrainingData::TrainingData(const std::string filename)
{
	trainingDataFile.open(filename.c_str());
}

void TrainingData::getTopology(Topology& topology)
{
	std::string line;
	std::string label;

	std::getline(trainingDataFile, line);
	std::stringstream ss(line);
	ss >> label;
	if (!this->isEOF() && label.compare("topology:") == 0)
	{
		unsigned n;
		do { ss >> n; topology.push_back(n); } while (!ss.eof());
	}
	else assert(!this->isEOF() && label.compare("topology:") == 0);
}

unsigned TrainingData::getInput(Data& input)
{
	input.clear();

	std::string line;
	std::getline(trainingDataFile, line);
	std::stringstream ss(line);

	std::string label;
	ss >> label;
	if (label.compare("in:") == 0)
	{
		double value;
		while (ss >> value)	input.push_back(value);
	}

	return input.size();
}

unsigned TrainingData::getTarget(Data& target)
{
	target.clear();

	std::string line;
	std::getline(trainingDataFile, line);
	std::stringstream ss(line);

	std::string label;
	ss >> label;
	if (label.compare("out:") == 0)
	{
		double value;
		while (ss >> value) target.push_back(value);
	}

	return target.size();
}

void TrainList::colectData(TrainingData& trainingData)
{
	TrainEpoch epoch;
	while (trainingData.getInput(epoch.input), trainingData.getTarget(epoch.target), !trainingData.isEOF())
	{
		epoches.push_back(epoch);
	}
}

void TrainList::repeatData(TrainFunction trainFunction, unsigned n)
{
	for (unsigned it = 0; it < n; ++it)
	{
		for (auto& epoch: epoches)
		{
			trainFunction(epoch);
		}
	}
}
