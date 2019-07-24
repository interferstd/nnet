#include "trainingSet.h"

TrainingSet::TrainingSet(const std::string filename)
{
	trainingDataFile.open(filename.c_str());
}

void TrainingSet::getTopology(std::vector<unsigned>& topology)
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
	} else abort();
}

unsigned TrainingSet::getInput(std::vector<double>& input)
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

unsigned TrainingSet::getTarget(std::vector<double>& target)
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
