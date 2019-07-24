#pragma once
#include <vector>
#include <fstream>
#include <string>
#include <iomanip>
#include "data.h"
#include "neuron.h"

class Net
{
public:
	Net(const Topology& topology);
	void predict(const Data& input);
	void fit(const Data& target);
	void fit(const Data& input, const Data& target);
	void getRecentResults(Data& result) const;
	double getRecentAvgError() const { return recentAvgError; }
	unsigned setDump(std::string name);
	unsigned getDump(std::string name);
private:
	std::vector<Layer> layers; // layers[layerNumber][NeuronNumber]
	double error;
	double recentAvgError;
	static double recentAvgSmoothingFactor;
};
