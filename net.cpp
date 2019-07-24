#include <cmath>
#include <cassert>
#include "neuron.h"
#include "net.h"


double Net::recentAvgSmoothingFactor = 100.0; // Number of training samples to average over

Net::Net(const Topology& topology)
{
	unsigned numLayers = topology.size();
	layers.reserve(numLayers);
	
	for (unsigned layerIndex = 0; layerIndex < numLayers; ++layerIndex)
	{
		layers.push_back(Layer());
		unsigned numOutputs = layerIndex == topology.size() - 1 ? 0 : topology[layerIndex + 1];
		unsigned numNerons = topology[layerIndex];

		layers.back().reserve(numNerons);
		for (unsigned neuronIndex = 0; neuronIndex <= numNerons; ++neuronIndex)
		{
			layers.back().push_back(Neuron(numOutputs, neuronIndex));
		}

		layers.back().back().setOutput(1.0);
	}
}

void Net::getRecentResults(Data& result) const
{
	result.clear();
	result.reserve(layers.back().size());

	for (unsigned n = 0; n < layers.back().size() - 1; ++n)
	{
		result.push_back(layers.back()[n].getOutput());
	}
}

void Net::fit(const Data& target)
{
	Layer& outputLayer = layers.back();
	error = 0.0;

	for (unsigned n = 0; n < outputLayer.size() - 1; ++n)
	{
		double delta = target[n] - outputLayer[n].getOutput();
		error += delta * delta;
	}
	error /= outputLayer.size() - 1;
	error = ::sqrt(error);

	recentAvgError =
		(recentAvgError * recentAvgSmoothingFactor + error)
		/ (recentAvgSmoothingFactor + 1.0);

	for (unsigned n = 0; n < outputLayer.size() - 1; ++n)
	{
		outputLayer[n].calcOutputGradients(target[n]);
	}

	for (unsigned layerIndex = layers.size() - 2; layerIndex > 0; --layerIndex)
	{
		Layer& hiddenLayer = layers[layerIndex];
		Layer& nextLayer = layers[layerIndex + 1];

		for (auto& hidenNeuron: hiddenLayer) hidenNeuron.calcHiddenGradients(nextLayer);
	}

	for (unsigned layerIndex = layers.size() - 1; layerIndex > 0; --layerIndex)
	{
		Layer& layer = layers[layerIndex];
		Layer& prevLayer = layers[layerIndex - 1];

		for (unsigned n = 0; n < layer.size() - 1; ++n)
		{
			layer[n].updateInputWeights(prevLayer);
		}
	}
}

void Net::predict(const Data& input)
{
	assert(input.size() == layers[0].size() - 1);

	for (unsigned i = 0; i < input.size(); ++i)
	{
		layers[0][i].setOutput(input[i]);
	}

	for (unsigned layerIndex = 1; layerIndex < layers.size(); ++layerIndex)
	{
		Layer& prevLayer = layers[layerIndex - 1];
		for (unsigned n = 0; n < layers[layerIndex].size() - 1; ++n)
		{
			layers[layerIndex][n].predict(prevLayer);
		}
	}
}

void Net::fit(const Data& input, const Data& target)
{
	this->predict(input);
	this->fit(target);
}

unsigned Net::setDump(std::string name)
{
	std::fstream dumpFile(name, std::ios::out);
	if (!dumpFile.is_open()) return false;
	dumpFile.precision(16);
	dumpFile << layers.size() << '\n';
	for (auto& layer: layers)
	{
		dumpFile << layer.size() << '\n';
		for (auto& neuron: layer) neuron.setDump(dumpFile);
	}
	dumpFile.close();
	return true;
}

unsigned Net::getDump(std::string name)
{
	std::fstream loadFile(name, std::ios::in);
	if (!loadFile.is_open()) return false;
	unsigned lenght;
	loadFile >> lenght;
	layers.resize(lenght);
	for (auto& layer: layers)
	{
		loadFile >> lenght;
		layer.resize(lenght, Neuron(0, 0));
		for (auto& neuron: layer) neuron.getDump(loadFile);
	}
	loadFile.close();
	return true;
}
