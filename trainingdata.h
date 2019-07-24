#pragma once
#include <vector>
#include <iostream>
#include <functional>
#include <cstdlib>
#include <cassert>
#include <cmath>
#include <fstream>
#include <sstream>
#include <list>
#include "data.h"

class TrainingData
{
public:
	TrainingData(const std::string filename);
	bool isEOF() { return trainingDataFile.eof(); }
	void getTopology(Topology& topology);
	unsigned getInput(Data& input);
	unsigned getTarget(Data& target);
private:
	std::ifstream trainingDataFile;
};

typedef struct
{
	Data input;
	Data target;
} TrainEpoch;

typedef std::function<void(const TrainEpoch&)> TrainFunction;

class TrainList
{
public:
	TrainList(): epoches() {}
	TrainList(TrainingData& trainingData) { colectData(trainingData); }
	void colectData(TrainingData& trainingData);
	void clear() { epoches.clear(); }
	void repeatData(TrainFunction trainFunction, unsigned n);
private:
	std::list<TrainEpoch> epoches;
};
