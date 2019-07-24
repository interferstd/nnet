#pragma once
#include <vector>
#include <iostream>
#include <cstdlib>
#include <cassert>
#include <cmath>
#include <fstream>
#include <sstream>
#include "data.h"

class TrainingSet
{
public:
	TrainingSet(const std::string filename);
	bool isEOF() { return trainingDataFile.eof(); }
	void getTopology(Topology& topology);
	unsigned getInput(Data& input);
	unsigned getTarget(Data& target);
private:
	std::ifstream trainingDataFile;
};
