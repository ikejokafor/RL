#ifndef __RL_HPP__
#define __RL_HPP__


#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <float.h>
#include <random>
#include <vector>
#include <ctime>
#include <iostream>


#include "util.hpp"

class QLearner {
    public:
	QLearner(int numStates, int numActionsPerState, bool *validActionsPerState, int *transitionMatrix, float discountFactor = 0.9f, int startState = -1, float epsilon = 0.2f);
		~QLearner();
		int GetNextAction();
		int GetBestAction(int state);
		void UpdateQTable(float reward);
		void printQMatrix();
        
        float *m_QMatrix;
        float m_discountFactor;
		float m_learningRate;
		float m_epsilon;
		float m_epsilonDecay;
		int m_currentState;
		int m_numStates;
		int m_numActionsPerState;
		int m_currentAction;
		bool *m_validActionsPerState;
		int *m_transitionMatrix;
		std::default_random_engine *m_gen;
		std::uniform_real_distribution<float> *m_dis;
	
    protected:
    
    private:


};

#endif