#include "RL.hpp"
using namespace std;

QLearner::QLearner(int numStates, int numActionsPerState, bool *validActionsPerState, int *transitionMatrix, float discountFactor, int startState, float epsilon) {
	m_numStates = numStates;
	m_numActionsPerState = numActionsPerState;
	m_validActionsPerState = (bool*)malloc(numStates * numActionsPerState * sizeof(bool));
	memcpy(m_validActionsPerState, validActionsPerState, (numStates * numActionsPerState * sizeof(bool)));
	m_discountFactor = discountFactor;
	m_epsilon = epsilon;
	m_QMatrix = (float*)malloc(numStates * numActionsPerState * sizeof(float));
	for (int i = 0; i < (numStates * numActionsPerState); i++){
		m_QMatrix[i] = 1000.0f;
	}
    cout << "......Initial Q-Matrix......" << endl;
	printQMatrix();
	
	m_transitionMatrix = (int*)malloc(numStates * numActionsPerState * sizeof(int));
	memcpy(m_transitionMatrix, transitionMatrix, (numStates * numActionsPerState * sizeof(int)));
	
	if(startState == -1) {
		m_currentState = 0;
	} else {
		m_currentState = startState;
	}
	

	m_gen = new default_random_engine();
	m_dis = new uniform_real_distribution<float>(0.0f, 1.0f);
}


QLearner::~QLearner() {
    free(m_QMatrix);
	free(m_validActionsPerState);
}


int QLearner::GetNextAction() {
	if((m_dis[0](m_gen[0])) <= m_epsilon) {
		srand(time(NULL));
		while(true) {
			m_currentAction = rand() % m_numActionsPerState;
			if (index2D(m_numStates, m_numActionsPerState, m_validActionsPerState, m_currentState, m_currentAction)) {
				//m_epsilon -= 0.001f;
				return m_currentAction;
			}
		}
	} else {
		m_currentAction = GetBestAction(m_currentState);
		//m_epsilon -= 0.001f;
		return m_currentAction;
	}
}


int QLearner::GetBestAction(int state) {
	vector<int> actionList;
	int bestAction;
	for (int action = 0; action < m_numActionsPerState; action++) {
		if (index2D(m_numStates, m_numActionsPerState, m_validActionsPerState, state, action)) {
			if (actionList.size() == 0) {
				actionList.push_back(action);
			} else if(index2D(m_numStates, m_numActionsPerState, m_QMatrix, state, action) == index2D(m_numStates, m_numActionsPerState, m_QMatrix, state, actionList[0])) {
				actionList.push_back(action);
			} else if(index2D(m_numStates, m_numActionsPerState, m_QMatrix, state, action) > index2D(m_numStates, m_numActionsPerState, m_QMatrix, state, actionList[0])) {
				actionList.clear();
				actionList.push_back(action);
			}
		}
	}
	
	if (actionList.size() == 1) {
		bestAction = actionList[0];
		return bestAction;
	} else {
		srand(time(NULL));
		bestAction = actionList[rand() % actionList.size()];
		return bestAction;
	}
}


void QLearner::UpdateQTable(float reward) {
	int nextState = index2D(m_numStates, m_numActionsPerState, m_transitionMatrix, m_currentState, m_currentAction);
	int bestAction = GetBestAction(nextState);
	index2D(m_numStates, m_numActionsPerState, m_QMatrix, m_currentState, m_currentAction)
		= (reward + m_discountFactor * index2D(m_numState, m_numActionsPerState, m_QMatrix, nextState, bestAction));
	m_currentState = nextState;
}

void QLearner::printQMatrix() {
	for (int i = 0; i < m_numStates; i++) {
		for (int j = 0; j < m_numActionsPerState; j++) {
			cout << index2D(agent->m_numStates, m_numActionsPerState, m_QMatrix, i, j) << " ";
		}
		cout << endl;
	}
	cout << endl;
}
