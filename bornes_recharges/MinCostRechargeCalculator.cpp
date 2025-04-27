#include "MinCostRechargeCalculator.h"
#include <vector>
#include <math.h>
#include <iostream>
#include <algorithm> // for std::max

// Nom(s) étudiant(s) / Name(s) of student(s):
// Islam, Hudaa Bint Afzal (20278949)
// 

// ce fichier contient les definitions des methodes de la classe MinCostRechargeCalculator
// this file contains the definitions of the methods of the MinCostRechargeCalculator class

using namespace std;

MinCostRechargeCalculator::MinCostRechargeCalculator()
{
}

int MinCostRechargeCalculator::CalculateMinCostRecharge(const vector<int>& RechargeCost) {
   // Fonction à compléter / function to complete

   if (RechargeCost.size() < 3) {
      return 0;
   }
   
   vector<int> cost = RechargeCost;
   int maxDistance = 3; 
   int minCost = 0;

   for (int i = maxDistance; i < RechargeCost.size(); i++) {
      minCost = min(cost[i-1], cost[i-2]);
      minCost = min(minCost, cost[i-3]);
      cost[i] = cost[i] + minCost;
   }

   int arrival = RechargeCost.size();
   minCost = min(cost[arrival-1], cost[arrival-2]);
   minCost = min(minCost, cost[arrival-3]);

   return minCost;
}
