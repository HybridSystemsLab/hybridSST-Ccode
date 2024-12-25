/*********************************************************************
* Software License Agreement (BSD License)
*
*  Copyright (c) 2024, University of Santa Cruz Hybrid Systems Laboratory
*  All rights reserved.
*
*  Redistribution and use in source and binary forms, with or without
*  modification, are permitted provided that the following conditions
*  are met:
*
*   * Redistributions of source code must retain the above copyright
*     notice, this list of conditions and the following disclaimer.
*   * Redistributions in binary form must reproduce the above
*     copyright notice, this list of conditions and the following
*     disclaimer in the documentation and/or other materials provided
*     with the distribution.
*   * Neither the name of Rutgers University nor the names of its
*     contributors may be used to endorse or promote products derived
*     from this software without specific prior written permission.
*
*  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
*  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
*  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
*  FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
*  COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
*  INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
*  BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
*  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
*  CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
*  LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
*  ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
*  POSSIBILITY OF SUCH DAMAGE.
*********************************************************************/

/* Authors: Beverly Xu */
/* Adapted from: ompl/geometric/planners/src/SST.cpp by  Zakary Littlefield of Rutgers the State University of New Jersey, New Brunswick */

#ifndef OMPL_GEOMETRIC_PLANNERS_SST_HySST_
#define OMPL_GEOMETRIC_PLANNERS_SST_HySST_

#include "ompl/geometric/planners/PlannerIncludes.h"
#include "ompl/datastructures/NearestNeighbors.h"
#include "ompl/control/Control.h"

namespace ompl
{
    namespace geometric
    {
        /**
           @anchor gHySST
           @par Hybrid Stable-sparse RRT (HySST) is an asymptotically near-optimal incremental
           sampling-based motion planning algorithm. It is recommended for geometric problems
           to use an alternative method that makes use of a steering function. Using HySST for
           geometric problems does not take advantage of this function.
           @par External documentation
           N. Wang and R. G. Sanfelice, "HySST: An Asymptotically Near-Optimal Motion Planning Algorithm for Hybrid Systems."
           [[PDF]](https://arxiv.org/pdf/2305.18649)
        */
        class HySST : public base::Planner
        {
        public:
            /** \brief Constructor */
            HySST(const base::SpaceInformationPtr &si);

            /** \brief Destructor */
            ~HySST() override;

            /** \brief Set the problem instance to solve */
            void setup() override;

            /** \brief Representation of a motion
             * @par This only contains pointers to parent motions as we
             * only need to go backwards in the tree. 
             */
            class Motion
            {
            public:
                /// \brief Default constructor
                Motion() = default;

                /// \brief Constructor that allocates memory for the state
                Motion(const base::SpaceInformationPtr &si) : state(si->allocState()) {}

                /// \brief Destructor
                virtual ~Motion() = default;

                /** \brief Get the state contained by the motion */
                virtual base::State *getState() const
                {
                    return state;
                }

                /** \brief Get the parent motion in the tree */
                virtual Motion *getParent() const
                {
                    return parent;
                }
                
                /// \brief The total cost accumulated from the root to this vertex
                base::Cost accCost_{0.};

                /// \brief The state contained by the motion
                base::State *state{nullptr};

                /// \brief The parent motion in the exploration tree
                Motion *parent{nullptr};

                /// \brief Number of children. Starting with 0.
                unsigned numChildren_{0};

                /// \brief If inactive, this node is not considered for selection.
                bool inactive_{false};

                /// \brief The integration steps defining the edge of the motion, between the parent and child vertices
                std::vector<base::State *> *solutionPair{nullptr};

                /// \brief The inputs associated with the solution pair
                std::vector<ompl::control::Control *> *inputs = new std::vector<ompl::control::Control *>();

                /// \brief The hybrid time parameterizing each state in the solution pair
                std::vector<std::pair<double, int>> *hybridTime = new std::vector<std::pair<double, int>>();
            };

            /** 
             * \brief Main solve function. 
             * @par Continue solving for some amount of time.
             * @param ptc The condition to terminate solving.
             * @return true if solution was found. */
            base::PlannerStatus solve(const base::PlannerTerminationCondition &ptc) override;

            /** 
             * \brief Get the PlannerData object associated with this planner
             * @param data the PlannerData object storing the edges and vertices of the solution
             */
            void getPlannerData(base::PlannerData &data) const override;

            /** \brief Clear all allocated memory. */
            void clear() override;

            /**
             * \brief Set the radius for selecting nodes relative to random sample.
             * @par This radius is used to mimic behavior of RRT* in that it promotes
             * extending from nodes with good path cost from the root of the tree.
             * Making this radius larger will provide higher quality paths, but has two
             * major drawbacks; exploration will occur much more slowly and exploration
             * around the boundary of the state space may become impossible. 
             * @param selectionRadius The maximum distance from the random sampled vertex, for a vertex in the search tree to be considered
             */
            void setSelectionRadius(double selectionRadius)
            {
                if (selectionRadius < 0)
                    throw Exception("Selection radius must be positive");
                selectionRadius_ = selectionRadius;
            }

            /** 
             * \brief Get the selection radius the planner is using 
             * @return The selection radius
             */
            double getSelectionRadius() const
            {
                return selectionRadius_;
            }

            /**
             * \brief Set the radius for pruning nodes.
             * @par This is the radius used to surround nodes in the witness set.
             * Within this radius around a state in the witness set, only one
             * active tree node can exist. This limits the size of the tree and
             * forces computation to focus on low path costs nodes. If this value
             * is too large, narrow passages will be impossible to traverse. In addition,
             * children nodes may be removed if they are not at least this distance away
             * from their parent nodes.
             * @param pruningRadius The radius used to prune vertices in the search tree
             */
            void setPruningRadius(double pruningRadius)
            {
                if (pruningRadius < 0)
                    throw ompl::Exception("Pruning radius must be non-negative");
                pruningRadius_ = pruningRadius;
            }

            /** 
             * \brief Get the pruning radius the planner is using 
             * @return The pruning radius
             */
            double getPruningRadius() const
            {
                return pruningRadius_;
            }

            /** \brief Set a different nearest neighbors datastructure */
            template <template <typename T> class NN>
            void setNearestNeighbors()
            {
                if (nn_ && nn_->size() != 0)
                    OMPL_WARN("Calling setNearestNeighbors will clear all states.");
                clear();
                nn_ = std::make_shared<NN<Motion *>>();
                witnesses_ = std::make_shared<NN<Motion *>>();
                setup();
            }

            /**
             * \brief Set of available random input methods
             * @par See https://ompl.kavrakilab.org/classompl_1_1RNG.html for more information about each distribution
             */
            enum inputSamplingMethods_
            {
                UNIFORM_01,
                UNIFORM_REAL,
                UNIFORM_INT,
                GAUSSIAN_01,
                GAUSSIAN_REAL,
                HALF_NORMAL_REAL,
                HALF_NORMAL_INT,
                QUATERNION,
                EURLER_RPY
            };

            /** 
             * \brief Define the continuous interval over which the flow input is sampled. 
             * @param min the minimum input values
             * @param max the maximum input values
             */
            void setFlowInputRange(std::vector<double> min, std::vector<double> max)
            {
                int size = max.size() > min.size() ? max.size() : min.size();
                if (min.size() != max.size())
                    throw Exception("Max input value (maxFlowInputValue) and min input value (minFlowInputValue) must be of the same size");
                for (int i = 0; i < size; i++)
                {
                    if (min.at(0) > max.at(0))
                        throw Exception("Max input value must be greater than or equal to min input value");
                }
                minFlowInputValue_ = min;
                maxFlowInputValue_ = max;
            }

            /** 
             * \brief Define the continuous interval over which the flow input is sampled. 
             * @param min the minimum input values
             * @param max the maximum input values
             */
            void setJumpInputRange(std::vector<double> min, std::vector<double> max)
            {
                int size = max.size() > min.size() ? max.size() : min.size();
                if (min.size() != max.size())
                    throw Exception("Max input value (maxJumpInputValue_) and min input value (minJumpInputValue) must be of the same size");
                for (int i = 0; i < size; i++)
                {
                    if (min.at(i) > max.at(i))
                        throw Exception("Max input value must be greater than or equal to min input value");
                }
                minJumpInputValue_ = min;
                maxJumpInputValue_ = max;
            }

            /** 
             * \brief Set the maximum time allocated to a full continuous simulator step.
             * @param tM the maximum time allocated. Must be greater than 0, and greater than than the time
             * allocated to a single continuous simulator call. 
             */
            void setTm(double tM)
            {
                if (tM <= 0)
                    throw Exception("Maximum flow time per propagation step must be greater than 0");
                if (!flowStepDuration_)
                {
                    if (tM < flowStepDuration_)
                        throw Exception("Maximum flow time per propagation step must be greater than or equal to the length of time for each flow "
                                        "integration step (flowStepDuration_)");
                }
                tM_ = tM;
            }

            /** 
             * \brief Set the time allocated to a single continuous simulator call, within the full period of a continuous simulator step. 
             * @param duration the time allocated per simulator step. Must be
             * greater than 0 and less than the time allocated to a full continuous
             * simulator step. 
             */
            void setFlowStepDuration(double duration)
            {
                if (duration <= 0)
                    throw Exception("Flow step length must be greater than 0");
                if (!tM_)
                {
                    if (tM_ < duration)
                        throw Exception("Flow step length must be less than or equal to the maximum flow time per propagation step (Tm)");
                }
                flowStepDuration_ = duration;
            }

            /** 
             * \brief Define the jump set
             * @param jumpSet the jump set associated with the hybrid system. 
             */
            void setGoalTolerance(double tolerance)
            {
                if (tolerance < 0)
                    throw Exception("Goal tolerance must be greater than or equal to 0");
                tolerance_ = tolerance;
            }

            /** 
             * \brief Define the jump set
             * @param jumpSet the jump set associated with the hybrid system. 
             */
            void setJumpSet(std::function<bool(Motion *)> jumpSet)
            {
                jumpSet_ = jumpSet;
            }

            /** 
             * \brief Define the flow set
             * @param flowSet the flow set associated with the hybrid system.
             */
            void setFlowSet(std::function<bool(Motion *)> flowSet)
            {
                flowSet_ = flowSet;
            }

            /** 
             * \brief Define the unsafe set
             * @param unsafeSet the unsafe set associated with the hybrid system.
             */
            void setUnsafeSet(std::function<bool(Motion *)> unsafeSet)
            {
                unsafeSet_ = unsafeSet;
            }

            /** 
             * \brief Define the distance measurement function
             * @param function the distance function associated with the motion planning problem.
             */
            void setDistanceFunction(std::function<double(base::State *, base::State *)> function)
            {
                distanceFunc_ = function;
            }

            /** 
             * \brief Define the discrete dynamics simulator
             * @param function the discrete simulator associated with the hybrid system.
             */
            void setDiscreteSimulator(std::function<base::State *(base::State *curState, std::vector<double> u, base::State *newState)> function)
            {
                discreteSimulator_ = function;
            }

            /** 
             * \brief Define the continuous dynamics simulator
             * @param function the continuous simulator associated with the hybrid system.
             */
            void setContinuousSimulator(std::function<base::State *(std::vector<double> inputs, base::State *curState, double tFlowMax, 
                                        base::State *newState)> function)
            {
                continuousSimulator_ = function;
            }

            /** 
             * \brief Define the collision checker
             * @param function the collision checker associated with the state space. Default is a point-by-point collision checker.
             */
            void setCollisionChecker(std::function<bool(Motion *motion, std::function<bool(Motion *motion)> obstacleSet, 
                                     double ts, double tf, base::State *newState, double *collisionTime)> function)
            {
                collisionChecker_ = function;
            }

            /** 
             * \brief Set solution batch size 
             * @param batchSize the number of solutions to be generated by the planner until the one with the best cost is returned.
             */
            void setBatchSize(int batchSize) {
                if(batchSize < 1)
                    throw Exception("Batch size must be greater than 0");
                batchSize_ = batchSize;
            }

            /** 
             * \brief Set the flow input sampling mode. 
             * @par See https://github.com/ompl/ompl/blob/main/src/ompl/util/RandomNumbers.h for details on each available mode.
             * @param mode the sampling mode.
             * @param inputs the required parameters for the given sampling mode. Not all modes require parameters.
             */
            void setFlowInputSamplingMode(inputSamplingMethods_ mode, std::vector<double> inputs)
            {
                inputSamplingMethod_ = mode;

                unsigned int targetParameterCount = 0;

                switch (mode)
                {
                case UNIFORM_INT:
                    targetParameterCount = 2;
                    getRandFlowInput_ = [this](int i) { return randomSampler_->uniformInt(minFlowInputValue_[i], maxFlowInputValue_[i]); };
                    break;
                case GAUSSIAN_REAL:
                    targetParameterCount = 2;
                    getRandFlowInput_ = [&](int i) { return randomSampler_->gaussian(inputs[0], inputs[1]); };
                    break;
                case HALF_NORMAL_REAL:
                    targetParameterCount = 2; // Can also be three, if want to specify focus,
                                              // which defaults to 3.0
                    getRandFlowInput_ = [&](int i) { return randomSampler_->halfNormalReal(inputs[0], inputs[1], inputs[2]); };
                    break;
                default:
                    targetParameterCount = 2;
                    getRandFlowInput_ = [this](int i) { return randomSampler_->uniformReal(minFlowInputValue_[i], maxFlowInputValue_[i]); };
                }

                if (inputs.size() == targetParameterCount || (mode == HALF_NORMAL_INT && targetParameterCount == 3))
                    inputSamplingParameters_ = inputs;
                else
                    throw Exception("Invalid number of input parameters for input sampling mode.");
            }

            /** 
             * \brief Set the jump input sampling mode. 
             * @par See https://github.com/ompl/ompl/blob/main/src/ompl/util/RandomNumbers.h for details on each available mode.
             * @param mode the sampling mode.
             * @param inputs the required parameters for the given sampling mode. Not all modes require parameters.
             */
            void setJumpInputSamplingMode(inputSamplingMethods_ mode, std::vector<double> inputs)
            {
                inputSamplingMethod_ = mode;

                unsigned int targetParameterCount = 0;

                switch (mode)
                {
                case UNIFORM_INT:
                    targetParameterCount = 2;
                    getRandJumpInput_ = [this](int i) { return randomSampler_->uniformInt(minJumpInputValue_[i], maxJumpInputValue_[i]); };
                    break;
                case GAUSSIAN_REAL:
                    targetParameterCount = 2;
                    getRandJumpInput_ = [&](int i) { return randomSampler_->gaussian(inputs[0], inputs[1]); };
                    break;
                case HALF_NORMAL_REAL:
                    targetParameterCount = 2; // Can also be three parameters, if want to specify focus,
                                              // which defaults to 3.0
                    getRandJumpInput_ = [&](int i) { return randomSampler_->halfNormalReal(inputs[0], inputs[1], inputs[2]); };
                    break;
                default:
                    targetParameterCount = 2;
                    getRandJumpInput_ = [this](int i) { return randomSampler_->uniformReal(minJumpInputValue_[i], maxJumpInputValue_[i]); };
                }

                if (inputs.size() == targetParameterCount || (mode == HALF_NORMAL_INT && targetParameterCount == 3))
                    inputSamplingParameters_ = inputs;
                else
                    throw Exception("Invalid number of input parameters for input sampling mode.");
            }

            /** \brief Check if all required parameters have been set. */
            void checkMandatoryParametersSet(void) const
            {
                if (!discreteSimulator_)
                    throw Exception("Jump map not set");
                if (!continuousSimulator_)
                    throw Exception("Flow map not set");
                if (!flowSet_)
                    throw Exception("Flow set not set");
                if (!jumpSet_)
                    throw Exception("Jump set not set");
                if (!unsafeSet_)
                    throw Exception("Unsafe set not set");
                if (tM_ < 0.0)
                    throw Exception("Max flow propagation time (Tm) not set");
                if (maxJumpInputValue_.size() == 0)
                    throw Exception("Max input value (maxJumpInputValue) not set");
                if (minJumpInputValue_.size() == 0)
                    throw Exception("Min input value (minJumpInputValue) not set");
                if (maxFlowInputValue_.size() == 0)
                    throw Exception("Max input value (maxFlowInputValue) not set");
                if (minFlowInputValue_.size() == 0)
                    throw Exception("Min input value (minFlowInputValue) not set");
                if (!flowStepDuration_)
                    throw Exception("Flow step length (flowStepDuration_) not set");
                if (pruningRadius_ < 0)
                    throw Exception("Pruning radius (pruningRadius_) not set");
                if (selectionRadius_ < 0)
                    throw Exception("Selection radius (selectionRadius_) not set");
            }

        protected:
            /// \brief Representation of a witness vertex in the search tree
            class Witness : public Motion
            {
            public:
                /// \brief Default Constructor
                Witness() = default;

                /// \brief Constructor that allocates memory for the state
                Witness(const base::SpaceInformationPtr &si) : Motion(si) {}

                /** 
                 * \brief Get the state contained by the representative motion 
                 * @return The state contained by the representative motion
                 */
                base::State *getState() const override
                {
                    return rep_->state;
                }

                /** 
                 * \brief Get the state contained by the parent motion of the representative motion 
                 * @return The state contained by the parent motion of the representative motion
                 */
                Motion *getParent() const override
                {
                    return rep_->parent;
                }

                /** 
                 * \brief Set the representative of the witness
                 * \param lRep The representative motion
                 */
                void linkRep(Motion *lRep)
                {
                    rep_ = lRep;
                }

                /** \brief The node in the tree that is within the pruning radius.*/
                Motion *rep_{nullptr};
            };

            /** 
             * \brief Random sampler for the full vector of flow input. 
             * @return a vector of inputs (as doubles) sampled
             */
            std::function<std::vector<double>(void)> sampleFlowInputs_ = [this](void)
            {
                std::vector<double> u;
                for (unsigned int i = 0; i < maxFlowInputValue_.size(); i++)
                    u.push_back(getRandFlowInput_(i));
                return u;
            };

            /** 
             * \brief Random sampler for the full vector of jump input. 
             * @return a vector of inputs (as doubles) sampled
             */
            std::function<std::vector<double>(void)> sampleJumpInputs_ = [this](void)
            {
                std::vector<double> u;
                for (unsigned int i = 0; i < maxJumpInputValue_.size(); i++)
                    u.push_back(getRandJumpInput_(i));
                return u;
            };

            /** 
             * \brief Random sampler for one value of flow input. 
             * @return a single, double-valued sampled input
             */
            std::function<double(int i)> getRandFlowInput_ = [this](int i) { return randomSampler_->uniformReal(minFlowInputValue_[i], maxFlowInputValue_[i]); };

            /** 
             * \brief Random sampler for one value of jump input. 
             * @return a single, double-valued sampled input
             */
            std::function<double(int i)> getRandJumpInput_ = [this](int i) { return randomSampler_->uniformReal(minJumpInputValue_[i], maxJumpInputValue_[i]); };

            /// \brief The most recent goal motion.  Used for PlannerData computation
            Motion *lastGoalMotion_{nullptr};

            /** \brief Runs the initial setup tasks for the tree. */
            void initTree(void);

            /** 
             * \brief Sample the random motion. 
             * @param randomMotion The motion to be initialized
             */
            void randomSample(Motion *randomMotion);

            /// \brief A nearest-neighbors datastructure containing the tree of motions
            std::shared_ptr<NearestNeighbors<Motion *>> nn_;

            /**
             * The following are all customizeable parameters,
             * and affect how @b cHySST generates trajectories.
             * Customize using setter functions above. */

            /**
             * \brief Collision checker. Default is point-by-point collision checking using the jump set.
             * @param motion The motion to check for collision
             * @param obstacleSet A function that returns true if the motion's solution pair intersects with the obstacle set
             * @param ts The start time of the motion. Default is -1.0
             * @param tf The end time of the motion. Default is -1.0
             * @param newState The collision state (if a collision occurs)
             * @param collisionTime The time of collision (if a collision occurs). If no collision occurs, this value is -1.0
             * @return true if a collision occurs, false otherwise
             */
            std::function<bool(Motion *motion,
                               std::function<bool(Motion *motion)> obstacleSet,
                               double ts, double tf, base::State *newState, double *collisionTime)>
                collisionChecker_ =
                    [this](Motion *motion,
                           std::function<bool(Motion *motion)> obstacleSet, double t = -1.0,
                           double tf = -1.0, base::State *newState, double *collisionTime) -> bool
            {
                if (obstacleSet(motion))
                    return true;
                return false;
            };

            /// \brief Name of input sampling method, default is "uniform"
            inputSamplingMethods_ inputSamplingMethod_{UNIFORM_01};

            /** 
             * \brief Compute distance between states, default is Euclidean distance 
             * @param state1 The first state
             * @param state2 The second state
             * @return The distance between the two states
             */
            std::function<double(base::State *state1, base::State *state2)> distanceFunc_ = [this](base::State *state1, base::State *state2) -> double
            {
                return si_->distance(state1, state2);
            };

            /// \brief The maximum flow time for a given flow propagation step. Must be set by the user.
            double tM_{-1.};

            /// \brief The distance tolerance from the goal state for a state to be regarded as a valid final state. Default is .1
            double tolerance_{.1};

            /// \brief The minimum step length for a given flow propagation step. Default value is 1e-6
            double minStepLength = 1e-06;

            /// \brief The flow time for a given integration step, within a flow propagation step. Must be set by user.
            double flowStepDuration_;

            /// \brief Minimum flow input values
            std::vector<double> minFlowInputValue_;

            /// \brief Maximum flow input values
            std::vector<double> maxFlowInputValue_;

            /** \brief Minimum jump input value */
            std::vector<double> minJumpInputValue_;

            /** \brief Maximum jump input value */
            std::vector<double> maxJumpInputValue_;

            /** 
             * \brief Simulator for propagation under jump regime
             * @param curState The current state
             * @param u The input
             * @param newState The newly propagated state
             * @return The newly propagated state
             */
            std::function<base::State *(base::State *curState, std::vector<double> u, base::State *newState)> discreteSimulator_;

            /** 
             * \brief Function that returns true if a motion intersects with the jump set, and false if not. 
             * @param motion The motion to check
             * @return True if the state is in the jump set, false if not
             */
            std::function<bool(Motion *motion)> jumpSet_;

            /** 
             * \brief Function that returns true if a motion intersects with the flow set, and false if not. 
             * @param motion The motion to check
             * @return True if the state is in the flow set, false if not
             */
            std::function<bool(Motion *motion)> flowSet_;

            /** 
             * \brief Function that returns true if a motion intersects with the unsafe set, and false if not. 
             * @param motion The motion to check
             * @return True if the state is in the unsafe set, false if not
             */
            std::function<bool(Motion *motion)> unsafeSet_;

            /// \brief The optimization objective. Default is a shortest path objective.
            base::OptimizationObjectivePtr opt_;

            /** \brief Calculate the cost of a motion. Default is using optimization objective. */
            std::function<base::Cost (Motion *motion)> costFunc_;

            /** 
             * \brief Simulator for propagation under flow regime
             * @param input The input
             * @param curState The current state
             * @param tFlowMax The random maximum flow time
             * @param newState The newly propagated state
             * @return The newly propagated state
             */
            std::function<base::State *(std::vector<double> input, base::State *curState, double tFlowMax, base::State *newState)> continuousSimulator_;

            /// \brief Random sampler for the input. Default constructor always seeds a different value, and returns a uniform real distribution.
            RNG *randomSampler_ = new RNG();

            /** 
             * \brief Construct the path, starting at the last edge. 
             * @param lastMotion The last motion in the solution
             * @return the planner status (APPROXIMATE, EXACT, or UNKNOWN)
             */
            base::PlannerStatus constructSolution(Motion *lastMotion);

            /// \brief Input sampling parameters
            std::vector<double> inputSamplingParameters_{};

            /** 
             * \brief Finds the best node in the tree withing the selection radius around a random sample.
             * @param sample The random sampled vertex to find the closest witness to.
             */
            Motion *selectNode(Motion *sample);

            /** 
             * \brief Find the closest witness node to a newly generated potential node.
             * @param node The vertex to find the closest witness to.
             */
            Witness *findClosestWitness(Motion *node);

            /** 
             * \brief Randomly propagate a new edge.
             * @param m The motion to extend
             * @param goal The goal state (to check if within tolerance of goal state)
             */
            std::vector<Motion *> extend(Motion *m, base::Goal *goalState);

            /** \brief Free the memory allocated by this planner */
            void freeMemory();

            /// \brief State sampler
            base::StateSamplerPtr sampler_;

            /// \brief A nearest-neighbors datastructure containing the tree of witness motions
            std::shared_ptr<NearestNeighbors<Motion *>> witnesses_;

            /// \brief The radius for determining the node selected for extension. Delta_s.
            double selectionRadius_{-1.};   

            /// \brief The radius for determining the size of the pruning region. Delta_bn.
            double pruningRadius_{-1.};

            /// \brief The random number generator
            RNG rng_;

            /// \brief Minimum distance from goal to final vertex of generated trajectories.
            double dist_;

            /// \brief The best solution (with best cost) we have found so far.
            std::vector<Motion *> prevSolution_;

            /// \brief The best solution cost we have found so far.
            base::Cost prevSolutionCost_{std::numeric_limits<double>::quiet_NaN()};

            /// \brief The number of solutions allowed until the most optimal solution is returned.
            int batchSize_{1};
        };
    }
}

#endif
