#include "../CommonMath/Trajectory.hpp"
#include "../CommonMath/RectPrism.hpp"
#include "../polyFit.h"
#include "../src/quartic.cpp"
#include "../CommonMath/RectPrism.hpp"
#include "../CommonMath/ConvexObj.hpp"
#include "../HySST.h"
#include "ompl/base/GoalTypes.h"
#include "ompl/base/Planner.h"
#include "ompl/base/spaces/RealVectorStateSpace.h"
#include "ompl/control/PathControl.h"
#include <fstream> // For file I/O
#include <iomanip> // For formatting output
#include "ompl/control/SpaceInformation.h"
#include <ompl/control/ODESolver.h>

using namespace CommonMath;

enum CollisionResult
{
    NoCollision = 0,
    Collision = 1,
    CollisionIndeterminable = 2,
};

struct DetailedCollisionResult
{
    CollisionResult collisionType;
    double collisionTime;
};

// Instantiate obstacles
CommonMath::RectPrism leftRect = CommonMath::RectPrism(Vec3(0.25, 2, 0), Vec3(0.5, 2, 2));
CommonMath::RectPrism topRect = CommonMath::RectPrism(Vec3(2.5, 2.75, 0), Vec3(4, 0.5, 2));
CommonMath::RectPrism bottomRect = CommonMath::RectPrism(Vec3(2.5, 1.25, 0), Vec3(4, 0.5, 2));

// Instantiate obstacles as shared pointers
std::shared_ptr<CommonMath::RectPrism> leftRectPrism = std::make_shared<CommonMath::RectPrism>((leftRect));
std::shared_ptr<CommonMath::RectPrism> topRectPrism = std::make_shared<CommonMath::RectPrism>((topRect));
std::shared_ptr<CommonMath::RectPrism> bottomRectPrism = std::make_shared<CommonMath::RectPrism>((bottomRect));

/** \brief Collision Checker for polynomial equations */
DetailedCollisionResult polyCollisionChecker(double ts, double tf, std::shared_ptr<CommonMath::ConvexObj> obstacle, double minTimeSection, unsigned iterationNumber, CommonMath::Trajectory _traj)
{
    DetailedCollisionResult testResult;

    // First find the position halfway between the start and end time of this segment
    double midTime = (ts + tf) / 2.0;
    Vec3 midpoint = _traj.GetValue(midTime);

    if (obstacle->IsPointInside(midpoint))
    {
        if ((tf - ts) <= minTimeSection)
        {
            // If the time section is small enough, consider it a collision
            testResult.collisionType = Collision;
            testResult.collisionTime = ts;
            return testResult;
        }
        else
        {
            // Recursively check both halves of the trajectory
            unsigned nextIterationNumber = iterationNumber + 1;
            DetailedCollisionResult firstHalfResult = polyCollisionChecker(ts, midTime, obstacle, minTimeSection, nextIterationNumber, _traj);
            DetailedCollisionResult secondHalfResult = polyCollisionChecker(midTime, tf, obstacle, minTimeSection, nextIterationNumber, _traj);

            // Merge or prioritize collision results from both halves as needed
            // For simplicity, this example prioritizes collisions in the first half
            if (firstHalfResult.collisionType != NoCollision)
                return firstHalfResult;
            else
                return secondHalfResult;
        }
    }

    if (tf - ts < minTimeSection)
    {
        // Our time resolution is too small, just give up (trajectory is likely tangent to obstacle surface)
        testResult.collisionType = CollisionIndeterminable;
        testResult.collisionTime = ts;
        return testResult;
    }

    Vec3 endPoint = _traj.GetValue(tf);
    if (obstacle->IsPointInside(endPoint))
    {
        // Recursively check both halves of the trajectory based on collision in the first half
        DetailedCollisionResult firstHalfResult;
        unsigned nextIterationNumber = iterationNumber + 1;
        firstHalfResult = polyCollisionChecker(ts, midTime, obstacle, minTimeSection, nextIterationNumber, _traj);
        if (firstHalfResult.collisionType != NoCollision)
            return firstHalfResult;
        else
        {
            // Recursively check the second half of the trajectory
            unsigned nextIterationNumber = iterationNumber + 1;
            return polyCollisionChecker(midTime, tf, obstacle, minTimeSection, nextIterationNumber, _traj);
        }
    }

    // Get the plane separating the midpoint and the obstacle
    CommonMath::Boundary tangentPlane = obstacle->GetTangentPlane(midpoint);

    // Take the dot product of the trajectory with the unit normal of the separating plane.
    // This gives the distance of the trajectory from the plane as a function of time
    double c[5] = {0, 0, 0, 0, 0};
    std::vector<Vec3> trajDerivativeCoeffs = _traj.GetDerivativeCoeffs();
    for (unsigned dim = 0; dim < 3; dim++)
    {
        c[0] += tangentPlane.normal[dim] * trajDerivativeCoeffs[0][dim]; // t**4
        c[1] += tangentPlane.normal[dim] * trajDerivativeCoeffs[1][dim]; // t**3
        c[2] += tangentPlane.normal[dim] * trajDerivativeCoeffs[2][dim]; // t**2
        c[3] += tangentPlane.normal[dim] * trajDerivativeCoeffs[3][dim]; // t
        c[4] += tangentPlane.normal[dim] * trajDerivativeCoeffs[4][dim]; // 1
    }

    // Solve the roots
    double roots[4];
    unsigned rootCount;
    if (fabs(c[0]) > double(1e-6))
        rootCount = Quartic::solve_quartic(c[1] / c[0], c[2] / c[0], c[3] / c[0], c[4] / c[0], roots);
    else
        rootCount = Quartic::solveP3(c[2] / c[1], c[3] / c[1], c[4] / c[1], roots);
    // The first "rootCount" entries of roots are now the unordered roots
    std::sort(roots, roots + rootCount);
    // The first "rootCount" entries of roots are now the roots in ascending order

    // Get both lists of points to check (in ascending order)
    std::vector<double> testPointsLow;
    std::vector<double> testPointsHigh;
    testPointsLow.reserve(6);
    testPointsHigh.reserve(6);
    testPointsLow.push_back(ts);       // ts is always the first critical point in testPointsLow
    testPointsHigh.push_back(midTime); // midTime is always the first critical point in testPointsHigh
    for (int i = 0; i < rootCount; i++)
    {
        if (roots[i] <= ts) // Skip root if it's before ts
            continue;
        else if (roots[i] < midTime) // Root is between ts and midTime
            testPointsLow.push_back(roots[i]);
        else if (roots[i] < tf) // Root is between midTime and tf
            testPointsHigh.push_back(roots[i]);
        else // Because the roots are in ascending order, there are no more roots are on (ts,tf)
            break;
    }
    testPointsLow.push_back(midTime); // midTime is always the last critical point in testPointsLow
    testPointsHigh.push_back(tf);     // tf is always the last critical point in testPointsHigh

    // Check testPointsLow first. If the collision already takes place in first half, we can ignore the second half and return the collision time.
    for (typename std::vector<double>::reverse_iterator it = testPointsLow.rbegin() + 1; it != testPointsLow.rend(); it++)
    {
        // Check whether the critical point occurs on the obstacle side of the plane
        if ((_traj.GetValue(*it) - tangentPlane.point).Dot(tangentPlane.normal) <= 0)
        {
            // This critical point is on the obstacle side of the plane, so we must
            // keep searching over the rest of the trajectory starting at the
            // previous critical point and ending at ts (recall we are searching
            // backwards in time).

            DetailedCollisionResult lowTestPointsResult;
            unsigned nextIterationNumber = iterationNumber + 1;
            lowTestPointsResult = polyCollisionChecker(ts, *(it - 1), obstacle, minTimeSection, iterationNumber, _traj);
            if (lowTestPointsResult.collisionType == NoCollision)
                break;
            else
                return lowTestPointsResult;
        }
    }
    for (typename std::vector<double>::iterator it = testPointsHigh.begin() + 1; it != testPointsHigh.end(); it++)
    {
        // Check whether the critical point occurs on the obstacle side of the plane
        if ((_traj.GetValue(*it) - tangentPlane.point).Dot(tangentPlane.normal) <= 0)
        {
            // This critical point is on the obstacle side of the plane, so we must
            // keep searching over the rest of the trajectory starting at the
            // previous critical point and ending at tf.
            DetailedCollisionResult highTestPointsResult;
            unsigned nextIterationNumber = iterationNumber + 1;
            highTestPointsResult = polyCollisionChecker(*(it - 1), tf, obstacle, minTimeSection, nextIterationNumber, _traj);
            if (highTestPointsResult.collisionType == NoCollision) // The section from the previous critical point until tf was feasible, meaning that all of the trajectory from midTime to tf does not collide with the obstacle
                break;
            else // Either a collision was detected between the previous critical point and tf, or the recursion became too deep (i.e. the time resolution too small) and the collision was indeterminable.
                return highTestPointsResult;
        }
    }
    // Both segments are free of collision

    testResult.collisionType = NoCollision;
    testResult.collisionTime = 100000; // A very large time to indicate no collision;
    return testResult;
}

// Fit the points to a fifth degree polynomial in order to use the UC Berkeley collision checker
Trajectory polyFit3D(std::vector<std::vector<double>> states, std::vector<double> tValues)
{ // state is a matrix of however many rows, but four columns (x, y, z, tF)
    Eigen::VectorXd yValues(states.size());
    Eigen::VectorXd xValues(states.size());
    Eigen::VectorXd xCoeffs(5);
    Eigen::VectorXd yCoeffs(5);
    Eigen::VectorXd zCoeffs(5);

    for (unsigned rows = 0; rows < states.size(); rows++)
    {
        xValues[rows] = states[rows][0];
        yValues[rows] = states[rows][1];
    }

    xCoeffs = polyfit_Eigen(tValues, xValues, 5);
    yCoeffs = polyfit_Eigen(tValues, yValues, 5);

    std::vector<Vec3> coeffs;
    // Change order of coefficients from 0->n to n->0
    for (int i = 5; i >= 0; i--)
        coeffs.push_back(Vec3(xCoeffs.coeffRef(i), yCoeffs.coeffRef(i), zCoeffs.coeffRef(i))); // Empty z coefficients

    // std::cout << "front t: " << tValues.front() << " back t: " << tValues.back() << std::endl;
    Trajectory traj(coeffs, tValues.front(), tValues.back());
    return traj;
}

double distanceFunc(ompl::base::State *state1, ompl::base::State *state2)
{
    double dist = 0;
    dist = sqrt(pow(state1->as<ompl::base::HybridStateSpace::StateType>()->as<ompl::base::RealVectorStateSpace::StateType>(0)->values[0] - state2->as<ompl::base::HybridStateSpace::StateType>()->as<ompl::base::RealVectorStateSpace::StateType>(0)->values[0], 2) + pow(state1->as<ompl::base::HybridStateSpace::StateType>()->as<ompl::base::RealVectorStateSpace::StateType>(0)->values[1] - state2->as<ompl::base::HybridStateSpace::StateType>()->as<ompl::base::RealVectorStateSpace::StateType>(0)->values[1], 2));
    return fabs(dist);
}

bool collisionregion1(double x1, double x2)
{ // True if collision
    return x1 <= 4.5 && x1 >= 0.5 && x2 >= 1 && x2 <= 1.5;
}

bool collisionregion2(double x1, double x2)
{ // True if collision
    return x1 <= 4.5 && x1 >= 0.5 && x2 >= 1.4 && x2 <= 1.5;
}

bool collisionregion3(double x1, double x2)
{ // True if collision
    return x1 <= 4.5 && x1 >= 0.5 && x2 >= 2.5 && x2 <= 3.0;
}

bool collisionregion4(double x1, double x2)
{ // True if collision
    return x1 <= 4.5 && x1 >= 0.5 && x2 >= 2.9 && x2 <= 3;
}

bool collisionregion5(double x1, double x2)
{ // True if collision
    return x1 <= 0.5 && x1 >= 0.0 && x2 >= 1.5 && x2 <= 2.5;
}

bool collisionregion6(double x1, double x2)
{ // True if collision
    return x1 <= 4.5 && x1 >= 4.4 && x2 >= 1 && x2 <= 1.5;
}

bool collisionregion7(double x1, double x2)
{ // True if collision
    return x1 <= 4.5 && x1 >= 4.4 && x2 >= 2.5 && x2 <= 3;
}

bool Xu(double x1, double x2)
{
    if (x1 <= 4.4 && x1 >= 0 && x2 >= 1.1 && x2 <= 1.4)
        return true;
    if (x1 <= 0.4 && x1 >= 0 && x2 >= 1.1 && x2 <= 2.9)
        return true;
    if (x1 <= 4.4 && x1 >= 0 && x2 >= 2.6 && x2 <= 2.9)
        return true;
    return false;
}

/** \brief Jump set is true whenever the multicopter is within the area of the c-shaped obstacle. */
bool jumpSet(ompl::control::HySST::Motion *motion)
{
    double x1 = motion->state->as<ompl::base::HybridStateSpace::StateType>()->as<ompl::base::RealVectorStateSpace::StateType>(0)->values[0];
    double x2 = motion->state->as<ompl::base::HybridStateSpace::StateType>()->as<ompl::base::RealVectorStateSpace::StateType>(0)->values[1];
    bool value = false;

    // Jump state
    if (Xu(x1, x2) || collisionregion1(x1, x2) || collisionregion2(x1, x2) || collisionregion3(x1, x2) || collisionregion4(x1, x2) || collisionregion5(x1, x2) || collisionregion6(x1, x2) || collisionregion7(x1, x2))
        value = true;

    return value;
}

/** \brief Flow set is true whenever the multicopter is outside of the area of the c-shaped obstacle. */
bool flowSet(ompl::control::HySST::Motion *motion)
{
    return !jumpSet(motion);
}

/** \brief Unsafe set is true whenever the multicopter is outside of the 6x7 rectangular planning space. */
bool unsafeSet(ompl::control::HySST::Motion *motion)
{
    std::vector<double> x_cur = {motion->state->as<ompl::base::HybridStateSpace::StateType>()->as<ompl::base::RealVectorStateSpace::StateType>(0)->values[0], motion->state->as<ompl::base::HybridStateSpace::StateType>()->as<ompl::base::RealVectorStateSpace::StateType>(0)->values[1], motion->state->as<ompl::base::HybridStateSpace::StateType>()->as<ompl::base::RealVectorStateSpace::StateType>(0)->values[2], motion->state->as<ompl::base::HybridStateSpace::StateType>()->as<ompl::base::RealVectorStateSpace::StateType>(0)->values[3]};
    if (x_cur[0] < 0.5 || x_cur[0] > 6 || x_cur[1] < 0 || x_cur[1] > 7 || Xu(x_cur[0], x_cur[1]))
        return true;
    return false;
}

/** \brief Simulates the dynamics of the multicopter when in flow regime, with no nonnegligble forces other than input acceleration. */
// ompl::base::State *continuousSimulator(std::vector<double> inputs, ompl::base::State *x_cur, double tFlow, ompl::base::State *new_state)
// {
//     double x1 = x_cur->as<ompl::base::HybridStateSpace::StateType>()->as<ompl::base::RealVectorStateSpace::StateType>(0)->values[0];
//     double x2 = x_cur->as<ompl::base::HybridStateSpace::StateType>()->as<ompl::base::RealVectorStateSpace::StateType>(0)->values[1];
//     double x3 = x_cur->as<ompl::base::HybridStateSpace::StateType>()->as<ompl::base::RealVectorStateSpace::StateType>(0)->values[2];
//     double x4 = x_cur->as<ompl::base::HybridStateSpace::StateType>()->as<ompl::base::RealVectorStateSpace::StateType>(0)->values[3];
//     double x5 = x_cur->as<ompl::base::HybridStateSpace::StateType>()->as<ompl::base::RealVectorStateSpace::StateType>(0)->values[4];
//     double x6 = x_cur->as<ompl::base::HybridStateSpace::StateType>()->as<ompl::base::RealVectorStateSpace::StateType>(0)->values[5];

//     x1 = x1 + x3 * tFlow + x5 * pow(tFlow, 2) / 2; // x = v0 * t + 1/2(at^2)
//     x2 = x2 + x4 * tFlow + x6 * pow(tFlow, 2) / 2; // x = v0 * t + 1/2(at^2)
//     x3 = x3 + (x5)*tFlow; // v = v0 + at
//     x4 = x4 + (x6)*tFlow;
//     x5 += inputs[0]*tFlow;
//     x6 += inputs[1]*tFlow; // Set control input to 0 for flow

//     new_state->as<ompl::base::HybridStateSpace::StateType>()->as<ompl::base::RealVectorStateSpace::StateType>(0)->values[0] = x1;
//     new_state->as<ompl::base::HybridStateSpace::StateType>()->as<ompl::base::RealVectorStateSpace::StateType>(0)->values[1] = x2;
//     new_state->as<ompl::base::HybridStateSpace::StateType>()->as<ompl::base::RealVectorStateSpace::StateType>(0)->values[2] = x3;
//     new_state->as<ompl::base::HybridStateSpace::StateType>()->as<ompl::base::RealVectorStateSpace::StateType>(0)->values[3] = x4;
//     new_state->as<ompl::base::HybridStateSpace::StateType>()->as<ompl::base::RealVectorStateSpace::StateType>(0)->values[4] = x5;
//     new_state->as<ompl::base::HybridStateSpace::StateType>()->as<ompl::base::RealVectorStateSpace::StateType>(0)->values[5] = x6;
//     return new_state;
// }

/** \brief Simulates the dynamics of the multicopter when in jump regime, with input from the surface. */
ompl::base::State *discreteSimulator(ompl::base::State *x_cur, const ompl::control::Control *u, ompl::base::State *new_state)
{
    double x1 = x_cur->as<ompl::base::HybridStateSpace::StateType>()->as<ompl::base::RealVectorStateSpace::StateType>(0)->values[0];
    double x2 = x_cur->as<ompl::base::HybridStateSpace::StateType>()->as<ompl::base::RealVectorStateSpace::StateType>(0)->values[1];
    double x3 = x_cur->as<ompl::base::HybridStateSpace::StateType>()->as<ompl::base::RealVectorStateSpace::StateType>(0)->values[2];
    double x4 = x_cur->as<ompl::base::HybridStateSpace::StateType>()->as<ompl::base::RealVectorStateSpace::StateType>(0)->values[3];

    double e = 0.43;
    double kappa = 0.20;
    double vn, vt, vnplus, vtplus;

    if (collisionregion1(x1, x2) || collisionregion2(x1, x2) || collisionregion3(x1, x2) || collisionregion4(x1, x2))
    {
        vn = x3;
        vt = x4;
        vnplus = -e * vn;
        vtplus = vt + kappa * (-e - 1) * std::atan(vt / vn) * vn;
        x3 = vnplus;
        x4 = vtplus;
    }
    else
    {
        vn = x4;
        vt = x3;
        vnplus = -e * vn;
        vtplus = vt + kappa * (-e - 1) * std::atan(vt / vn) * vn;
        x3 = vtplus;
        x4 = vnplus;
    }

    new_state->as<ompl::base::HybridStateSpace::StateType>()->as<ompl::base::RealVectorStateSpace::StateType>(0)->values[0] = x1;
    new_state->as<ompl::base::HybridStateSpace::StateType>()->as<ompl::base::RealVectorStateSpace::StateType>(0)->values[1] = x2;
    new_state->as<ompl::base::HybridStateSpace::StateType>()->as<ompl::base::RealVectorStateSpace::StateType>(0)->values[2] = x3;
    new_state->as<ompl::base::HybridStateSpace::StateType>()->as<ompl::base::RealVectorStateSpace::StateType>(0)->values[3] = x4;
    new_state->as<ompl::base::HybridStateSpace::StateType>()->as<ompl::base::RealVectorStateSpace::StateType>(0)->values[4] = 0;
    new_state->as<ompl::base::HybridStateSpace::StateType>()->as<ompl::base::RealVectorStateSpace::StateType>(0)->values[5] = 0;
    return new_state;
}

/** \brief Collision checker for the multicopter, courtesy of Berkeley Hybrid Systems Lab. */
bool collisionChecker(ompl::control::HySST::Motion *motion, std::function<bool(ompl::control::HySST::Motion *motion)> obstacleSet, ompl::base::State *new_state, double *collisionTime)
{
    double ts = motion->solutionPair->at(0)->as<ompl::base::HybridStateSpace::StateType>()->as<ompl::base::HybridTimeStateSpace::StateType>(1)->position;
    double tf = new_state->as<ompl::base::HybridStateSpace::StateType>()->as<ompl::base::HybridTimeStateSpace::StateType>(1)->position;
    std::vector<std::vector<double>> *propStepStatesDouble = new std::vector<std::vector<double>>();
    for (int i = 0; i < motion->solutionPair->size(); i++)
    {
        std::vector<double> row;
        for (int j = 0; j < 6; j++)
            row.push_back(motion->solutionPair->at(i)->as<ompl::base::HybridStateSpace::StateType>()->as<ompl::base::RealVectorStateSpace::StateType>(0)->values[j]);
        propStepStatesDouble->push_back(row);
    }

    std::vector<double> tValues;
    for (int i = 0; i < motion->solutionPair->size(); i++)
    {
        tValues.push_back(motion->solutionPair->at(i)->as<ompl::base::HybridStateSpace::StateType>()->as<ompl::base::HybridTimeStateSpace::StateType>(1)->position);
    }

    if(tValues.front() == tValues.back()) 
        return false;

    ts = tValues.front();
    tf = tValues.back();


    Trajectory _traj = polyFit3D(*propStepStatesDouble, tValues);

    DetailedCollisionResult leftCollisionResult = polyCollisionChecker(ts, tf, leftRectPrism, 1e-03, 0, _traj);
    DetailedCollisionResult topCollisionResult = polyCollisionChecker(ts, tf, topRectPrism, 1e-03, 0, _traj);
    DetailedCollisionResult bottomCollisionResult = polyCollisionChecker(ts, tf, bottomRectPrism, 1e-03, 0, _traj);

    DetailedCollisionResult trueCollisionResult;
    trueCollisionResult.collisionType == NoCollision;
    bool run = true;

    if (leftCollisionResult.collisionType == Collision)
        trueCollisionResult = leftCollisionResult;
    else if (topCollisionResult.collisionType == Collision)
        trueCollisionResult = topCollisionResult;
    else if (bottomCollisionResult.collisionType == Collision)
        trueCollisionResult = bottomCollisionResult;
    else
    {
        trueCollisionResult.collisionType == NoCollision;
        run = false;
    }

    bool collision = run && trueCollisionResult.collisionType == Collision;
    std::vector<double> collision_point;

    if (collision)
    {
        Vec3 collision_point = _traj.GetValue(trueCollisionResult.collisionTime);
        std::vector<Vec3> collision_vel_coeffs = _traj.GetDerivativeCoeffs();
        collision_vel_coeffs.insert(collision_vel_coeffs.begin(), Vec3(0.0, 0.0, 0.0));
        Trajectory _deriv_traj(collision_vel_coeffs, ts, tf);
        Vec3 vel_collision_point = _deriv_traj.GetValue(trueCollisionResult.collisionTime);

        // push back final motion with collision as last point
        new_state->as<ompl::base::HybridStateSpace::StateType>()->as<ompl::base::RealVectorStateSpace::StateType>(0)->values[0] = collision_point[0];
        new_state->as<ompl::base::HybridStateSpace::StateType>()->as<ompl::base::RealVectorStateSpace::StateType>(0)->values[1] = collision_point[1];
        new_state->as<ompl::base::HybridStateSpace::StateType>()->as<ompl::base::RealVectorStateSpace::StateType>(0)->values[2] = vel_collision_point[0];
        new_state->as<ompl::base::HybridStateSpace::StateType>()->as<ompl::base::RealVectorStateSpace::StateType>(0)->values[3] = vel_collision_point[1];
        *collisionTime = trueCollisionResult.collisionTime;
    }
    return collision && run;
}

void flowODE(const ompl::control::ODESolver::StateType& q, const ompl::control::Control* c, ompl::control::ODESolver::StateType& qdot)
{
    // Retrieve control values.  Velocity is the first entry, steering angle is second.
    const double *u = c->as<ompl::control::RealVectorControlSpace::ControlType>()->values;
    const double u_1 = u[0];
    const double u_2 = u[1];
 
    // Retrieve the current orientation of the multicopter.
    const double v_1 = q[2];
    const double v_2 = q[3];
    const double a_1 = q[4];
    const double a_2 = q[5];
 
    // Ensure qdot is the same size as q.  Zero out all values.
    qdot.resize(q.size(), 0);
 
    qdot[0] = v_1;
    qdot[1] = v_2;
    qdot[2] = a_1;
    qdot[3] = a_2;
    qdot[4] = u_1;
    qdot[5] = u_2;
}


int main()
{
    // std::uint_fast32_t seed = 7;
    // ompl::RNG::setSeed(seed);
    
    // Set the bounds of space
    ompl::base::RealVectorStateSpace *statespace = new ompl::base::RealVectorStateSpace(0);
    statespace->addDimension(0.5, 6.0);
    statespace->addDimension(0, 5);
    statespace->addDimension(-10, 10);
    statespace->addDimension(-10, 100000);
    statespace->addDimension(-10, 10);
    statespace->addDimension(-10, 10);

    ompl::base::StateSpacePtr stateSpacePtr(statespace);
    ompl::base::HybridStateSpace *hybridSpace = new ompl::base::HybridStateSpace(stateSpacePtr);
    ompl::base::StateSpacePtr hybridSpacePtr(hybridSpace);

    // Define control space
    ompl::control::RealVectorControlSpace *flowControlSpace = new ompl::control::RealVectorControlSpace(hybridSpacePtr, 2);
    ompl::control::RealVectorControlSpace *jumpControlSpace = new ompl::control::RealVectorControlSpace(hybridSpacePtr, 2);

    ompl::base::RealVectorBounds flowBounds(2);
    flowBounds.setLow(0, -0.5);
    flowBounds.setLow(1, -1);
    flowBounds.setHigh(0, 1);
    flowBounds.setHigh(1, 1);
    flowControlSpace->setBounds(flowBounds);

    ompl::base::RealVectorBounds jumpBounds(2);
    jumpBounds.setLow(0, 0);
    jumpBounds.setLow(0, 0);
    jumpBounds.setHigh(0, 0);
    jumpBounds.setHigh(0, 0);
    jumpControlSpace->setBounds(jumpBounds);

    ompl::control::RealVectorControlUniformSampler flowControlSampler(flowControlSpace);
    flowControlSpace->setControlSamplerAllocator([flowControlSpace](const ompl::control::ControlSpace *space) -> ompl::control::ControlSamplerPtr {
        return std::make_shared<ompl::control::RealVectorControlUniformSampler>(space);
    });

    ompl::control::RealVectorControlUniformSampler jumpControlSampler(jumpControlSpace);     // Doesn't do anything because the bounds for jump input are just [0, 0], but here for demonstration
    jumpControlSpace->setControlSamplerAllocator([jumpControlSpace](const ompl::control::ControlSpace *space) -> ompl::control::ControlSamplerPtr {
        return std::make_shared<ompl::control::RealVectorControlUniformSampler>(space);
    });

    ompl::control::ControlSpacePtr flowControlSpacePtr(flowControlSpace);
    ompl::control::ControlSpacePtr jumpControlSpacePtr(jumpControlSpace);

    ompl::control::CompoundControlSpace *controlSpace = new ompl::control::CompoundControlSpace(hybridSpacePtr);
    controlSpace->addSubspace(flowControlSpacePtr);
    controlSpace->addSubspace(jumpControlSpacePtr);

    ompl::control::ControlSpacePtr controlSpacePtr(controlSpace);

    // Construct a space information instance for this state space
    ompl::control::SpaceInformationPtr si(new ompl::control::SpaceInformation(hybridSpacePtr, controlSpacePtr));
    ompl::control::ODESolverPtr odeSolver (new ompl::control::ODEBasicSolver<> (si, &flowODE));
    
    si->setStatePropagator(ompl::control::ODESolver::getStatePropagator(odeSolver));
    si->setPropagationStepSize(0.01);
    si->setMinMaxControlDuration(1, 1);

    si->setup();

    // Set start state to be (1, 2)
    ompl::base::ScopedState<> start(hybridSpacePtr);
    start->as<ompl::base::HybridStateSpace::StateType>()->as<ompl::base::RealVectorStateSpace::StateType>(0)->values[0] = 1;
    start->as<ompl::base::HybridStateSpace::StateType>()->as<ompl::base::RealVectorStateSpace::StateType>(0)->values[1] = 2;
    start->as<ompl::base::HybridStateSpace::StateType>()->as<ompl::base::RealVectorStateSpace::StateType>(0)->values[2] = 0;
    start->as<ompl::base::HybridStateSpace::StateType>()->as<ompl::base::RealVectorStateSpace::StateType>(0)->values[3] = 0;
    start->as<ompl::base::HybridStateSpace::StateType>()->as<ompl::base::RealVectorStateSpace::StateType>(0)->values[4] = 0;
    start->as<ompl::base::HybridStateSpace::StateType>()->as<ompl::base::RealVectorStateSpace::StateType>(0)->values[5] = 0;

    // Set goal state to be (5, 4)
    ompl::base::ScopedState<> goal(hybridSpacePtr);
    goal->as<ompl::base::HybridStateSpace::StateType>()->as<ompl::base::RealVectorStateSpace::StateType>(0)->values[0] = 5;
    goal->as<ompl::base::HybridStateSpace::StateType>()->as<ompl::base::RealVectorStateSpace::StateType>(0)->values[1] = 2;

    // Create a problem instance
    ompl::base::ProblemDefinitionPtr pdef(new ompl::base::ProblemDefinition(si));

    // Set the start and goal states
    pdef->setStartAndGoalStates(start, goal);

    ompl::control::HySST cHySST(si);

    // Set parameters
    cHySST.setProblemDefinition(pdef);
    cHySST.setup();
    cHySST.setDistanceFunction(distanceFunc);
    // cHySST.setContinuousSimulator(continuousSimulator);
    cHySST.setDiscreteSimulator(discreteSimulator);
    cHySST.setFlowSet(flowSet);
    cHySST.setJumpSet(jumpSet);
    cHySST.setTm(0.5);
    cHySST.setFlowStepDuration(0.01);
    // cHySST.setFlowInputRange(std::vector<double>{-0.5, -1}, std::vector<double>{1, 1});
    // cHySST.setJumpInputRange(std::vector<double>{0, 0}, std::vector<double>{0, 0});
    cHySST.setUnsafeSet(unsafeSet);
    cHySST.setCollisionChecker(collisionChecker);
    cHySST.setSelectionRadius(0.03);
    cHySST.setPruningRadius(0.02);
    cHySST.setBatchSize(2);

    // attempt to solve the planning problem within 10 seconds
    ompl::base::PlannerStatus solved = cHySST.solve(ompl::base::timedPlannerTerminationCondition(10000000));
    std::cout << "solution status: " << solved << std::endl;

    // print path to RViz2 data file
    std::ofstream outFile("../../examples/visualize/src/points.txt");
    pdef->getSolutionPath()->as<ompl::control::PathControl>()->printAsMatrix(outFile);
}
