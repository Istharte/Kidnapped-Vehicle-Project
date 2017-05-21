/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 */

#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <math.h> 
#include <iostream>
#include <sstream>
#include <string>
#include <iterator>

#include "particle_filter.h"

using namespace std;

// Private methods
// extract landmark in sensor range.
void ExtractLandmark(double p_x, double p_y, std::vector<LandmarkObs> &sensor_landmarks, Map map_landmarks,
                      double sensor_range) {
    LandmarkObs single_landmark;
    
    for (int i = 0; i < map_landmarks.landmark_list.size(); i++)
    {
        if (dist(p_x, p_y, map_landmarks.landmark_list[i].x_f, 
                 map_landmarks.landmark_list[i].y_f) < sensor_range)
        {
            single_landmark.id = map_landmarks.landmark_list[i].id_i;
            single_landmark.x = map_landmarks.landmark_list[i].x_f;
            single_landmark.y = map_landmarks.landmark_list[i].y_f;
            sensor_landmarks.push_back(single_landmark);
        }
    }
}

// transform measurement to ground coordinate.
void TransformObs(std::vector<LandmarkObs> &observations, double p_x, double p_y, double p_theta) {
    double o_x;
    double o_y;
    vector<LandmarkObs> observation_temp;
    LandmarkObs single_landmark;
    observation_temp.clear();
    
    for (int i = 0; i < observations.size(); i++)
    {
        o_x = observations[i].x;
        o_y = observations[i].y;
        
        single_landmark.x = p_x + o_x * cos(p_theta) - o_y * sin(p_theta);
        single_landmark.y = p_y + o_x * sin(p_theta) + o_y * cos(p_theta);
        observation_temp.push_back(single_landmark);
    }
    
    observations = observation_temp;
}

// Calculate Gaussian.
inline double Gaussian(LandmarkObs predicted, double std_landmark[], LandmarkObs sensor) {
    double pow_x;
    double pow_y;
    
    pow_x = (predicted.x - sensor.x) * (predicted.x - sensor.x) / std_landmark[0] / std_landmark[0];
    pow_y = (predicted.y - sensor.y) * (predicted.y - sensor.y) / std_landmark[1] / std_landmark[1];
    
    return exp(-1.0 * sqrt(pow_x + pow_y) / 2.0) / (2.0 * M_PI * std_landmark[0] * std_landmark[1]);
    
}

// ParticleFilter methods
void ParticleFilter::init(double x, double y, double theta, double std[]) {
	// TODO: Set the number of particles. Initialize all particles to first position (based on estimates of 
	//   x, y, theta and their uncertainties from GPS) and all weights to 1. 
	// Add random Gaussian noise to each particle.
	// NOTE: Consult particle_filter.h for more information about this method (and others in this file).
    default_random_engine gen;
    
    // set the number of particles.
    num_particles = 1000;
    
    // Declare single particle.
	Particle one_particle;
    
    // creates a normal (Gaussian) distribution for x,y and theta.
	normal_distribution<double> dist_x(x, std[0]);
	normal_distribution<double> dist_y(y, std[1]);
	normal_distribution<double> dist_psi(theta, std[2]);

    // Initialize all particles to first position and all weights to 1.
    for (int i = 0; i<num_particles; i++)
    {
        one_particle.id = i+1;
        one_particle.x = dist_x(gen);
        one_particle.y = dist_y(gen);
        one_particle.theta = dist_psi(gen);
        one_particle.weight = 1.0;
        
        particles.push_back(one_particle);
        
        weights.push_back(1.0);
    }
    
    is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
	// TODO: Add measurements to each particle and add random Gaussian noise.
	// NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
	//  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
	//  http://www.cplusplus.com/reference/random/default_random_engine/
    default_random_engine gen;
    
    for (int i = 0; i < num_particles; i++)
    {
        double new_yaw = particles[i].theta + yaw_rate * delta_t;
        double v_by_y = velocity / yaw_rate;
        particles[i].x += v_by_y * (sin(new_yaw) - sin(particles[i].theta));
        particles[i].y += v_by_y * (cos(particles[i].theta) - cos(new_yaw));
        particles[i].theta = new_yaw;
        
        // creates a normal (Gaussian) distribution for x,y and theta.
        normal_distribution<double> dist_x(particles[i].x, std_pos[0]);
        normal_distribution<double> dist_y(particles[i].y, std_pos[1]);
        normal_distribution<double> dist_psi(particles[i].theta, std_pos[2]);
        
        // add random Gaussian noise.
        particles[i].x = dist_x(gen);
        particles[i].y = dist_y(gen);
        particles[i].theta = dist_psi(gen);
        
        particles[i].id = i+1;
    }

}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
	// TODO: Find the predicted measurement that is closest to each observed measurement and assign the 
	//   observed measurement to this particular landmark.
	// NOTE: this method will NOT be called by the grading code. But you will probably find it useful to 
	//   implement this method and use it as a helper during the updateWeights phase.
    double min_length;
    double now_length;
    
    // set temporaly observation vector.
    vector<LandmarkObs> observations_temp;
    
    int min_id = 0;
    LandmarkObs single_landmark;
    
    for (int i = 0; i < observations.size(); i++)
    {
        min_length = 50;
        
        for (int j = 0; j < predicted.size(); j++)
        {
            
            now_length = dist(predicted[j].x, predicted[j].y, observations[i].x, observations[i].y);
            
            if (min_length > now_length)
            {
                min_length = now_length;
                min_id = j;
            }
        }
        single_landmark.id = predicted[min_id].id;
        single_landmark.x = observations[i].x;
        single_landmark.y = observations[i].y;
        
        observations_temp.push_back(single_landmark);
        
    }
    observations.clear();
    observations = observations_temp;
    observations_temp.clear();
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
		std::vector<LandmarkObs> observations, Map map_landmarks) {
	// TODO: Update the weights of each particle using a mult-variate Gaussian distribution. You can read
	//   more about this distribution here: https://en.wikipedia.org/wiki/Multivariate_normal_distribution
	// NOTE: The observations are given in the VEHICLE'S coordinate system. Your particles are located
	//   according to the MAP'S coordinate system. You will need to transform between the two systems.
	//   Keep in mind that this transformation requires both rotation AND translation (but no scaling).
	//   The following is a good resource for the theory:
	//   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
	//   and the following is a good resource for the actual equation to implement (look at equation 
	//   3.33. Note that you'll need to switch the minus sign in that equation to a plus to account 
	//   for the fact that the map's y-axis actually points downwards.)
	//   http://planning.cs.uiuc.edu/node99.html
    
    // To store landmarks in sensor range.
    vector<LandmarkObs> sensor_landmarks;
    
    // store transformed observations.
    vector<LandmarkObs> observation_t;
    
    LandmarkObs single_landmark;
    
    double p_x;
    double p_y;
    double p_theta;
    
    vector<int> associations;
    vector<double> sense_x;
    vector<double> sense_y;
    
    for (int i = 0; i < num_particles; i++)
    {
        sensor_landmarks.clear();
        
        p_x = particles[i].x;
        p_y = particles[i].y;
        p_theta = particles[i].theta;
        
        // extract landmarks in sensor range.
        ExtractLandmark(p_x, p_y, sensor_landmarks, map_landmarks, sensor_range);
        
        // Transform observations.
        observation_t = observations;
        TransformObs(observation_t, p_x, p_y, p_theta);
        
        // associate observation to landmarks.
        dataAssociation(sensor_landmarks, observation_t);
        associations.clear();
        sense_x.clear();
        sense_y.clear();
        
        for (int m = 0; m < observation_t.size(); m++)
        {
            associations.push_back(observation_t[m].id);
            sense_x.push_back(observation_t[m].x);
            sense_y.push_back(observation_t[m].y);
        }
        
        particles[i] = SetAssociations(particles[i], associations, sense_x, sense_y);
        
        // reset weights
        particles[i].weight = 1.0;
        
        // update weight.
        for (int l = 0; l < observation_t.size(); l++)
        {
            for (int n = 0; n < sensor_landmarks.size(); n++)
            {
                if (sensor_landmarks[n].id == associations[l])
                {
                    single_landmark = sensor_landmarks[n];
                }
            }
            
            // Calculate weight.
            particles[i].weight = particles[i].weight * Gaussian(single_landmark, std_landmark, observation_t[l]);
            weights[i] = particles[i].weight;
            
        }
    }
}

void ParticleFilter::resample() {
	// TODO: Resample particles with replacement with probability proportional to their weight. 
	// NOTE: You may find std::discrete_distribution helpful here.
	//   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
    random_device rd;
    mt19937 gen(rd());
    discrete_distribution <int> d(weights.begin(), weights.end());
    
    // Set of resample temporary particles
    vector<Particle> particles_temp;
    
    double beta = 0.0;
    int index;
    
    for (int i = 0; i < num_particles; i++)
    {
        index = d(gen);
        particles_temp.push_back(particles[index]);
        
    }
    
    particles.clear();
    particles = particles_temp;
    particles_temp.clear();
    

}

Particle ParticleFilter::SetAssociations(Particle particle, std::vector<int> associations, std::vector<double> sense_x, std::vector<double> sense_y)
{
	//particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
	// associations: The landmark id that goes along with each listed association
	// sense_x: the associations x mapping already converted to world coordinates
	// sense_y: the associations y mapping already converted to world coordinates

	//Clear the previous associations
	particle.associations.clear();
	particle.sense_x.clear();
	particle.sense_y.clear();

	particle.associations= associations;
 	particle.sense_x = sense_x;
 	particle.sense_y = sense_y;

 	return particle;
}

string ParticleFilter::getAssociations(Particle best)
{
	vector<int> v = best.associations;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<int>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseX(Particle best)
{
	vector<double> v = best.sense_x;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseY(Particle best)
{
	vector<double> v = best.sense_y;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
