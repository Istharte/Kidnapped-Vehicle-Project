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
    
    int min_id;
    
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
        observations_temp.push_back(predicted[min_id]);
        cout<<"selected id: "<<observations_temp[i].id<<"\n";
        cout<<"obs,pred :" << observations[i].x<<", "<< observations[i].y<<", "<<predicted[min_id].x<<", "<<predicted[min_id].y<<"\n\n";
        
    }
    observations.clear();
    observations = observations_temp;
    observations_temp.clear();
    cout<<"end\n";
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
    vector<LandmarkObs> sensor_landmarks;
    
    LandmarkObs single_landmark;
    
    double p_x;
    double p_y;
    double p_theta;
    
    double o_x;
    double o_y;
    
    double pow_x;
    double pow_y;
    
    vector<int> associations;
    vector<double> sense_x;
    vector<double> sense_y;
    
    for (int i = 0; i < num_particles; i++)
    {
        sensor_landmarks.clear();
        
        p_x = particles[i].x;
        p_y = particles[i].y;
        p_theta = particles[i].theta;
        
        for (int j = 0; j < map_landmarks.landmark_list.size(); j++)
        {
            if (dist(p_x, p_y, map_landmarks.landmark_list[j].x_f, 
                     map_landmarks.landmark_list[j].y_f) < sensor_range)
            {
                single_landmark.id = map_landmarks.landmark_list[j].id_i;
                single_landmark.x = map_landmarks.landmark_list[j].x_f;
                single_landmark.y = map_landmarks.landmark_list[j].y_f;
                sensor_landmarks.push_back(single_landmark);
            }
        }
        
        // Transform observations.
        for (int k = 0; k < observations.size(); k++)
        {
            o_x = observations[k].x;
            o_y = observations[k].y;
            observations[k].x = p_x + o_x * cos(p_theta) + o_y * sin(p_theta);
            observations[k].y = p_y - o_x * sin(p_theta) + o_y * cos(p_theta);
        }
        
        dataAssociation(sensor_landmarks, observations);
        associations.clear();
        sense_x.clear();
        sense_y.clear();
        for (int m = 0; m < observations.size(); m++)
        {
            associations.push_back(observations[m].id);
            sense_x.push_back(observations[m].x);
            sense_y.push_back(observations[m].y);
        }
        
        particles[i] = SetAssociations(particles[i], associations, sense_x, sense_y);
        
        // update weight.
        for (int l = 0; l < observations.size(); l++)
        {
            for (int n = 0; n < sensor_landmarks.size(); n++)
            {
                if (sensor_landmarks[n].id == associations[l])
                {
                    single_landmark = sensor_landmarks[n];
                }
            }
            pow_x = pow(single_landmark.x - observations[l].x, 2.0) / pow(std_landmark[0], 2.0);
            pow_y = pow(single_landmark.y - observations[l].y, 2.0) / pow(std_landmark[1], 2.0);
            particles[i].weight = particles[i].weight * exp(-1.0/2.0*(pow_x + pow_y)
                                                            /sqrt(2.0 * M_PI * std_landmark[0] 
                                                            * std_landmark[1]));
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
    discrete_distribution <int> d(1, num_particles);
    uniform_real_distribution<double> dd(0.0, 1.0);
    
    // Set of resample temporary particles
    vector<Particle> particles_temp;
    
    double beta = 0.0;
    int index = d(gen);
    double max_weight = *max_element(weights.begin(), weights.end());
    
    for (int i = 0; i < num_particles; i++)
    {
        beta += dd(gen) * max_weight * 2;
        while (beta > particles[index].weight)
        {
            beta -= particles[index].weight;
            index = (index + 1) % num_particles;
        }
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
