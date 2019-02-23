/**
 * particle_filter.cpp
 *
 * Created on: Dec 12, 2016
 * Author: Tiffany Huang
 */

#include "particle_filter.h"

#include <math.h>
#include <algorithm>
#include <iostream>
#include <iterator>
#include <numeric>
#include <random>
#include <string>
#include <vector>
#include <boost/bind.hpp>

#include "helper_functions.h"

using std::string;
using std::vector;



void ParticleFilter::init(double x, double y, double theta, double std[]) {
  /**
   * TODO: Set the number of particles. Initialize all particles to 
   *   first position (based on estimates of x, y, theta and their uncertainties
   *   from GPS) and all weights to 1. 
   * TODO: Add random Gaussian noise to each particle.
   * NOTE: Consult particle_filter.h for more information about this method 
   *   (and others in this file).
   */
  num_particles = 100;  // TODO: Set the number of particles
  std::default_random_engine gen;
  
  // these lines create a normal (Gaussian) distribution for x, y, and theta
  std::normal_distribution<double> X_gaussian_init(x, std[0]);
  std::normal_distribution<double> Y_gaussian_init(y, std[1]);
  std::normal_distribution<double> Theta_gaussian_init(theta, std[2]);
  
  // init particles
  for (int i = 0; i < num_particles; ++i) {
    Particle p;
    p.id = i;
    p.x = X_gaussian_init(gen);
    p.y = Y_gaussian_init(gen);
    p.theta = Theta_gaussian_init(gen);
    p.weight = 1.0;
    
    particles.push_back(p);
  }
  is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], 
                                double velocity, double yaw_rate) {
  /**
   * TODO: Add measurements to each particle and add random Gaussian noise.
   * NOTE: When adding noise you may find std::normal_distribution 
   *   and std::default_random_engine useful.
   *  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
   *  http://www.cplusplus.com/reference/random/default_random_engine/
   */
  std::default_random_engine gen;
  
  // these lines create a normal (Gaussian) distribution noise for x, y, and theta
  std::normal_distribution<double> X_gaussian_init(0, std_pos[0]);
  std::normal_distribution<double> Y_gaussian_init(0, std_pos[1]);
  std::normal_distribution<double> Theta_gaussian_init(0, std_pos[2]);
  
  for (int i = 0; i < num_particles; ++i) {
    // calculate prediction for x
    particles[i].x += (velocity/yaw_rate) * 
                      (sin(particles[i].theta + yaw_rate*delta_t) - sin(particles[i].theta));
    // add noise
    particles[i].x += X_gaussian_init(gen);
    // calculate prediction for y
    particles[i].y += (velocity/yaw_rate) * 
                      (cos(particles[i].theta) - cos(particles[i].theta + yaw_rate*delta_t));
    // add noise
    particles[i].y += Y_gaussian_init(gen);
    // calculate prediction and noise for theta
    particles[i].y += yaw_rate*delta_t + Theta_gaussian_init(gen);
  }
}

void ParticleFilter::dataAssociation(vector<LandmarkObs> predicted, 
                                     vector<LandmarkObs>& observations) {
  /**
   * TODO: Find the predicted measurement that is closest to each 
   *   observed measurement and assign the observed measurement to this 
   *   particular landmark.
   * NOTE: this method will NOT be called by the grading code. But you will 
   *   probably find it useful to implement this method and use it as a helper 
   *   during the updateWeights phase.
   */
  double min_dist = std::numeric_limits<double>::max();;
  for (int i = 0; i < observations.size(); ++i) {
    for (int j = 0; j < predicted.size(); ++j) {
      double curr_dist = dist(observations[i].x, observations[i].y, predicted[j].x, predicted[j].y);
      if (curr_dist < min_dist) {
        min_dist = curr_dist;
        observations[i].id = predicted[j].id;
      }
    }
  }
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
                                   const vector<LandmarkObs> &observations, 
                                   const Map &map_landmarks) {
  /**
   * TODO: Update the weights of each particle using a mult-variate Gaussian 
   *   distribution. You can read more about this distribution here: 
   *   https://en.wikipedia.org/wiki/Multivariate_normal_distribution
   * NOTE: The observations are given in the VEHICLE'S coordinate system. 
   *   Your particles are located according to the MAP'S coordinate system. 
   *   You will need to transform between the two systems. Keep in mind that
   *   this transformation requires both rotation AND translation (but no scaling).
   *   The following is a good resource for the theory:
   *   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
   *   and the following is a good resource for the actual equation to implement
   *   (look at equation 3.33) http://planning.cs.uiuc.edu/node99.html
   */
  for (int i = 0; i < num_particles; ++i) {
    double pi_x = particles[i].x;
    double pi_y = particles[i].y;
    double pi_theta = particles[i].theta;
    
    // create a vector for predicted landmark locations complying to the MAP's coordinate system.
    vector<LandmarkObs> predictions;
    
    for (int j = 0; j < map_landmarks.landmark_list.size(); ++j) {
      float lmj_id = map_landmarks.landmark_list[j].id_i;
      float lmj_x = map_landmarks.landmark_list[j].x_f;
      float lmj_y = map_landmarks.landmark_list[j].y_f;
      
      // add j th landmark to predictions, 
      // only if it is within the sensor range from i th particle.
      if (dist(pi_x, lmj_x, pi_y, lmj_y) <= sensor_range) {
        predictions.push_back(LandmarkObs{lmj_id, lmj_x, lmj_y});
      }
    }
    
    // create a vector for observations transformed 
    // from the VEHICLE'S coordinate system to the MAP's coordinate system
    // by both rotation and translation
    vector<LandmarkObs> transformed_obs;
    
    for (int k = 0; k < observations.size(); ++k) {
      int t_obs_id = observations[k].id;
      double t_obs_x = observations[k].x * cos(pi_theta) - observations[k].y * sin(pi_theta) + pi_x;
      double t_obs_y = observations[k].x * sin(pi_theta) + observations[k].y * cos(pi_theta) + pi_y;
      
      transformed_obs.push_back(LandmarkObs{t_obs_id, t_obs_x, t_obs_y});
    }
    
    // run dataAssociation() to find which observations correspond to which landmarks
    // (likely by using a nearest-neighbors data association).
    dataAssociation(predictions, transformed_obs);
    
    // since we are 2-dimensional, it is bivariate case.
    // the formular was taken from wikipeida at
    // https://en.wikipedia.org/wiki/Multivariate_normal_distribution.
    // (assuming correlation between x and y is zero)
    double temp_weight = 1;
    for (int l = 0; l < transformed_obs.size(); ++l) {
      double t_ob_x = transformed_obs[l].x;
      double t_ob_y = transformed_obs[l].y;
      vector<LandmarkObs>::iterator it = find(predictions.begin(), predictions.end(),
                                              boost::bind(&LandmarkObs::id, _1) == transformed_obs[l].id);
      double p_x = it->x;
      double p_y = it->y;
      
      temp_weight *= (1/(2*M_PI*std_landmark[0]*std_landmark[1]) * 
                      exp((-1/2)*(pow(t_ob_x-p_x, 2)/pow(std_landmark[0], 2) + 
                                  pow(t_ob_y-p_y, 2)/pow(std_landmark[1], 2) -
                                  2*(t_ob_x-p_x)*(t_ob_y-p_y)/(std_landmark[0]*std_landmark[1]))));
    }
    // update the weight of i th particle using a bivariate Gaussian distribution
    particles[i].weight = temp_weight;
}

void ParticleFilter::resample() {
  /**
   * TODO: Resample particles with replacement with probability proportional 
   *   to their weight. 
   * NOTE: You may find std::discrete_distribution helpful here.
   *   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
   */
  // Set of temp particles
  std::vector<Particle> temp_particles;
  std::default_random_engine gen;
  
  // these lines create a normal (Gaussian) distribution noise for x, y, and theta
  std::normal_distribution<int> Index_gaussian_init(0, num_particles-1);
  int index = Index_gaussian_init(gen);
  auto minmax_weight = std::minmax_element(particles.begin(), particles.end(),
                                           [] (Particle const& lhs, Size const& rhs) {
                                             return lhs.weight < rhs.weight;});
  int max_weight = minmax_weight.second->weight;
  double beta = 0;
  std::normal_distribution<double> Beta_gaussian_init(0, max_weight);
  
  for (int i = 0; i < num_particles; ++i) {
    beta += Beta_gaussian_init(gen) * 2;
    while (beta > particles[index].weight) {
      beta -= particles[index].weight;
      index = (index + 1) % num_particles;
    }
    temp_particles.push_back(particles[index]);
  }
  particles = temp_particles;
}

void ParticleFilter::SetAssociations(Particle& particle, 
                                     const vector<int>& associations, 
                                     const vector<double>& sense_x, 
                                     const vector<double>& sense_y) {
  // particle: the particle to which assign each listed association, 
  //   and association's (x,y) world coordinates mapping
  // associations: The landmark id that goes along with each listed association
  // sense_x: the associations x mapping already converted to world coordinates
  // sense_y: the associations y mapping already converted to world coordinates
  particle.associations= associations;
  particle.sense_x = sense_x;
  particle.sense_y = sense_y;
}

string ParticleFilter::getAssociations(Particle best) {
  vector<int> v = best.associations;
  std::stringstream ss;
  copy(v.begin(), v.end(), std::ostream_iterator<int>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length()-1);  // get rid of the trailing space
  return s;
}

string ParticleFilter::getSenseCoord(Particle best, string coord) {
  vector<double> v;

  if (coord == "X") {
    v = best.sense_x;
  } else {
    v = best.sense_y;
  }

  std::stringstream ss;
  copy(v.begin(), v.end(), std::ostream_iterator<float>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length()-1);  // get rid of the trailing space
  return s;
}
