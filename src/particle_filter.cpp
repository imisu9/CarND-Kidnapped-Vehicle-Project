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

#include "helper_functions.h"

using std::string;
using std::vector;

std::default_random_engine gen;

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
    weights.push_back(p.weight);
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
  
  // these lines create a normal (Gaussian) distribution noise for x, y, and theta
  std::normal_distribution<double> X_gaussian_init(0, std_pos[0]);
  std::normal_distribution<double> Y_gaussian_init(0, std_pos[1]);
  std::normal_distribution<double> Theta_gaussian_init(0, std_pos[2]);
  
  for (int i = 0; i < num_particles; ++i) {
    // calculate prediction
    if (std::abs(yaw_rate) > 0.00001) {
      particles[i].x += (velocity/yaw_rate) * 
        (sin(particles[i].theta + yaw_rate*delta_t) - sin(particles[i].theta));
      particles[i].y += (velocity/yaw_rate) *
        (cos(particles[i].theta) - cos(particles[i].theta + yaw_rate*delta_t));
      particles[i].theta += yaw_rate*delta_t;
    }
    else {
      particles[i].x += velocity * delta_t * cos(particles[i].theta);
      particles[i].y += velocity * delta_t * sin(particles[i].theta);
    }

    // add noise
    particles[i].x += X_gaussian_init(gen);
    particles[i].y += Y_gaussian_init(gen);
    particles[i].theta += Theta_gaussian_init(gen);
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
  for (unsigned i = 0; i < observations.size(); ++i) {
    double min_dist = std::numeric_limits<double>::max();
    for (unsigned j = 0; j < predicted.size(); ++j) {
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
    std::vector<int> associations;
    std::vector<double> sense_x;
    std::vector<double> sense_y;
    
    // create a vector for predicted landmark locations complying to the MAP's coordinate system.
    vector<LandmarkObs> predictions;
    for (unsigned j = 0; j < map_landmarks.landmark_list.size(); ++j) {
      int lmj_id = map_landmarks.landmark_list[j].id_i;
      float lmj_x = map_landmarks.landmark_list[j].x_f;
      float lmj_y = map_landmarks.landmark_list[j].y_f;
      
      // add j th landmark to predictions, 
      // only if it is within the sensor range from i th particle.
      if (dist(pi_x, pi_y, lmj_x, lmj_y) <= sensor_range) {
        predictions.push_back(LandmarkObs{lmj_id, lmj_x, lmj_y});
      }
    }
    
    // create a vector for observations transformed 
    // from the VEHICLE'S coordinate system to the MAP's coordinate system
    // by both rotation and translation
    vector<LandmarkObs> transformed_obs;
    for (unsigned k = 0; k < observations.size(); ++k) {
      int t_obs_id = observations[k].id;
      double t_obs_x = pi_x + (observations[k].x*cos(pi_theta)) - (observations[k].y*sin(pi_theta));
      double t_obs_y = pi_y + (observations[k].x*sin(pi_theta)) + (observations[k].y*cos(pi_theta));
      
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
    for (unsigned l = 0; l < transformed_obs.size(); ++l) {
      double t_ob_x = transformed_obs[l].x;
      double t_ob_y = transformed_obs[l].y;
      int landmark_id = transformed_obs[l].id;
      /*
      vector<LandmarkObs>::iterator it = std::find_if(predictions.begin(), predictions.end(),
                                                      [&landmark_id] (const LandmarkObs lm_ob) {return lm_ob.id == landmark_id;});
      */
      double p_x = 0.0;
      double p_y = 0.0;
      for (unsigned m = 0; m < predictions.size(); ++m) {
        if (predictions[m].id == landmark_id) {
          p_x = predictions[m].x;
          p_y = predictions[m].y;
          break;
        }
      }
      
      temp_weight = (1/(2*M_PI*std_landmark[0]*std_landmark[1])) *
                     exp((-1/2)*(pow(t_ob_x-p_x, 2)/pow(std_landmark[0], 2) +
                                 pow(t_ob_y-p_y, 2)/pow(std_landmark[1], 2)));
      if (temp_weight > 0) {
        particles[i].weight *= temp_weight;
      }
      associations.push_back(landmark_id);
      sense_x.push_back(t_ob_x);
      sense_y.push_back(t_ob_y);
    }
    weights[i] = particles[i].weight;
    SetAssociations(particles[i], associations, sense_x, sense_y);
  }
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
  
  // these lines create a normal (Gaussian) distribution noise for x, y, and theta
  std::uniform_int_distribution<int> Index_unidist(0, num_particles-1);
  int index = Index_unidist(gen);
  
  int max_weight = *std::max(weights.begin(), weights.end());
  std::uniform_real_distribution<double> Beta_unidist(0, max_weight);
  double beta = 0.0;
  
  for (int i = 0; i < num_particles; ++i) {
    beta += Beta_unidist(gen) * 2.0;
    while (beta > weights[index]) {
      beta -= weights[index];
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
