import numpy as np
import matplotlib.pyplot as plt
import math
import json

# Set the parameters for the normal distribution of V0 - starting speed of bullet
num_samples = 250

mean_v = 680  # Mean of the distribution, according to AR-15 manuals we can define it as 680 m/s
std_dev_v = 1  # Standard deviation of the distribution

# Generate random values from the normal distribution
samples_v_normal = np.random.normal(mean_v, std_dev_v, num_samples)

# Set the parameters for the normal distribution of rifle barrel angle (radians)
mean_angle = 0.8  # Mean of the distribution in radians, according to common logic of fire works
std_dev_angle = 0.5  # Standard deviation of the distribution

# Generate random values from the normal distribution
samples_angle_normal = (np.random.normal(mean_angle, std_dev_angle, num_samples))

# Set the parameters for the uniform distribution of V0 - starting speed of bullet m/s
low_v = 500  # Lower bound of the distribution
high_v = 800  # Upper bound of the distribution

# Generate random values from the uniform distribution
samples_v_uniform = np.random.uniform(low_v, high_v, num_samples)

# Set the parameters for the uniform distribution of rifle barrel angle (radians)
low_angle  = 0.2  # Lower bound of the distribution
high_angle  = 1.2  # Upper bound of the distribution

# Generate random values from the uniform distribution
samples_angle_uniform = np.random.uniform(low_angle, high_angle, num_samples)

def generateJsonFile(filename, samples_v, samples_angle):
    pairs = []
    for i in range(len(samples_v)):
        pair = {'v': samples_v[i], 'angle': samples_angle[i]}
        pairs.append(pair)
    with open(filename, 'w') as json_file:
        json.dump(pairs, json_file)
generateJsonFile('pairs_v_normal_angle_normal.json', samples_v_normal, samples_angle_normal)
generateJsonFile('pairs_v_normal_angle_uniform.json', samples_v_normal, samples_angle_uniform)
generateJsonFile('pairs_v_uniform_angle_normal.json', samples_v_uniform, samples_angle_normal)
generateJsonFile('pairs_v_uniform_angle_uniform.json', samples_v_uniform, samples_angle_uniform)


def getVbyIndex(x, data):
    result = []
    for i in range(len(x)):
        result.append(data[i]['v'])
    return result

def getAnglebyIndex(x, data):
    result = []
    for i in range(len(x)):
        result.append(data[i]['angle'])
    return result

def calculateL(x, data):
    result = []
    for i in range(len(x)):
        result.append((data[i]['v']**2) * math.sin(2*data[i]['angle']) / 9.8)
    return result

def generateGraph(y, x_title, y_title, title, filename):
    plt.hist(y, bins='auto', density=True)
    plt.xlabel(x_title)
    plt.ylabel(y_title)
    plt.title(title)
    plt.grid(True)
    plt.savefig(filename)
    plt.close('all')

def executeScenario(vDistribution, aDistribution):
    with open('pairs_' + vDistribution + '_'+ aDistribution + '.json', 'r') as json_file:
        data = json.load(json_file)

    range_length = len(data)
    # The specified value
    x = list(range(1, range_length +1))
    y1 = getVbyIndex(x, data)
    y2 = np.round(getAnglebyIndex(x, data), 1)
    y3 = np.round(calculateL(x, data), 1)
    print(y2)
    print(y3)
    generateGraph(y1, 'm/s', 'probability', 'V distribution ' + vDistribution, vDistribution + '-distribution.png')
    generateGraph(y2, 'Radian', 'probability', 'Angle distribution '  + aDistribution, aDistribution + '-distribution.png')
    generateGraph(y3, 'm', 'probability', 'L distribution ' + vDistribution + ' - ' + aDistribution, vDistribution + '-' + aDistribution + '-distribution.png')




executeScenario('v_normal', 'angle_normal')
executeScenario('v_normal', 'angle_uniform')
executeScenario('v_uniform', 'angle_normal')
executeScenario('v_uniform', 'angle_uniform')
